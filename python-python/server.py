from __future__ import absolute_import
from __future__ import print_function
import time

import dist_sgd_pb2
from sets import Set

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad

from nnet.neural_net import *
from protobuf_utils.utils import * 

import traceback

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

_REQUIRED_CHILDREN = 1

images_fname = 'data/images(64).npy'
labels_fname = 'data/output_labels(64).npy'

def log_info(value):
    print(str(time.time()) + ' ' + str(value))

# import bpdb; bpdb.set_trace()
class ParamFeeder(dist_sgd_pb2.BetaParamFeederServicer):
    def __init__(self, W = None, prevBatch=None):
        self.child_ids = Set([])
        # Load and process Caltech data
        self.train_images, self.train_labels, self.test_images, self.test_labels = load_caltech100(images_fname, labels_fname)
        self.image_input_d = self.train_images.shape[1]

        # Network parameters
        self.layer_sizes = [self.image_input_d, 800, 600, 400, 350, 250, 101]
        # self.layer_sizes = [self.image_input_d, 200, 180, 150, 120, 101]

        self.L2_reg = 1.0

        # Training parameters
        self.param_scale = 0.1
        self.learning_rate = 1e-5
        self.momentum = 0.9
        self.batch_size = 256
        self.num_epochs = 50

        # Make neural net functions
        self.N_weights, self.pred_fun, self.loss_fun, self.frac_err = make_nn_funs(self.layer_sizes, self.L2_reg)
        self.loss_grad = grad(self.loss_fun)

        # Initialize weights
        if W is None:
            rs = npr.RandomState()
            self.W = rs.randn(self.N_weights) * self.param_scale
        else:
            # Passed in weights
            self.W = W

        self.param_len = self.W.shape[0]
        log_info("# of parameters:")
        log_info(self.param_len)

        # Train with sgd
        self.batch_idxs = make_batches(self.train_images.shape[0], self.batch_size)        

        self.cur_dir = np.zeros(self.N_weights).astype(np.float32)

        self.epoch = 0
        if prevBatch is None:
            self.batch_num = 0
        else:
            self.batch_num = prevBatch

        self.n_batches = len(self.batch_idxs)
        self.n_childs = 0
        self.max_client_id = 0
        self.prev_test_perf = 1

        # The batches that are currently being processed
        self.batches_processing = {}

        # The batches that were failed to process, model training machine may have failed
        # Send these batches to a new machine
        self.batches_unprocessed = []

        log_info('Data loaded on server, waiting for clients....')
        log_info('Number of child processes: 0')

    def log_info_perf(self, epoch):
        test_perf  = self.frac_err(self.W, self.test_images, self.test_labels)
        train_perf = self.frac_err(self.W, self.train_images, self.train_labels)
        if test_perf > self.prev_test_perf:
            self.learning_rate = 0.1 * self.learning_rate
        self.prev_test_perf = test_perf
        log_info("Epoch {0}, TrainErr {1:5}, TestErr {2:5}, LR {3:2}".format(self.epoch, train_perf, test_perf, self.learning_rate))

    # TODO: Any batches that are taking too long are removed from batches_processing and added to batches_unprocessed
    # def clean_unprocessed(self):

    def ping(self, request, context):
        return dist_sgd_pb2.empty() 

    def GetUpdates(self, request_iterator, context):
        # CHECK TO SEE IF THE REQUEST IS STALE AND SHOULD BE THROWN OUT, INDICATED BY CLIENT_ID
        tensor_bytes = ''  
        for subtensor in request_iterator:
            tensor_bytes = tensor_bytes + subtensor.tensor_content

        grad_W = convert_bytes_to_array(tensor_bytes)
        # TODO: Should do some checks with how long the tensor is, fail if its incorrect size
        # Throw error and return status=0 then

        # Gradient descent with momentum
        # self.cur_dir = self.momentum * self.cur_dir + (1.0 - self.momentum) * grad_W
        # self.W -= 0.5 * self.learning_rate * self.cur_dir
        
        # Basic gradient descent
        self.W -= 0.5 * self.learning_rate * grad_W

        return dist_sgd_pb2.StatusCode(status=1)

    def SendNextBatch(self, request, context):
        # Does not start until a sufficient number of child processes exists
        if request.client_id == 0:
            self.max_client_id += 1
            request.client_id = self.max_client_id
        else:
            self.max_client_id = max(request.client_id, self.max_client_id)

        self.child_ids.add(request.client_id)
        if len(self.child_ids) != self.n_childs:
            self.n_childs = len(self.child_ids)
            log_info('Number of child processes: ' + str(len(self.child_ids)))
        if len(self.child_ids) < _REQUIRED_CHILDREN:
            return dist_sgd_pb2.NextBatch(client_id=request.client_id, data_indx = -1)

        # Logs information about previous batch timing
        if request.prev_data_indx != -1:
            log_info('Time taken to process batch {0} was {1:.2f} by client {2}'.format(request.prev_data_indx, (time.time() - self.batches_processing[request.prev_data_indx]), request.client_id))
            del self.batches_processing[request.prev_data_indx]

        # log_info epoch information if we've hit the end of an epoch
        if self.batch_num == self.n_batches:
            self.batch_num, self.epoch = 0, self.epoch + 1
            self.log_info_perf(self.epoch)

        # Takes any previously failed batches first, otherwise takes next batch
        if self.batches_unprocessed != []:
            cur_batchnum = self.batches_unprocessed.pop(0)
        else:
            cur_batchnum, self.batch_num =  self.batch_num, self.batch_num + 1

        # log_info('Telling client %d to process batch %d' % (request.client_id, cur_batchnum))
        # Should probably also add in the client_id as a key for this
        self.batches_processing[cur_batchnum] = time.time()

        return dist_sgd_pb2.NextBatch(client_id=request.client_id, data_indx = cur_batchnum)

    def SendParams(self, request, context):
        CHUNK_SIZE = 524228
        tensor_bytes = convert_array_to_bytes(self.W)
        tensor_bytes_len = len(tensor_bytes)
        tensor_chunk_count = 0
        try:
            while len(tensor_bytes):
                tensor_chunk_count += 1
                tensor_content = tensor_bytes[:CHUNK_SIZE]
                tensor_bytes = tensor_bytes[CHUNK_SIZE:]
                yield dist_sgd_pb2.SubTensor(tensor_len = tensor_bytes_len, tensor_chunk = tensor_chunk_count, tensor_content = tensor_content, data_indx= -1)
        except Exception, e:
            traceback.print_exc()

def serve(hostname, W = None, prev_batch = None, local_id = None):
    hostname = '[::]:50051'

    # Allow argument that allows this parameter to be changed
    BATCH_TRAIN_TIMEOUT = 60
    param_feeder = ParamFeeder(W, prev_batch)
    server = dist_sgd_pb2.beta_create_ParamFeeder_server(param_feeder)
    server.add_insecure_port(hostname)
    server.start()
    try:
        while True:
            time.sleep(BATCH_TRAIN_TIMEOUT)
            # param_feeder.clean_unprocessed()

    except KeyboardInterrupt:
        server.stop(0)
        raise KeyboardInterrupt

if __name__ == '__main__':
    serve('[::]:50051')