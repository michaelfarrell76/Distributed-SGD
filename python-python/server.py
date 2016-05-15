# ------------------------------------------------------------
# Implements a parameter server. The server takes parameter updates in and
# sends back the most up to date parameters. This server also keeps track of 
# the current training/test error.  
# ------------------------------------------------------------

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
from server_utils.utils import * 

import traceback

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

_REQUIRED_CHILDREN = 1

# Data files that we are training from. This is the small demo set. 
images_fname = 'data/images(64).npy'
labels_fname = 'data/output_labels(64).npy'

class ParamFeeder(dist_sgd_pb2.BetaParamFeederServicer):
    def __init__(self, W = None, prevBatch=None):
        # Keeps track of all child IDs that it has seen so far
        self.child_ids = Set([])

        # Load and process Caltech data
        self.train_images, self.train_labels, self.test_images, self.test_labels = load_caltech100(images_fname, labels_fname)
        self.image_input_d = self.train_images.shape[1]

        # Network parameters
        self.layer_sizes = [self.image_input_d, 800, 600, 400, 350, 250, 101]

        # Training parameters
        self.param_scale = 0.1
        self.learning_rate = 1e-5
        self.momentum = 0.9
        self.batch_size = 256
        self.num_epochs = 50
        self.L2_reg = 1.0

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

        # Set the current batch to zero unless it has been passed in
        self.epoch = 0
        if prevBatch is None:
            self.batch_num = 0
        else:
            self.batch_num = prevBatch
        self.n_batches = len(self.batch_idxs)

        # Initialize information about the clients
        self.n_childs = 0
        self.max_client_id = 0

        # Intializes starting information about training 
        self.prev_test_perf = 1

        # The batches that are currently being processed
        self.batches_processing = {}

        # The batches that were failed to process, model training machine may have failed
        # Send these batches to a new machine
        self.batches_unprocessed = []

        log_info('Data loaded on server, waiting for clients....')
        log_info('Number of child processes: 0')

    # Logs the current performance of the model. Called once per epoch.
    def log_info_perf(self, epoch):
        test_perf  = self.frac_err(self.W, self.test_images, self.test_labels)
        train_perf = self.frac_err(self.W, self.train_images, self.train_labels)
        if test_perf > self.prev_test_perf:
            self.learning_rate = 0.1 * self.learning_rate
        self.prev_test_perf = test_perf
        log_info("Epoch {0}, TrainErr {1:5}, TestErr {2:5}, LR {3:2}".format(self.epoch, train_perf, test_perf, self.learning_rate))

    # Streams updates from the client.
    def GetUpdates(self, request_iterator, context):
        tensor_bytes = ''  
        for subtensor in request_iterator:
            tensor_bytes = tensor_bytes + subtensor.tensor_content

        # Serialize the tensor
        grad_W = convert_bytes_to_array(tensor_bytes)

        # Gradient descent
        self.W -= 0.5 * self.learning_rate * grad_W

        return dist_sgd_pb2.StatusCode(status=1)

    # Sends the next batch that the client should process
    def SendNextBatch(self, request, context):
        # Figure out what the maximum client_id is. If client_id does not exist, 
        # assigns the client a new id. 
        if request.client_id == 0:
            self.max_client_id += 1
            request.client_id = self.max_client_id
        else:
            self.max_client_id = max(request.client_id, self.max_client_id)

        # Does not start until a sufficient number of child processes exists
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

        # Save the time that the next batch was sent out on the server
        self.batches_processing[cur_batchnum] = time.time()

        return dist_sgd_pb2.NextBatch(client_id=request.client_id, data_indx = cur_batchnum)

    # This sends the parameters from the server to the client by converting the tensor into a 
    # protobuffer and streaming it 
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

    # Function to ping the server to see if it is available
    def ping(self, request, context):
        return dist_sgd_pb2.empty() 

# Main function that is called to instantiate the server and have 
# it connect and send or receieve parameters from clients.
def serve(hostname, W = None, prev_batch = None, local_id = None):
    # Set up the server on port 50051
    hostname = '[::]:50051'
    BATCH_TRAIN_TIMEOUT = 60

    # Instantiate the server and add the port
    param_feeder = ParamFeeder(W, prev_batch)
    server = dist_sgd_pb2.beta_create_ParamFeeder_server(param_feeder)
    server.add_insecure_port(hostname)

    # Begin the server 
    server.start()
    try:
        while True:
            time.sleep(BATCH_TRAIN_TIMEOUT)

    except KeyboardInterrupt:
        server.stop(0)
        raise KeyboardInterrupt

if __name__ == '__main__':
    serve('[::]:50051')