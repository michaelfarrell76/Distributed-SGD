from __future__ import print_function
from __future__ import absolute_import
from grpc.beta import implementations
import time

import dist_sgd_pb2
import argparse

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad

from nnet.neural_net import *
from protobuf_utils.utils import * 

_TIMEOUT_SECONDS = 6000

def run(client_id):
	# Load and process Caltech data
	train_images, train_labels, test_images, test_labels = load_caltech100()
	image_input_d = train_images.shape[1]

    # Network parameters
	layer_sizes = [image_input_d, 800, 600, 400, 350, 250, 101]

	L2_reg = 1.0

	# Training parameters
	param_scale = 0.1
	learning_rate = 1e-3
	momentum = 0.9
	batch_size = 256
	num_epochs = 50

	# Make neural net functions
	N_weights, pred_fun, loss_fun, frac_err = make_nn_funs(layer_sizes, L2_reg)
	loss_grad = grad(loss_fun)

	# Train with sgd
	batch_idxs = make_batches(train_images.shape[0], batch_size)
	cur_dir = np.zeros(N_weights)

	       
	channel = implementations.insecure_channel('localhost', 50051)
	stub = dist_sgd_pb2.beta_create_ParamFeeder_stub(channel)

	prev_data_indx = -1

	print('Data loaded and connected to server:')
	
	try:
		# prev_data_indx of -2 means failure, should probably change this to a different value / more specific field
		response = stub.SendNextBatch(dist_sgd_pb2.PrevBatch(client_id=client_id, prev_data_indx=prev_data_indx), _TIMEOUT_SECONDS)
		while response.data_indx != -2:
			# Keep on trying to get your first batch
			while response.data_indx == -1:
				time.sleep(5)
				print('Waiting for server to send next batch')
				response = stub.SendNextBatch(dist_sgd_pb2.PrevBatch(client_id=client_id, prev_data_indx=prev_data_indx), _TIMEOUT_SECONDS)
			print('Processing parameters in batch %d!' % response.data_indx)

			# Generates the W matrix 
			get_parameters_time = time.time()
			W_bytes = ''
			W_subtensors_iter = stub.SendParams(dist_sgd_pb2.ClientInfo(client_id=client_id), _TIMEOUT_SECONDS)
			for W_subtensor_pb in W_subtensors_iter:
				# TODO: Error checking of some sort in here
				# SOME ERROR IS THROWN HERE
				W_bytes = W_bytes + W_subtensor_pb.tensor_content
			W = convert_bytes_to_array(W_bytes)
			print('Received parameters in {0:.2f}s'.format(time.time() - get_parameters_time))

			# Calculate the gradients
			grad_start = time.time()
			grad_W = loss_grad(W, train_images[batch_idxs[response.data_indx]], train_labels[batch_idxs[response.data_indx]])
			print('Done calculating gradients in {0:.2f}s'.format(time.time() - grad_start))
			
			# Serialize the gradients
			tensor_compress_start = time.time()
			tensor_bytes = convert_array_to_bytes(grad_W)
			tensor_iterator = convert_tensor_iter(tensor_bytes, response.data_indx)
			print('Done compressing gradients in {0:.2f}s'.format(time.time() - tensor_compress_start))

			# Send the gradients
			send_grad_start = time.time()
			stub.GetUpdates(tensor_iterator, _TIMEOUT_SECONDS) 
			print('Done sending gradients through in {0:.2f}s'.format(time.time() - send_grad_start))

			# Get the next batch to process
			prev_data_indx = response.data_indx
			response = stub.SendNextBatch(dist_sgd_pb2.PrevBatch(client_id=client_id, prev_data_indx=prev_data_indx), _TIMEOUT_SECONDS)

	except KeyboardInterrupt:
		pass
  	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--id')
	args = parser.parse_args()
	arg_id = int(args.id)
	assert(arg_id > 0)
	# TODO: Client id should not need to be passed later
	run(arg_id)