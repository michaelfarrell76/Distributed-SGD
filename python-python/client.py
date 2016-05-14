from __future__ import print_function
from __future__ import absolute_import
from grpc.beta import implementations
import time
import sys

import dist_sgd_pb2
import argparse
import traceback

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad

from nnet.neural_net import *
from protobuf_utils.utils import * 

from server import serve
from paxos import run_paxos
import subprocess

images_fname = 'data/images(64).npy'
labels_fname = 'data/output_labels(64).npy'

_TIMEOUT_SECONDS = 20
TENSOR_TIMEOUT_SECONDS = 60
SERVER_PORT = 50051

def gen_local_address(local_id):
	if local_id is None:
		addr = subprocess.check_output("ip addr show eth0 | grep 'inet' | cut -d ' ' -f8", shell=True)
		return addr[:-1]
	else:
		server_addresses = gen_server_addresses(local_id)
		return server_addresses[local_id - 1]

def gen_server_addresses(local_id, local_address=None):
	if local_id is None:
		names, ips = [], []
		# Ugly formatting when directly using pipes, using files instead
		with open('ips.txt', 'w') as f:
		    subprocess.call(["gcloud", "compute", "instances", "list"], stdout=f)
		with open('ips.txt', 'r') as f:
		    lines = f.readlines()
		    for line in lines[1:]:
		        line_arr = filter((lambda x: x != '') , line.split(' '))
		       	names.append(line_arr[0])
		        ips.append(line_arr[3])
		        if line_arr[3] == local_address:
		        	local_name = line_arr[0]
		instance_ips = []
		local_name_arr = local_name.split('-')
		for i in range(len(ips)):
			name_arr = names[i].split('-')
			if name_arr[0] == local_name_arr[0] and name_arr[1] == local_name_arr[1]:
				instance_ips.append(ips[i])
		return instance_ips
	if local_id is not None:
		return ['[::]:50052', '[::]:50053', '[::]:50044']

def find_server(local_id=None):
	TOT_ATTEMPTS = 1
	for i in range(TOT_ATTEMPTS):
		local_address = gen_local_address(local_id)
		server_addresses = gen_server_addresses(local_id, local_address)
		server_addresses.remove(local_address)
		for server_address in server_addresses:
			if local_id is not None:
				channel = implementations.insecure_channel('localhost', SERVER_PORT)
			else:
				channel = implementations.insecure_channel(server_address, SERVER_PORT)
			stub = dist_sgd_pb2.beta_create_ParamFeeder_stub(channel)
			try:
				response = stub.ping(dist_sgd_pb2.empty(), _TIMEOUT_SECONDS)
				return server_address
			except Exception as e:
				if ('ExpirationError' in str(e) or 'NetworkError' in str(e)):
					continue
				else:
					# More severe error, should log and crash
					traceback.print_exc()
					sys.exit(1)
		time.sleep(1 * TOT_ATTEMPTS)
	return ''

def connect_server_stub(server_addr, local_id):
	if local_id is not None:
		channel = implementations.insecure_channel('localhost', SERVER_PORT)
	else:
		channel = implementations.insecure_channel(server_addr, SERVER_PORT)
	stub = dist_sgd_pb2.beta_create_ParamFeeder_stub(channel)
	return stub

def run(local_id = None):
	# Load and process Caltech data
	train_images, train_labels, test_images, test_labels = load_caltech100(images_fname, labels_fname)
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

	prev_data_indx = -1

	consec_expiration = 0

	server_addr = ''
	while server_addr == '':
		server_addr = run_paxos(local_id)
		if server_addr == '':
			server_addr = find_server(local_id)
	print('Server address is ' + server_addr)

	if server_addr == gen_local_address(local_id):
		print('Transforming into the server')
		try:
			serve(server_addr, None, prev_data_indx, local_id)
		except KeyboardInterrupt as e:
			print('interrupted')
			sys.exit(0)
		return

	stub = connect_server_stub(server_addr, local_id)
	client_id = 0

	print('Data loaded and connected to server:')
	
	# prev_data_indx of -2 means failure, should probably change this to a different value / more specific field
	try:
		response = stub.SendNextBatch(dist_sgd_pb2.PrevBatch(client_id=client_id, prev_data_indx=prev_data_indx), _TIMEOUT_SECONDS)
		while response.data_indx != -2:
			client_id = response.client_id
			# Keep on trying to get your first batch
			while response.data_indx == -1:
				time.sleep(5)
				print('Waiting for server to send next batch')
				response = stub.SendNextBatch(dist_sgd_pb2.PrevBatch(client_id=client_id, prev_data_indx=prev_data_indx), _TIMEOUT_SECONDS)
			print('Processing parameters in batch %d!' % response.data_indx)

			# Generates the W matrix 
			get_parameters_time = time.time()
			W_bytes = ''
			W_subtensors_iter = stub.SendParams(dist_sgd_pb2.ClientInfo(client_id=client_id), TENSOR_TIMEOUT_SECONDS)
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

			consec_expiration = 0
	except KeyboardInterrupt as e:
		sys.exit(1)
	except Exception as e:
		# TODO, This should be logged rather than simply ed
		# Failure of server should be caught and determined
		if ('ExpirationError' in str(e) or 'NetworkError' in str(e)):
			consec_expiration += 1
			if consec_expiration == 2:
				print('Failure to connect to server_stub. Starting Paxos')
				while server_addr == '':
					server_addr = run_paxos(local_id)
					if server_addr == '':
						server_addr = find_server(local_id)
				if server_addr == gen_local_address(local_id):
					serve(server_addr, W, prev_data_indx, local_id)
					return
				stub = connect_server_stub(server_addr)
		else:
			print(traceback.print_exc())
			sys.exit(0)

if __name__ == '__main__':
	print('Starting client')
	parser = argparse.ArgumentParser()
	parser.add_argument('--id')
	args = parser.parse_args()
	local_id = args.id
	if local_id is not None:
		local_id = int(local_id)
		assert(local_id > 0)
	while True:
		run(local_id)