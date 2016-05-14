from __future__ import print_function
from __future__ import absolute_import
from grpc.beta import implementations
import time
import sys
from threading import Thread

import paxos_pb2
import argparse
import traceback

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
import random

from protobuf_utils.utils import * 
import subprocess

_TIMEOUT_SECONDS = 1
PAXOS_PORT_STR = 50052

# Actual impelementation of the PaxosServer that is used to communicate between the clients. 
# These are instantiated simply to determine the future main server from many different clients.
class PaxosServer(paxos_pb2.BetaPaxosServerServicer):
    def __init__(self, hostname):
    	self.consensus_value = None
    	self.n = 0
    	self.v = ''
    	self.n_v = 0
    	self.backoff = 1
    	self.consensus_reached = False
    	self.address = hostname
    	self.new_server = None

    def prepare(self, request, context):
    	return paxos_pb2.ack(n=self.n, v=self.v, n_v=self.n_v)

    def accept(self, request, context):
    	if request.n > self.n:
    		self.n_v = request.n
    		self.v = request.v
        	return paxos_pb2.acquiescence(accept_bool=True)
        else:
        	return paxos_pb2.acquiescence(accept_bool=False)

    def accepted(self, request, context):
    	self.consensus_reached = True
    	self.new_server = request.v
    	return paxos_pb2.empty()

    def ping(self, request, context):
    	return paxos_pb2.empty() 

def run_server(server, paxos_server):
	server.start()
	while True:
		time.sleep(0.1)
		try:
			if paxos_server.consensus_reached:
				if paxos_server.new_server != '':
					print('Consensus reached, server shutting down')
				time.sleep(5)
				server.stop(0)
				break
			time.sleep(1)
		except KeyboardInterrupt:
			server.stop(0)

def create_server(hostname):
    # Allow argument that allows this parameter to be changsed
	paxos_server = PaxosServer(hostname)
	server = paxos_pb2.beta_create_PaxosServer_server(paxos_server)
	server.add_insecure_port(hostname)
	return paxos_server, server

def send_proposals(server_stubs, self_paxos_server):
	self_paxos_server.n = self_paxos_server.n + 1
	self_paxos_server.v = self_paxos_server.address

	n_proposal = self_paxos_server.n
	value = self_paxos_server.address
	print('Making a proposal from {0} for n = {1} '.format(self_paxos_server.address, n_proposal))

	n_so_far = 0
	failed = False
	responded = 0

	for server_stub in server_stubs:
		# Makes the connection to the server
		try:
			response = server_stub.prepare(paxos_pb2.proposal(n=n_proposal), _TIMEOUT_SECONDS)

			# Sees a higher n value then it's current value and immediately stops the process
			if response.n >= n_proposal:
				failed = True
				print('Proposal ' + str(n_proposal) + ' failed')
				break
			else:
			 	if response.n_v > n_so_far:
					n_so_far = response.n
					value = response.v
				responded += 1
		except Exception as e:
			if ('ExpirationError' in str(e)):
				print('Failure to connect to server_stub')
				continue
			else:
				# More severe error, should log and crash
				traceback.print_exc()
				sys.exit(1)

	# No proposals have been sent so far
	if value is None:
		value = self_paxos_server.address

	if responded < len(server_stubs) / 2.0:
		failed = True
		
	return(failed, n_proposal, value)

def request_accept(server_stubs, self_paxos_server, n_proposal, value):
	accepted = 0
	for stub in server_stubs:
		try:
			response = stub.accept(paxos_pb2.request_acceptance(n=n_proposal, v=value), _TIMEOUT_SECONDS)
		except Exception as e:
			traceback.print_exc()
			return False

		if response.accept_bool:
			accepted += 1
	if accepted > len(server_stubs) / 2.0:
		print('Proposal accepted')
		return True
	else:
		print('Proposal {0} rejected with value {1}'.format(n_proposal, value))
		return False


def check_stubs_up(stubs):
	# Make sure that all machines are aware that the Paxos algorithm is finishing
	# Not all machines are aware that the server have failed at the same time 
	responses = 0
	for stub in stubs:
		try:
			response = stub.ping(paxos_pb2.empty(), _TIMEOUT_SECONDS)
			responses += 1
		except Exception as e:
			if ('ExpirationError' in str(e)):
				print('Failure to connect to server_stub during startup')
				continue
			else:
				# More severe error, should log and crash
				traceback.print_exc()
				sys.exit(1)
	if responses < len(stubs):
		return False
	else:
		return True

def gen_server_stubs(self_paxos_server, local_id):
	# Make sure that all machines are aware that the Paxos algorithm is finishing
	# Not all machines are aware that the server have failed at the same time 
	TOT_ATTEMPTS = 3
	for i in range(TOT_ATTEMPTS):
		server_addresses = gen_server_addresses(local_id)
		server_addresses.remove(self_paxos_server.address)
		stubs = []
		for server_address in server_addresses:
			if not self_paxos_server.consensus_reached:
				if local_id is not None:
					server_port = int(server_address[-5:])
					channel = implementations.insecure_channel('localhost', server_port)
				else:
					channel = implementations.insecure_channel(server_address, PAXOS_PORT_STR)

				stub = paxos_pb2.beta_create_PaxosServer_stub(channel)
				stubs.append(stub)
		all_stubs_responsive = check_stubs_up(stubs)
		if all_stubs_responsive:
			return stubs
		time.sleep(1 * TOT_ATTEMPTS)
	return None

def broadcast_consensus(server_stubs, self_paxos_server, value):
	for stub in server_stubs:
		# Makes the connection to the server
		response = stub.accepted(paxos_pb2.consensus(n=self_paxos_server.n, v=value), _TIMEOUT_SECONDS)

def start_paxos(server_stubs, self_paxos_server):
	proposal_failed, n_proposal, value = send_proposals(server_stubs, self_paxos_server)
	if not proposal_failed and not self_paxos_server.consensus_reached:
		# Have everyone accept the proposal
		accepted = request_accept(server_stubs, self_paxos_server, n_proposal, value)
		if accepted and not self_paxos_server.consensus_reached:
			# If accepted, let everyone know that the server has been chosen
			broadcast_consensus(server_stubs, self_paxos_server, value)
			self_paxos_server.new_server = value
			self_paxos_server.consensus_reached = True
			return True

	# If proposal failed, backoff to try again later
	self_paxos_server.backoff = self_paxos_server.backoff * 2
	return False

def paxos_loop(self_paxos_server, local_id):
	# This send_proposal_time should be based on the actual batch size
	time_slept = 0
	send_proposal_time = int(self_paxos_server.backoff)
	while not self_paxos_server.consensus_reached:
		time.sleep(1)
		time_slept += 1
		# Send a proposal at allocated time
		if time_slept == send_proposal_time and not self_paxos_server.consensus_reached:
			time.sleep(random.random())
			server_stubs = gen_server_stubs(self_paxos_server, local_id)
			if server_stubs is None:
				self_paxos_server.new_server = ''
				break
			start_paxos(server_stubs, self_paxos_server)
			send_proposal_time = int(5 * random.random() * self_paxos_server.backoff)
			time_slept = 0

def gen_local_address(local_id):
	if local_id is None:
		addr = subprocess.check_output("ip addr show eth0 | grep 'inet' | cut -d ' ' -f8", shell=True)
		return addr[:-1]
	else:
		server_addresses = gen_server_addresses(local_id)
		return server_addresses[local_id - 1]

def gen_server_addresses(local_id):
	if local_id is None:
		internal_ip = []
		# Ugly formatting when directly using pipes, using files instead
		with open('ips.txt', 'w') as f:
		    subprocess.call(["gcloud", "compute", "instances", "list"], stdout=f)
		with open('ips.txt', 'r') as f:
		    lines = f.readlines()
		    for line in lines[1:]:
		        line_arr = filter((lambda x: x != '') , line.split(' '))
		        internal_ip.append(line_arr[3])
		return internal_ip
	if local_id is not None:
		return ['[::]:50052', '[::]:50053', '[::]:50044']

# This is the final function that exterior functions will call
def run_paxos(local_id=None):
	hostname = gen_local_address(local_id)
	print(hostname + ' called to run Paxos for determining the server')

	paxos_server, server = create_server(hostname)
	try:
		Thread(target=run_server, args=(server,paxos_server,)).start()
		start_paxos = time.time()
		paxos_loop(paxos_server, local_id)
		if paxos_server.new_server != '':
			print('Done, new server is: {0} finished paxos in {1:2}s'.format(paxos_server.new_server, time.time()-start_paxos))
		else:
			print('Failure to connect to other allocated instances. Stopping paxos.')
	except KeyboardInterrupt:
		sys.exit(0)
	finally:
		paxos_server.consensus_reached = True
		server.stop(0)
	return paxos_server.new_server

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--id')
	args = parser.parse_args()
	local_id = args.id
	if local_id is not None:
		local_id = int(local_id)
		assert(local_id > 0)
	print(run_paxos(local_id))
