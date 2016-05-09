import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
import dist_sgd_pb2

def convert_array_to_bytes(params):
    if (params.dtype == np.float64):
        params = params.astype(np.float32)
    param_bytes = params.tostring()
    return param_bytes

def convert_bytes_to_array(param_bytes):
    params = np.fromstring(param_bytes, dtype=np.float32)
    return params

def convert_tensor_iter(tensor_bytes, data_indx):
	CHUNK_SIZE = 524228
	tensor_bytes_len = len(tensor_bytes)
	tensor_chunk_count = 0
	while len(tensor_bytes):
	    tensor_chunk_count += 1
	    tensor_content = tensor_bytes[:CHUNK_SIZE]
	    tensor_bytes = tensor_bytes[CHUNK_SIZE:]
	    yield dist_sgd_pb2.SubTensor(tensor_len = tensor_bytes_len, tensor_chunk = tensor_chunk_count, tensor_content = tensor_content, data_indx = data_indx)