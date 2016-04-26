from __future__ import absolute_import
from __future__ import print_function
from scipy.ndimage import imread
from scipy.misc import imresize
from sklearn.cross_validation import train_test_split
from os import listdir
from os.path import isfile, join
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from autograd.util import quick_grad_check

# {0: 'accordion', 1: 'airplanes', 2: 'anchor', 3: 'ant', 4: 'BACKGROUND_Google', 5: 'barrel', 6: 'bass', 7: 'beaver', 8: 'binocular', 9: 'bonsai', 10: 'brain', 11: 'brontosaurus', 12: 'buddha', 13: 'butterfly', 14: 'camera', 15: 'cannon', 16: 'car_side', 17: 'ceiling_fan', 18: 'cellphone', 19: 'chair', 20: 'chandelier', 21: 'cougar_body', 22: 'cougar_face', 23: 'crab', 24: 'crayfish', 25: 'crocodile', 26: 'crocodile_head', 27: 'cup', 28: 'dalmatian', 29: 'dollar_bill', 30: 'dolphin', 31: 'dragonfly', 32: 'electric_guitar', 33: 'elephant', 34: 'emu', 35: 'euphonium', 36: 'ewer', 37: 'Faces', 38: 'Faces_easy', 39: 'ferry', 40: 'flamingo', 41: 'flamingo_head', 42: 'garfield', 43: 'gerenuk', 44: 'gramophone', 45: 'grand_piano', 46: 'hawksbill', 47: 'headphone', 48: 'hedgehog', 49: 'helicopter', 50: 'ibis', 51: 'inline_skate', 52: 'joshua_tree', 53: 'kangaroo', 54: 'ketch', 55: 'lamp', 56: 'laptop', 57: 'Leopards', 58: 'llama', 59: 'lobster', 60: 'lotus', 61: 'mandolin', 62: 'mayfly', 63: 'menorah', 64: 'metronome', 65: 'minaret', 66: 'Motorbikes', 67: 'nautilus', 68: 'octopus', 69: 'okapi', 70: 'pagoda', 71: 'panda', 72: 'pigeon', 73: 'pizza', 74: 'platypus', 75: 'pyramid', 76: 'revolver', 77: 'rhino', 78: 'rooster', 79: 'saxophone', 80: 'schooner', 81: 'scissors', 82: 'scorpion', 83: 'sea_horse', 84: 'snoopy', 85: 'soccer_ball', 86: 'stapler', 87: 'starfish', 88: 'stegosaurus', 89: 'stop_sign', 90: 'strawberry', 91: 'sunflower', 92: 'tick', 93: 'trilobite', 94: 'umbrella', 95: 'watch', 96: 'water_lilly', 97: 'wheelchair', 98: 'wild_cat', 99: 'windsor_chair', 100: 'wrench', 101: 'yin_yang'}

def make_nn_funs(layer_sizes, L2_reg):
    shapes = zip(layer_sizes[:-1], layer_sizes[1:])
    N = sum((m+1)*n for m, n in shapes)

    def unpack_layers(W_vect):
        for m, n in shapes:
            yield W_vect[:m*n].reshape((m,n)), W_vect[m*n:m*n+n]
            W_vect = W_vect[(m+1)*n:]

    def predictions(W_vect, inputs):
        for W, b in unpack_layers(W_vect):
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        return outputs - logsumexp(outputs, axis=1, keepdims=True)

    def loss(W_vect, X, T):
        log_prior = -L2_reg * np.dot(W_vect, W_vect)
        log_lik = np.sum(predictions(W_vect, X) * T)
        return - log_prior - log_lik

    def frac_err(W_vect, X, T):
        return np.mean(np.argmax(T, axis=1) != np.argmax(predictions(W_vect, X), axis=1))

    return N, predictions, loss, frac_err

def convert_bw_to_rgb(im):
    im.resize((im.shape[0], im.shape[1], 1))
    return np.repeat(im.astype(np.uint8), 3, 2)

def standarizeImage(im):
	if len(im.shape) < 3:
		im = convert_bw_to_rgb(im)
	im = np.array(im, 'float32') 
	if im.shape[0] != 64:
		im = imresize(im, (64, 64, 3))
	if np.amax(im) > 1.1:
		im = im / 255.0
	assert((np.amax(im) > 0.01) & (np.amax(im) <= 1))
	assert((np.amin(im) >= 0.00))
	return im

def gen_data():
	category_paths = [f for f in listdir('101_ObjectCategories/')]
	image_paths = [f for f in listdir('101_ObjectCategories/menorah/') if isfile(join('101_ObjectCategories/menorah/', f))]

	images = []
	output_labels = []
	# Include all categories with mappings to the integer representing the category
	categories_dict = {}

	category = 0
	for category_path in category_paths:
		image_paths = [f for f in listdir('101_ObjectCategories/' + category_path + '/')]
		for image_path in image_paths:
			im = standarizeImage(imread('101_ObjectCategories/' + category_path + '/' + image_path))
			if im.shape == (64, 64, 3):
				images.append(im)
				output_labels.append(category)
		categories_dict[category] = category_path
		category = category + 1

	images = np.array(images)
	partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
	images = partial_flatten(images)

	np.save('images.npy', images)
	np.save('output_labels.npy', output_labels)

def make_batches(N_data, batch_size):
    return [slice(i, min(i+batch_size, N_data))
            for i in range(0, N_data, batch_size)]

def load_caltech100(): 
	# gen_data()
	one_hot = lambda x, K: np.array(x[:,None] == np.arange(K)[None, :], dtype=int)
	images = np.load('images.npy')
	output_labels = np.load('output_labels.npy')
	train_images, valid_images, train_labels, valid_labels = train_test_split(images, output_labels, test_size=0.20, random_state=1729)
	train_labels = one_hot(train_labels, 101)
	valid_labels = one_hot(valid_labels, 101)
    # import bpdb; bpdb.set_trace()
	return train_images, train_labels, valid_images, valid_labels

if __name__ == '__main__':
    
	# Load and process Caltech data
	train_images, train_labels, test_images, test_labels = load_caltech100()
	image_input_d = train_images.shape[1]

    # Network parameters
	layer_sizes = [image_input_d, 1500, 650, 101]
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

	# Initialize weights
	rs = npr.RandomState()
	W = rs.randn(N_weights) * param_scale

	# Check the gradients numerically, just to be safe
	# quick_grad_check(loss_fun, W, (train_images, train_labels))

	print("    Epoch      |    Train err  |   Test err  ")

	def print_perf(epoch, W):
	    test_perf  = frac_err(W, test_images, test_labels)
	    train_perf = frac_err(W, train_images, train_labels)
	    print("{0:15}|{1:15}|{2:15}".format(epoch, train_perf, test_perf))

	# Train with sgd
	batch_idxs = make_batches(train_images.shape[0], batch_size)
	cur_dir = np.zeros(N_weights)

	for epoch in range(num_epochs):
	    print_perf(epoch, W)
	    for idxs in batch_idxs:
	        grad_W = loss_grad(W, train_images[idxs], train_labels[idxs])
	        print(grad_W.shape)
	        cur_dir = momentum * cur_dir + (1.0 - momentum) * grad_W
	        W -= learning_rate * cur_dir