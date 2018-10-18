import random
from PIL import Image
from better_neural_network import Neural_Network
import math

import time

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# learning_rate = 10000


def threed_to_oned(input_matrix):
	return [input_matrix[k][r][c] for k in range(len(input_matrix)) for r in range(len(input_matrix[0])) for c in range(len(input_matrix[0][0]))]

def oned_to_threed(input_matrix, x, y, z):
	# for c in range(z):
	# 	for r in range(y):
	# 		for k in range(x):
				# print k, r, c
				# print x, y, z
				# print k + r*y + c
				# print input_matrix[k*x + r*y + c]
	return [[[input_matrix[k + r*y + c*x*y] for k in range(x)] for r in range(y)] for c in range(z)]





can_print = False

class filter():
	"""a 3d filter that can change its weights over time
	"""
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z
		self.nn = Neural_Network([x*y*z, 1], relu=False)
		# self.weights = [[[random.random() for xx in range(x)] for i in range(y)] for j in range(z)] OLD
		
	def accept_input(self, input_matrix):
		"""
		input is going to be a 3d 'slice' out of a larger whole, and we want to just kind of sum up everything multiplied by our weights
		"""
		# print "self.weights:", len(self.weights), len(self.weights[0]), len(self.weights[0][0])
		# print "input_matrix:", len(input_matrix), len(input_matrix[0]), len(input_matrix[0][0])
		# return sum(   [self.weights[z][x][y] * input_matrix[z][x][y] for x in range(len(input_matrix[0])) for y in range(len(input_matrix[0][0])) for z in range(len(input_matrix))]) OLD

		fully_connected_input = threed_to_oned(input_matrix)
		# print "---------------"
		# print fully_connected_input
		# print "---------------"
		nn_output = self.nn.get_output(fully_connected_input)
		return nn_output[0]


	def train(self, input_matrix, change_above):
		"""
		Ok so it gets the change of the thing above it
		Its error is the change above it times the weights
		Its change is its error times its values times 1 - its value
		"""
		self.accept_input(input_matrix)
		return oned_to_threed(self.nn.backpropigate(change_above, print_me=False)[0], self.x, self.y, self.z)

class convolutional_layer():
	"""ok so this makes a few filters and applies each of them to the input layer"""
	def __init__(self, num_filters, step_size, filter_radius, input_z, zpad=True):
		self.num_filters = num_filters
		self.step_size = step_size
		self.filter_radius = filter_radius
		self.zpad = zpad
		self.filters = [filter(filter_radius * 2 + 1, filter_radius * 2 + 1, input_z) for i in range(num_filters)]

	def filter_one(self, input_matrix, center, radius):
		new_input = [[[0]*(radius*2 + 1) for i in range(radius*2 + 1)] for j in range(len(input_matrix))]
		for z in range(len(input_matrix)):
			# this iterates over every layer of depth (Z)
			for x in range(radius*2 + 1):
				for y in range(radius*2 + 1):
					if center["x"]+x-radius >= len(input_matrix[z]) or center["x"]+x-radius < 0 or center["y"]+y-radius >= len(input_matrix[z][x]) or center["y"]+y-radius < 0:
						new_input[z][x][y] = 0
					else:
						new_input[z][x][y] = input_matrix[z][center["x"]+x-radius][center["y"]+y-radius]
		return new_input
		# return filter.accept_input(new_input)


	def filter_all(self, input_matrix):
		output = [[[0 for x in range(len(input_matrix[0])/self.step_size)] for y in range(len(input_matrix[0][0])/self.step_size)] for f in range(len(self.filters))]
		start_x = 0 if self.zpad else self.filter_radius
		start_y = 0 if self.zpad else self.filter_radius

		for f in range(len(self.filters)):
			for x in range(start_x, len(input_matrix[0]), self.step_size):
				for y in range(start_y, len(input_matrix[0][0]), self.step_size):
					# print f, x, y
					# print len(output), len(output[0]), len(output[0][0]), len(output[0][0])
					output[f][y/self.step_size][x/self.step_size] = self.filters[f].accept_input(self.filter_one(input_matrix, {"x": x, "y": y}, self.filter_radius))
		self.output = output
		return output

	def generate_error(self, input_matrix, error_matrix):
		"""
		Ok so what do we know?
			well we know that the correct output is going to be like [[[1]], [[0]], [[1]]... ], and so SHOULD the output  of this layer
		So whats first?
			generate the output
			make it throw an error if its output is not the correct size
		Alright done, now for the hard(er) part
		Iterate over every output and the error is calcualted as correct - expected

		make the error matrix, it should be the same size as the input matrix
		"""

		output = [[[0 for x in range(len(input_matrix[0][0]))] for y in range(len(input_matrix[0]))] for z in range(len(input_matrix))]
		
		start_x = 0 if self.zpad else self.filter_radius
		start_y = 0 if self.zpad else self.filter_radius

		for f in range(len(self.filters)):
			for x in range(start_x, len(input_matrix[0]), self.step_size):
				for y in range(start_y, len(input_matrix[0][0]), self.step_size):
					filter_out = self.filters[f].train(self.filter_one(input_matrix, {"x": x, "y": y}, self.filter_radius), [error_matrix[f][x/self.step_size][y/self.step_size]])
					for zz in range(len(filter_out)):
						for yy in range(len(filter_out[0])):
							for xx in range(len(filter_out[0][0])):
								# print zz, xx + x, yy + y
								# print len(output), len(output[0]), len(output[0][0])
								if xx + x - self.filter_radius < len(output[0]) and yy + y - self.filter_radius < len(output[0][0]):
									output[zz][xx + x - self.filter_radius][yy + y - self.filter_radius] += filter_out[zz][yy][xx]


		self.output = output
		return output

class image_input(object):
	"""docstring for image_input"""
	def __init__(self, image_name):
		image = Image.open(image_name)
		image = image.resize((40, 40), Image.ANTIALIAS)
		pix = image.load()

		self.output = [[[pix[r,c][z] / 256.0 for c in range(image.size[1])] for r in range(image.size[0])] for z in range(3)]

def pretty_print(matrix):
	for i in range(len(matrix)):
		print "############### depth", i, "################", len(matrix[i]), "by", len(matrix[i][0])
		for r in range(len(matrix[i])):
			for c in range(len(matrix[i][r])):
				print ("%.2f" % matrix[i][r][c]).zfill(5),
			print ""

class max_pool(object):
	"""docstring for max_pool"""
	def __init__(self, step_size):
		self.step_size = step_size
		self.output = None
		self.locations = None


	def pool(self, input_matrix):
		input_x = len(input_matrix[0])
		input_y = len(input_matrix[0][0])
		input_z = len(input_matrix)

		self.output = [[[0 for x in range(input_x / self.step_size)] for y in range(input_y / self.step_size)] for z in range(input_z)]
		# pretty_print(input_matrix)
		for z in range(len(input_matrix)):
			for x in range(0, len(input_matrix[0]), self.step_size):
				for y in range(0, len(input_matrix[0][0]), self.step_size):
					values_to_find_max = []
					for dx in range(self.step_size*-1, self.step_size):
						for dy in range(self.step_size*-1, self.step_size):
							if y+dy < len(input_matrix[0]) and x+dx < len(input_matrix[0][0]):
								values_to_find_max += [input_matrix[z][y+dy][x+dx]]
					# print values_to_find_max
					self.output[z][y/self.step_size][x/self.step_size] = max(values_to_find_max)
		return self.output

	def reverse_pool(self, input_matrix, error_matrix):
		input_x = len(input_matrix[0])
		input_y = len(input_matrix[0][0])
		input_z = len(input_matrix)
		new_error = output = [[[0 for x in range(input_x)] for y in range(input_y)] for z in range(input_z)]

		for z in range(len(input_matrix)):
			for x in range(self.step_size, len(input_matrix[0]), self.step_size):
				for y in range(self.step_size, len(input_matrix[0][0]), self.step_size):
					values_to_find_max = []
					for dx in range(0, self.step_size):
						for dy in range(0, self.step_size):
							values_to_find_max += [input_matrix[z][x+dx][y+dy]]

					# instead of the line below, we want to:
					# find the x and y of the max element
					# set the output at that x and y to that max element
					# self.output[z][x/self.step_size][y/self.step_size] = max(values_to_find_max)

					max_value_location = values_to_find_max.index(max(values_to_find_max))
					max_value_y = y + (max_value_location % self.step_size)
					max_value_x = x + (max_value_location / self.step_size)
					# print len(values_to_find_max), self.step_size
					# print max_value_x, max_value_y, len(new_error[0]), len(new_error[0][0]), x, max_value_location 
					new_error[z][max_value_x][max_value_y] = error_matrix[z][x/self.step_size][y/self.step_size]
		return new_error

class fully_connected_layer():
	"""docstring for fully_connected_layer"""
	def __init__(self, sizes):
		self.nn = Neural_Network(sizes)
		self.output = None

	def get_output(self, input_matrix):
		if len(self.nn.layers[0]) != len(input_matrix)*len(input_matrix[0])*len(input_matrix[0][0]):
			raise ValueError('The size of the neural networks one dimentional output doesnt match the size of this layers 3 dimentional output')
		fully_connected_input = [input_matrix[k][r][c] for k in range(len(input_matrix)) for r in range(len(input_matrix[0])) for c in range(len(input_matrix[0][0]))]
		nn_output = self.nn.get_output(fully_connected_input)
		self.output = nn_output
		return nn_output

	def get_error(self, input_matrix, correct_output):
		# new_input = [[[1/(1+ (math.e ** (-1*input_matrix[a][b][c]))) for c in range(len(input_matrix[0][0]))] for b in range(len(input_matrix[0]))] for a in range(len(input_matrix))]
		output = self.get_output(input_matrix)
		final_error = [correct_output[i] - output[i] for i in range(len(correct_output))]

		# print final_error
		return oned_to_threed(self.nn.backpropigate(final_error, True)[0], len(input_matrix[0][0]), len(input_matrix[0]), len(input_matrix))

		


# # inputs: 3 by 20 by 20
# layer_c_1 = image_input("c_1.png")
# layer_c_2 = image_input("c_2.png")
# layer_c_3 = image_input("c_3.png")
# layer_c_4 = image_input("c_4.png")

# layer_a_1 = image_input("a_1.png")
# layer_a_2 = image_input("a_2.png")
# layer_a_3 = image_input("a_3.png")
# layer_a_4 = image_input("a_4.png")


# input is 1 by 28 by 28
layer_1 = convolutional_layer(num_filters=4, step_size=1, filter_radius=2, input_z=1, zpad=True) # 32 by (28/1)=28 by (28/1) = 28
layer_2 = max_pool(2)                                                                            # 32 by (28/2) = 14 by (28/2) = 14
layer_3 = convolutional_layer(num_filters=8, step_size=1, filter_radius=2, input_z=4, zpad=True) # 64 by (14/1) = 14 by (14/1) = 14
layer_4 = max_pool(2)                                                                            # 64 by (14/2) = 7 by (14/2) = 7
layer_5 = fully_connected_layer([392, 10])


# # layer_0_output = layer_0.output
# # layer_1_output = layer_1.filter_all(layer_0_output)
# # layer_2_output = layer_2.pool(layer_1_output)
# # layer_3_output = layer_3.filter_all(layer_2_output)



def train(image_input, answer):
	global can_print
	answer_array = [0 for i in range(10)]
	answer_array[answer] = 1

	layer_0_output = image_input
	layer_1_output = layer_1.filter_all(layer_0_output)
	# print "layer_1_output"
	# pretty_print(layer_1_output)
	layer_2_output = layer_2.pool(layer_1_output)
	# print "layer_2_output"
	# pretty_print(layer_2_output)
	layer_3_output = layer_3.filter_all(layer_2_output)
	# print "layer_3_output"
	# pretty_print(layer_3_output)
	layer_4_output = layer_4.pool(layer_3_output)
	result = layer_5.get_output(layer_4_output)
	# print "layer_5_output"
	# print(result)

	e = layer_5.get_error(layer_4_output, answer_array)
	# pretty_print(e)
	# print layer_3.filters[0].nn.weights[0]
	# print 1/0

	layer_4_error = layer_4.reverse_pool(layer_3_output, e)
	
	layer_3_error = layer_3.generate_error(layer_2_output, layer_4_error)
	# pretty_print(layer_3_error)
	layer_2_error = layer_2.reverse_pool(layer_1_output, layer_3_error)
	layer_1_error = layer_1.generate_error(layer_0_output, layer_2_error)
	# can_print = False
	# print e
	# pretty_print(e)

	return result

def run(image_input):
	layer_0_output = image_input
	layer_1_output = layer_1.filter_all(layer_0_output)
	layer_2_output = layer_2.pool(layer_1_output)
	layer_3_output = layer_3.filter_all(layer_2_output)
	layer_4_output = layer_4.pool(layer_3_output)
	result = layer_5.get_output(layer_4_output)

	return result

# for i in range(10000):
# 	train(layer_c_1, 1)
# 	train(layer_a_1, 0)
# 	train(layer_c_2, 1)
# 	train(layer_a_2, 0)

# 	print "testing now..."

# 	run(layer_c_3)
# 	run(layer_c_4)
# 	run(layer_a_3)
# 	run(layer_a_4)


# 	print "---"



##################################

from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split

mnist = fetch_mldata('MNIST original')

train_img, test_img, train_lbl, test_lbl = train_test_split(
    mnist.data, mnist.target, test_size=1/7.0, random_state=2)


# plt.figure(figsize=(20,4))
# for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
# 	plt.subplot(1, 5, index + 1)
# 	plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
# 	plt.title('Training: %i\n' % label, fontsize = 20)

# plt.show()

i = 0
for index, (image, label) in enumerate(zip(train_img, train_lbl)):
	# print len(np.reshape(image, (28, 28)))
	results = train( [np.multiply(np.reshape(image, (28, 28)), 1/256.0)], int(label))
	# print [np.multiply(np.reshape(image, (28, 28)), 1/256.0)]
	best = results.index(max(results))
	if len(set(results)) >= 2:
		second_best_num = sorted(set(results))[-2]
		print bcolors.OKGREEN if best == int(label) and max(results) > second_best_num + .2 else bcolors.WARNING if best == int(label) else bcolors.FAIL, label, ": %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % (results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7], results[8], results[9])
		i += 1
		if i % 5 == 0:
			print "       0    1    2    3    4    5    6    7    8    9"

time.sleep(500)