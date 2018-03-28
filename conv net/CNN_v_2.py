import random
from PIL import Image
from better_neural_network import Neural_Network
import math

learning_rate = .01


def threed_to_oned(input_matrix):
	return [input_matrix[k][r][c] for k in range(len(input_matrix)) for r in range(len(input_matrix[0])) for c in range(len(input_matrix[0][0]))]

def oned_to_threed(input_matrix, x, y, z):
	return [[[input_matrix[k][r][c] for k in range(x)] for r in range(y)] for c in range(z)]







class filter():
	"""a 3d filter that can change its weights over time
	"""
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z
		self.nn = Neural_Network([x*y*z, 1])
		# self.weights = [[[random.random() for xx in range(x)] for i in range(y)] for j in range(z)] OLD
		
	def accept_input(self, input_matrix):
		"""
		input is going to be a 3d 'slice' out of a larger whole, and we want to just kind of sum up everything multiplied by our weights
		"""
		# print "self.weights:", len(self.weights), len(self.weights[0]), len(self.weights[0][0])
		# print "input_matrix:", len(input_matrix), len(input_matrix[0]), len(input_matrix[0][0])
		# return sum(   [self.weights[z][x][y] * input_matrix[z][x][y] for x in range(len(input_matrix[0])) for y in range(len(input_matrix[0][0])) for z in range(len(input_matrix))]) OLD

		fully_connected_input = threed_to_oned(input_matrix)
		nn_output = self.nn.get_output(fully_connected_input)
		return nn_output


	def train(self, input_matrix, change_above):
		"""
		Ok so it gets the change of the thing above it
		Its error is the change above it times the weights
		Its change is its error times its values times 1 - its value
		"""

		return oned_to_threed(self.nn.backpropigate(change_above)[0], self.x, self.y, self.z)

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
					output[f][x/self.step_size][y/self.step_size] = self.filters[f].accept_input(self.filter_one(input_matrix, {"x": x, "y": y}, self.filter_radius))
		self.output = output
		return output

	def generate_error(self, input_matrix, correct_output):
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

		self.filter_all(input_matrix)
		if len(self.output[0][0]) != 1:
			raise ValueError("The network is not a final output layer, dont call this method on it, or change the parameters of this layer accordingly")

		initial_error_matrix = [0 for i in range(len(correct_output))]
		final_error_matrix = [[[0] for i in range(len(input_matrix[0][0])) for j in range(len(input_matrix[0]))] for k in range(len(input_matrix))]

		for i in range(len(correct_output)):
			initial_error_matrix[i] = correct_output[i] - self.output[i][0][0]

		# print(initial_error_matrix)

		# ok now loop back 

		start_x = 0 if self.zpad else self.filter_radius
		start_y = 0 if self.zpad else self.filter_radius

		for f in range(len(self.filters)):
			for x in range(start_x, len(input_matrix[0]), self.step_size):
				for y in range(start_y, len(input_matrix[0][0]), self.step_size):
					change = self.filters[f].train(self.filter_one(input_matrix, {"x": x, "y": y}, self.filter_radius), initial_error_matrix[f])
					pass






class image_input(object):
	"""docstring for image_input"""
	def __init__(self, image_name):
		image = Image.open(image_name)
		pix = image.load()

		self.output = [[[pix[r,c][z] / 768.0 * 2 + 1 for c in range(image.size[1])] for r in range(image.size[0])] for z in range(3)]

def pretty_print(matrix):
	for i in range(len(matrix)):
		print "############### depth", i, "################", len(matrix[i]), "by", len(matrix[i][0])
		for r in range(len(matrix[i][0])):
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
		self.locations = [[[0 for x in range(input_x / self.step_size)] for y in range(input_y / self.step_size)] for z in range(input_z)]

		for z in range(len(input_matrix)):
			for x in range(self.step_size, len(input_matrix[0]), self.step_size):
				for y in range(self.step_size, len(input_matrix[0][0]), self.step_size):
					values_to_find_max = []
					for dx in range(self.step_size*-1, self.step_size):
						for dy in range(self.step_size*-1, self.step_size):
							values_to_find_max += [input_matrix[z][x+dx][y+dy]]
					self.output[z][x/self.step_size][y/self.step_size] = max(values_to_find_max)
					self.locations[z][x/self.step_size][y/self.step_size] = values_to_find_max.index(max(values_to_find_max))
		return self.output

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
		


layer_0 = image_input("4.png")
layer_1 = convolutional_layer(num_filters=2, step_size=2, filter_radius=1, input_z=3)
layer_2 = max_pool(2)
layer_3 = convolutional_layer(num_filters=4, step_size=8, filter_radius=4, input_z=2, zpad=False)

layer_0_output = layer_0.output
layer_1_output = layer_1.filter_all(layer_0_output)
layer_2_output = layer_2.pool(layer_1_output)
layer_3_output = layer_3.filter_all(layer_2_output)

e = layer_3.generate_error(layer_2_output, [1,1,0,0])
# pretty_print(layer_0_output)
# pretty_print(layer_1_output)
# pretty_print(layer_2_output)
# pretty_print(layer_3_output)

		