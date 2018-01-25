# for random intializatin of weights
# also for randomly zeroing weights during runtime
import random, math






class Neural_Network(object):
	""" Neural_Network
		@author Devin Uner
		this is a module for creating neural networks of arbitrary size
		it has several methods available, but mostly you should use the train, and get_output methods
		leave everything else alone, thats all private (mostly math) stuff

		EXAMPLE
		# create a network with 2 input neurons, 3 hidden neurons, 2 more hidden neurons, and one output neuron
		my_NN = Neural_Network([2, 3, 2, 1])
		# train it
		for i in range(1000):
			my_NN.train([0,0], [0])
			my_NN.train([0,1], [1])
			my_NN.train([1,0], [1])
		# get the output of something its never seen
		#print my_NN.get_output([1,1])
	"""
	def __init__(self, num_neurons_per_layer_):
		# the number of layers in the neural network
		self.layer_count = len(num_neurons_per_layer_)

		# the layers themselves
		self.layers = [[0]*layer_len for layer_len in num_neurons_per_layer_]

		# initialize the weights of the neural network, try to read in the weights, and if they are there great, if not, make new ones
		self.weights = []
		try:
			self.read_data()
		except:
			#print "failed loading past weights, creating new ones..."
			for i in range(0, len(num_neurons_per_layer_)-1):
				self.weights += [[     [random.uniform(-1,1) for x in range(num_neurons_per_layer_[i+1])]     for j in range(0,num_neurons_per_layer_[i])]]


	# sigmoid function, used as the activation function of the network
	# can also compute the dirivative of the sigmoid function as requested
	# default is to compute the value and not the dertivative
	# PRIVATE dont mess with this, you should never need to call it
	def sigmoid(self, x, deriv=False):
		if(deriv):
			return x*(1-x)
		return 1/(1+ (math.e ** (-1*x)))


	# computes the dot product of x and y, calling the provided function on every number in the resulting matrix
	# PRIVATE dont mess with this, you should never need to call it
	def dot_with_func(self, x, y, func):
		new = []
		for c in range(0,len(y[0])):
			new += [func(sum([x[r]*y[r][c] for r in range(0, len(y))]))]
		return new

	# gets the output of the network given the input
	# does NOT backpropigate error OR learn from ANY mistakes
	# PUBLIC feel free to call this one
	def get_output(self, data):
		if(len(data) != len(self.layers[0])):
			raise ValueError("invalid length of input :)")
		# set the initial input
		self.layers[0] = data
		for i in range(1, self.layer_count-1):
			# calculate the next line of output
			self.layers[i] = self.dot_with_func(self.layers[i-1], self.weights[i-1], lambda x: self.sigmoid(x,False))

			# set bias values
			self.layers[i][0] = 1
		self.layers[-1] = self.dot_with_func(self.layers[-2], self.weights[-1], lambda x: self.sigmoid(x,False))
		return self.layers[-1]

	# trains the network with the given input and output data
	# returns the output
	# PUBLIC feel free to call this one
	def train(self, input_data, output_data):
		# calculate the output given the input
		y = self.get_output(input_data)
		error = [output_data[i] - y[i] for i in range(len(y))]
		error = self.backpropigate(error)
		return error[-1]

	# backpropigate the given final error
	# this is how the network learns
	# this is called from the train method, not from you
	# PRIVATE dont mess with this, you should never need to call it
	def backpropigate(self, final_error):
		# initialize the change and error
		error = [  [] for i in range(0, len(self.layers)) ]
		change = [  [] for i in range(0, len(self.layers)) ]

		

		# calculate the final error and final change
		error[-1] = final_error
		change[-1] = [error[-1][i]*self.sigmoid(self.layers[-1][i],True) for i in range(len(error[-1]))]

		# #print the error
		# #print "error is: " "{0:.5f}".format(abs(sum(error[-1])))

		# backpropigate
		for n in range(len(self.layers)-2, -1, -1):
			# transform weights[n]
			new = [[0]*len(self.weights[n]) for j in range(len(self.weights[n][0]))]
			for r in range(len(self.weights[n])):
				for c in range(len(self.weights[n][0])):
					new[c][r] = self.weights[n][r][c]
			# calculate the error and change
			error[n] = self.dot_with_func(change[n+1], (new), lambda x: x)
			change[n] = [error[n][i] * self.sigmoid(self.layers[n][i], True) for i in range(len(error[n]))]

		# add the change to the weights
		for i in range(0, len(self.weights)):
			for r in range(len(self.weights[i])):
				for c in range(len(self.weights[i][0])):
					self.weights[i][r][c] += self.layers[i][r]*change[i+1][c]
					# if(random.random() < 0.000001):
					# 	self.weights[i][r][c] = 0

		# for errornum in error:
		# 	#print errornum,"\n"


		return error

	# save the current weights to a series of files
	# PUBLIC feel free to call this one
	def save_data(self):
		num = 0
		for weight_layer in self.weights:
			num+=1
			with open("weight" + str(num) + ".txt", "w") as output_file:
				for weight in weight_layer:
					output_file.write(str(weight).replace("[","").replace("]","").replace(",","")+"\n")

	# reads in the weights from files in the current directory
	# PUBLIC feel free to call this one
	def read_data(self):
		for weight_layer in range(1, self.layer_count):
			with open("weight" + str(weight_layer) + ".txt", "r") as output_file:
				self.weights += [[]]
				for line in output_file:
					self.weights[weight_layer-1] += [[float(x) for x in line.split(" ")]]


	def learn_from_file(self, file_name, classifiers, iterations=100):
		f = open(file_name)
		inputs = []
		outputs = []
		for line in f:
			numbers = [float(i) for i in line.split(",")]
			current_in = []
			current_out = []
			for x in range(len(classifiers)):
				if classifiers[x] == "i":
					current_in += [numbers[x]]
				if classifiers[x] == "o":
					current_out += [numbers[x]]
			inputs += [current_in]
			outputs += [current_out]
		f.close()
		# print inputs
		# print outputs

		for i in range(iterations):
			if i%100 == 0:
				print i
			for j in range(len(inputs)):
				self.train(inputs[j], outputs[j])

	def validate_from_file(self, file_name, classifiers):
			f = open(file_name)
			inputs = []
			outputs = []
			for line in f:
				numbers = [float(i) for i in line.split(",")]
				current_in = []
				current_out = []
				for x in range(len(classifiers)):
					if classifiers[x] == "i":
						current_in += [numbers[x]]
					if classifiers[x] == "o":
						current_out += [numbers[x]]
				inputs += [current_in]
				outputs += [current_out]
			f.close()
			# print inputs
			# print outputs

			for j in range(len(inputs)):
				print "network calculated ", self.get_output(inputs[j]), " correct answer was ", outputs[j]




