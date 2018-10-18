# for random intializatin of weights
# also for randomly zeroing weights during runtime
import random, math, urllib2, json

random.seed(1)



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
	def __init__(self, num_neurons_per_layer_, name="default_name"):
		# the number of layers in the neural network. i.e. the length of the list supplied as argument 1
		# Neural_Network([2, 3, 2, 1]) would be layer_count 4
		self.layer_count = len(num_neurons_per_layer_)

		# This initializes all the layers to 0 such that
		# a network declared with the following would have these layers: 
		# myNN = Neural_Network([4, 8, 4, 1], "sam")
		# [[0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0], [0]]
		self.layers = [[0]*layer_len for layer_len in num_neurons_per_layer_]
		# print [[0]*layer_len for layer_len in num_neurons_per_layer_]

		# initialize the weights of the neural network, try to read in the weights, and if they are there great, if not, make new ones
		self.weights = []
		try:
			self.read_data()
		except:
			#print "failed loading past weights, creating new ones..."
			# An example of weights would be as follows for network myNN = Neural_Network([4, 8, 4, 1], "sam"): 
			# Notice the output layer does not need any weights going from it, but the layer connecting to the output layer does. 
			 # Also, each layer has enough weights to connect to all nodes in the next layer
			# [
			# 	[
			# 		[-0.7312715117751976, 0.6948674738744653, 0.5275492379532281, -0.4898619485211566, -0.009129825816118098, -0.10101787042252375, 0.3031859454455259, 0.5774467022710263]
			# 		, 
			# 		[-0.8122808264515302, -0.9433050469559874, 0.6715302078397394, -0.13446586418989326, 0.524560164915884, -0.9957878932977786, -0.10922561189039715, 0.44308006468156513]
			# 		, 
			# 		[-0.5424755574590947, 0.8905413911078446, 0.8028549152229671, -0.9388200339328929, -0.9491082780130784, 0.08282494558699316, 0.8782983255570211, -0.23759152462357513]
			# 		,
			# 		[-0.5668012057387732, -0.15576684883456537, -0.9419184248502641, -0.5566166674539299, -0.12422481269885588, -0.008375517236298702, -0.5338310994848547, -0.5382669169180314]

			# 	]
			# 	, 
			# 	[
			# 		[-0.5624379253246228, -0.08079306852453283, -0.42043677081902886, -0.9570205894681822]
			# 		, 
			# 		[0.6751559513251457, 0.11290864530486688, 0.28458872586489115, -0.6281874682105646]
			# 		,
			# 		[0.9850868243521302, 0.7198930575905798, -0.7582200803883872, -0.3346096292797418]
			# 		, 
			# 		[0.44296881516653674, 0.4223835393905593, 0.8728811735989193, -0.15578600007716958]
			# 		, 
			# 		[0.660071386548654, 0.34061113282814204, -0.3932629781341648, 0.1751612122871189]
			# 		, 
			# 		[0.7649580016637154, 0.6923948368566255, 0.010567641159200836, 0.17800451596510336]
			# 		, 
			# 		[-0.9309483396973168, -0.5145200529138647, 0.5948084951086057, -0.17137200139845143]
			# 		, 
			# 		[-0.6539851968418982, 0.09759752277630596, 0.40608152413126297, 0.3489716610046545]
			# 	]
			# 	, 
			# 	[
			# 		[-0.25059395899671943]
			# 		, 
			# 		[-0.12207673991087375]
			# 		, 
			# 		[0.016852976499963646]
			# 		, 
			# 		[0.5568852300002916]
			# 	]
			# ]
			for i in range(0, len(num_neurons_per_layer_)-1):
				self.weights += [[     [   random.uniform(-1,1) for x in range(num_neurons_per_layer_[i+1])  ]     for j in range(0,num_neurons_per_layer_[i])    ]]
			
		self.name = name


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
	# import numpy
	# result = numpy.dot( numpy.array(A)[:,0], B)
	def dot_with_func(self, x, y, func):
		new = []
		for c in range(0,len(y[0])):
			new += [func(sum([x[r]*y[r][c] for r in range(0, len(y))])
						)
					]
		return new
		# new = []
		# for r in range(len(x)): 
		# 	new += [func(sum([x[r]*y[r][c] for c in range(0, len(y[0]))]))]
		# return new

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
			# self.layers[i][0] = 1
			# Negative list indexes mean count from the right, so self.layers[-1] is the last element aka the output layer
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
		# print "error is:", "{0:.5f}".format(abs(sum(error[-1])))


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
			# print i
			# print "weights are: ", self.weights[i]
			# print "layers are: ", self.layers[i]
			# print "change is: ", change[i+1]
			# print "error is: ", error[i]
		# for errornum in error:
		# 	#print errornum,"\n"
		# print 2
		# print "layers are: ", self.layers[2]
		# print "error is: ", error[2]

		return error

	# save the current weights to a series of files
	# PUBLIC feel free to call this one
	def save_data(self):
		with open(self.name + "_weights.json", "w") as f:
			json.dump(self.weights, f)


	# reads in the weights from files in the current directory
	# PUBLIC feel free to call this one
	def read_data(self):
		with open(self.name + "_weights.json", "r") as f:
			self.weights = json.load(f)


	def learn_from_url(self, link, classifiers, iterations=100):
		f = urllib2.urlopen(link).read()
		inputs = []
		outputs = []
		for line in f.split("\n"):
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

		
		for i in range(iterations):
			# if i%100 == 0:
			# 	print i
			for j in range(len(inputs)):
				self.train(inputs[j], outputs[j])

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


	def validate_from_url(self, link, classifiers):
		total_errors = []
		f = urllib2.urlopen(link).read()
		inputs = []
		outputs = []
		for line in f.split("\n"):
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
		# print inputs
		# print outputs

		for j in range(len(inputs)):
			print "network calculated ", self.get_output(inputs[j]), " correct answer was ", outputs[j]
			total_errors += [sum(abs(outputs[j][a] - self.get_output(inputs[j][a])) for a in range(len(outputs[j])))]
		return sum(total_errors) / len(total_errors)

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




