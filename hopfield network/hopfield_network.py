from random import randint
import math

class hopfield_network(object):
	"""docstring for hopfield_network"""
	def __init__(self, _num_nodes, chaos, _name="default_name"):
		self.num_nodes = _num_nodes
		self.name = _name

		self.values = [0]*self.num_nodes
		self.weights = [[0.0 if randint(0,100) < chaos else None for j in range(self.num_nodes)] for i in range(self.num_nodes)]

	def sigmoid(self, x, deriv=False):
		if(deriv):
			return x*(1-x)
		return 2/(1+ (math.e ** (-1*x))) - 1

	def train(self, new_values):
		if len(new_values) != len(self.values):
			return "Error" # TODO add some sort of error handeling here...
		for i in range(len(new_values)):
			for j in range(len(new_values)):
				if i != j and self.weights[i][j] != None and self.weights[j][i] != None:
					self.weights[i][j] += (new_values[i] * new_values[j])
					self.weights[j][i] = self.weights[i][j]

	def get_output(self, inputs, iterations):
		for i in range(len(inputs)):
			self.values[i] = inputs[i]

		for i in range(iterations):
			a = randint(0, len(self.values)-1)
			self.values[a] = self.sigmoid(sum([inputs[j] * (self.weights[a][j] if self.weights[a][j] != None else 0) for j in range(len(self.weights))]))

	def print_values(self):
		for i in self.values:
			print str(int(i*100)/100.0).zfill(4), " ",


for chaos_level in range(0, 101):
	print chaos_level, 
	best_error = None
	for iteration in range(1, 100):
		c = hopfield_network(10, chaos_level, "asdf")
		c = hopfield_network(10, chaos_level, "asdf")
		c.train( [1  ,1  ,1  ,1  ,1  ,-1  ,-1  ,-1  ,-1  ,-1  ])
		c.train( [-1  ,-1  ,1  ,-1  ,-1  ,1  ,1  ,-1  ,-1  ,1  ])
		c.train( [-1  ,-1  ,-1  ,-1  ,-1  ,1  ,-1  ,-1  ,-1  ,1  ])
		x =      [1  ,-1  ,1  ,-1  ,1  ,-1  ,1  ,-1  ,1  ,-1  ]
		c.train(x)
		c.get_output([1  ,-1  ,0  ,-1  ,1  ,1  ,1  ,-1  ,1  ,-1  ], 100)
		error = 0
		for node_num in range(c.num_nodes):
			error += abs(x[node_num] - c.values[node_num])
		# c.print_values()
		# print error
		if best_error == None or best_error > error:
			best_error = error
	print best_error