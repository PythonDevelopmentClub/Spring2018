from random import randint
import math

class hopfield_network(object):
	"""docstring for hopfield_network"""
	def __init__(self, _num_nodes, _name="default_name"):
		self.num_nodes = _num_nodes
		self.name = _name

		self.values = [0]*self.num_nodes
		self.weights = [[0.0]*self.num_nodes for i in range(self.num_nodes)]


	def sigmoid(self, x, deriv=False):
		if(deriv):
			return x*(1-x)
		return 1/(1+ (math.e ** (-1*x)))

	def train(self, new_values):
		if len(new_values) != len(self.values):
			return "Error" # TODO add some sort of error handeling here...
		for i in range(len(new_values)):
			for j in range(len(new_values)):
				if i != j:
					self.weights[i][j] += 0.5 * self.sigmoid(self.sigmoid(self.weights[i][j]), True) * (1.0 if new_values[i] == new_values[j] else -1.0)
					self.weights[j][i] = self.weights[i][j]

	def get_output(self, inputs, iterations):
		for i in range(len(inputs)):
			self.values[i] = inputs[i]

		for i in range(iterations):
			a = randint(0, len(self.values)-1)
			self.values[a] =self.sigmoid(sum([inputs[j] * self.weights[a][j] for j in range(len(self.weights))]))

	def print_values(self):
		for i in self.values:
			print str(int(i*100)/100.0), " ",

c = hopfield_network(10, "asdf")
for i in range(100):
	c.train( [1  ,1  ,1  ,0  ,0  ,1  ,0  ,0  ,1  ,1  ])
	c.train( [0  ,0  ,0  ,0  ,1  ,0  ,1  ,1  ,1  ,1  ])
c.get_output([0.5,0  ,0  ,0  ,0.5,0.5,1  ,1  ,1  ,1  ], 100)
c.print_values()
