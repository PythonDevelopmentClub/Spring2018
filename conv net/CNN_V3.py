import random
import math
from better_neural_network import Neural_Network
import time
import urllib2
random.seed(1234)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# how matricies work
# matrix[x][y][z]
# x = len(matrix)       = w
# y = len(matrix[0])    = l
# z = len(matrix[0][0]) = h
# use x y and z for actual values, and w, l and h for the lengths of things

# takes a three dimensional matrix and turns it into a one dimensional matrix
# basically does the inverse of oned_to_threed
def threed_to_oned(input_matrix):
	return [input_matrix[x][y][z] for x in range(len(input_matrix)) for y in range(len(input_matrix[0])) for z in range(len(input_matrix[0][0]))]

def softmax(arr):
	arr = [math.exp(i) for i in arr]
	s = sum(arr)
	return [i / s for i in arr]

def fully_connected_multiply(input_matrix, weights):
	new_matrix = make_matrix(len(weights[0]))

	for x in range(len(input_matrix)):
		for y in range(len(weights[0])):
			new_matrix[y] += input_matrix[x] * weights[x][y]

	return new_matrix

def function_on_array(input_arr, func):
	return [func(i) for i in input_arr]



def fully_connected_backwards(input_matrix, weights, error_above):
	# the change in the weights is f'(input_matrix) * error
	# the new error is the change * the weights

	transformed_input_matrix = function_on_array(input_matrix, lambda x: ReLU(x, True))

	change = [[transformed_input_matrix[x] * error_above[y] for x in range(len(transformed_input_matrix))] for y in range(len(error_above))]

	# print "-------------------"
	# pretty_print([weights])
	# print input_matrix
	# print error_above

	new_weights = [[weights[y][x] + change[x][y]*0.00001 for x in range(len(change))] for y in range(len(change[0]))]

	new_error = [sum([weights[y][x] * change[x][y] for x in range(len(change))]) for y in range(len(change[0]))]

	return (new_weights, new_error)

# takes a one dimensional matrix and turns it into a three dimensional matrix
# basically does the inverse of threed_to_oned
def oned_to_threed(input_matrix, w, l, h):
	return [[[input_matrix[z + y*h + x*l*h] for z in range(h)] for y in range(l)] for x in range(w)]


def pretty_print(matrix):
	for i in range(len(matrix)):
		print "############### depth", i, "################", len(matrix[i]), "by", len(matrix[i][0])
		for r in range(len(matrix[i])):
			for c in range(len(matrix[i][r])):
				print ("%.2f" % matrix[i][r][c]).zfill(5),
			print ""

# computes a sub matrix
# from x_1 to x_2, y_1 to y_2, and z_1 to z_2, inclusive
def sub_matrix(input_matrix, x_1, x_2, y_1=None, y_2=None, z_1=None, z_2=None):

	if y_1 == None and y_2 == None and z_1 == None and z_2 == None:
		return input_matrix[x_1:x_2+1] # add zfill here sometime in the future
	elif z_1 == None and z_2 == None:
		output_matrix = make_matrix(x_2 - x_1 + 1, y_2 - y_1 + 1)
		for x in range(x_2 - x_1 + 1):
			for y in range(y_2 - y_1 + 1):
				if x + x_1 < len(input_matrix) and y + y_1 < len(input_matrix[0]):
					output_matrix[x][y] = input_matrix[x + x_1][y + y_1]
				else:
					output_matrix[x][y] = 0
		return output_matrix
	else:
		output_matrix = make_matrix(x_2 - x_1 + 1, y_2 - y_1 + 1, z_2 - z_1 + 1)

		for x in range(x_2 - x_1 + 1):
			for y in range(y_2 - y_1 + 1):
				for z in range(z_2 - z_1 + 1):
					if x + x_1 < len(input_matrix) and y + y_1 < len(input_matrix[0]) and z + z_1 < len(input_matrix[0][0]):
						output_matrix[x][y][z] = input_matrix[x + x_1][y + y_1][z + z_1]
					else:
						output_matrix[x][y][z] = 0
		return output_matrix

# multiplies two matricies together
def multiply(matrix_a, matrix_b):
	w = len(matrix_a)
	l = len(matrix_a[0])
	h = len(matrix_a[0][0])
	return [[[matrix_a[x][y][z]*matrix_b[x][y][z] for z in range(h)] for y in range(l)] for x in range(w)]

def sum_matrix(matrix):
	return sum(threed_to_oned(matrix))

def find_max_location(two_d_matrix):
	maxs = [max(two_d_matrix[r]) for r in range(len(two_d_matrix))]

	r = maxs.index(max(maxs))

	c = two_d_matrix[r].index(maxs[r])

	return (r, c)

def find_max_value(two_d_matrix):
	maxs = [max(two_d_matrix[r]) for r in range(len(two_d_matrix))]


	return max(maxs)

def make_matrix(x, y=0, z=0):
	if y == 0 and z == 0:
		return [0 for i in range(x)]
	elif z == 0:
		return [[0 for i in range(y)] for j in range(x)]
	else:
		return [[[0 for i in range(z)] for j in range(y)] for k in range(x)]

def make_random_matrix(x, y=0, z=0):
	if y == 0 and z == 0:
		return [random.gauss(0, 0.05) for i in range(x)]
	elif z == 0:
		return [[random.gauss(0, 0.05)  for i in range(y)] for j in range(x)]
	else:
		return [[[random.gauss(0,0.05) for i in range(z)] for j in range(y)] for k in range(x)]

def find_max_locations_of_two_d(two_d_matrix, step_size):
	answer = make_matrix(len(two_d_matrix) / step_size, len(two_d_matrix[0]) / step_size)



	for x in range(0, len(two_d_matrix), step_size):
		for y in range(0, len(two_d_matrix[0]), step_size):
			answer[x / step_size][y / step_size] = find_max_location(sub_matrix(two_d_matrix, x, x + step_size - 1, y, y + step_size - 1))
	return answer

def find_max_values_of_two_d(two_d_matrix, step_size):
	answer = make_matrix(len(two_d_matrix) / step_size, len(two_d_matrix[0]) / step_size)



	for x in range(0, len(two_d_matrix), step_size):
		for y in range(0, len(two_d_matrix[0]), step_size):
			answer[x / step_size][y / step_size] = find_max_value(sub_matrix(two_d_matrix, x, x + step_size - 1, y, y + step_size - 1))


	return answer

def forward_max_pool(input_matrix, step_size):
	result = ([], [])
	for f in range(len(input_matrix)):
		result[0].append(find_max_locations_of_two_d(input_matrix[f], step_size))
		result[1].append(find_max_values_of_two_d(input_matrix[f], step_size))

	return result

def backwards_max_pool(max_locations, step_size, error_matrix):
	new_result = make_matrix(len(max_locations), len(max_locations[0])*step_size, len(max_locations[0][0])*step_size)

	for x in range(len(new_result)):
		for y in range(0, len(new_result[0]), step_size):
			for z in range(0, len(new_result[0][0]), step_size):
				offset = max_locations[x][y/step_size][z/step_size]
				# pretty_print(error_matrix)
				# print x, y/step_size, z/step_size
				new_result[x][y + offset[0]][z + offset[1]] = error_matrix[x][y/step_size][z/step_size]
	return new_result

def ReLU(x, deriv=False):
	if deriv:
		return 1 if x > 0 else 0
	return max(0, x)

def sigmoid(x, deriv=False):
	if(deriv):
		return x*(1-x)
	return 1/(1+ (math.e ** (-1*x)))

def function_on_matrix(input_matrix, func):

	answer = make_matrix(len(input_matrix), len(input_matrix[0]), len(input_matrix[0][0]))

	for x in range(len(input_matrix)):
		for y in range(len(input_matrix[0])):
			for z in range(len(input_matrix[0][0])):
				answer[x][y][z] = func(input_matrix[x][y][z])
	return answer

def forward_convolution(input_matrix, filters, step_size):
	filter_diameter = len(filters[0][0])
	output_matrix = make_matrix(len(filters),
							    len(input_matrix[0]) / step_size, 
							    len(input_matrix[0][0]) / step_size)
	input_matrix = z_fill(input_matrix, filter_diameter-1)
	# pretty_print(input_matrix)
	for f in range(len(filters)):
		for y in range(filter_diameter, len(input_matrix[0])-filter_diameter, step_size):
			for z in range(filter_diameter, len(input_matrix[0][0])-filter_diameter, step_size):

				sub_input_matrix = sub_matrix(input_matrix, 0, len(input_matrix), y, y+filter_diameter-1, z, z+filter_diameter -1)
				# print "CONFIRM FORWARD CONVOLUTION"
				# pretty_print(sub_input_matrix)
				# print "---"
				# pretty_print(filters[f])
				# print "==="
				# print ReLU(sum_matrix(multiply(filters[f], sub_input_matrix)))
				# asdf = raw_input()
				output_matrix[f][(y - filter_diameter) / step_size][(z - filter_diameter) / step_size] = ReLU(sum_matrix(multiply(filters[f], sub_input_matrix)))
	return output_matrix

def multiply_matrix_by_scalar(input_matrix, scalar_val):
	w = len(input_matrix)
	l = len(input_matrix[0])
	h = len(input_matrix[0][0])
	return [[[input_matrix[x][y][z]*scalar_val for z in range(h)] for y in range(l)] for x in range(w)]

def z_fill(input_matrix, num_zeros):
	output = make_matrix(len(input_matrix), len(input_matrix[0]) + 2*num_zeros, len(input_matrix[0][0])+2*num_zeros)
	for x in range(len(input_matrix)):
		for y in range(len(input_matrix[0])):
			for z in range(len(input_matrix[0][0])):
				output[x][y+num_zeros][z+num_zeros] = input_matrix[x][y][z]
	return output

def copy_matrix(input_matrix):
	w = len(input_matrix)
	l = len(input_matrix[0])
	h = len(input_matrix[0][0])
	return [[[input_matrix[x][y][z] for z in range(h)] for y in range(l)] for x in range(w)]

def add_matricies(matrix_a, matrix_b):
	w = len(matrix_a)
	l = len(matrix_a[0])
	h = len(matrix_a[0][0])
	return [[[matrix_a[x][y][z]+matrix_b[x][y][z] for z in range(h)] for y in range(l)] for x in range(w)]

def backwards_convolution(input_matrix, filters, step_size, final_error):
	lower_error = make_matrix(len(input_matrix), len(input_matrix[0]), len(input_matrix[0][0]))
	filter_diameter = len(filters[0][0])
	new_filters = [copy_matrix(filters[f]) for f in range(len(filters))]
	for f in range(len(filters)):
		for y in range(0, len(input_matrix[0]), step_size):
			for z in range(0, len(input_matrix[0][0]), step_size):

				sub_input_matrix = sub_matrix(input_matrix, 0, len(input_matrix), y, y+filter_diameter-1, z, z+filter_diameter -1)
				change = multiply_matrix_by_scalar(function_on_matrix(sub_input_matrix, lambda x: ReLU(x, True)), final_error[f][y/step_size][z/step_size])
				# pretty_print(new_filters[f])
				# pretty_print(change)
				scaled_change = multiply_matrix_by_scalar(change, 0.01)
				scaled_change = multiply(scaled_change, sub_input_matrix)
				new_filters[f] = add_matricies(new_filters[f], scaled_change)
				# pretty_print(new_filters[f])
				# time.sleep(1)

				error = multiply(filters[f], change)

				for ff in range(len(input_matrix)):
					for dy in range(filter_diameter):
						for dz in range(filter_diameter):
							if y + dy < len(lower_error[ff]) and z + dz < len(lower_error[ff][0]):
								lower_error[ff][y + dy][z + dz] += error[ff][dy][dz]

	return (new_filters, lower_error)

from sklearn.datasets import fetch_mldata
import numpy as np
# import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split

mnist = fetch_mldata('MNIST original')

train_img, test_img, train_lbl, test_lbl = train_test_split(
    mnist.data, mnist.target, test_size=1/7.0, random_state=2)

layer_1_filters = [make_random_matrix(1, 5, 5) for i in range(4)]

layer_3_filters = [make_random_matrix(4, 5, 5) for i in range(4)]

layer_5 = Neural_Network([49*4, 10])

# ok so now we gotta make this object oriented...
# well not neccesarily, we DO however need to make it all parameterized
def make_chain():
	pass

def process_chain():
	pass

def train(initial_input_matrix, label):
	global layer_3_filters, layer_1_filters
	correct_output = [0 for i in range(10)]
	correct_output[label] = 1

	layer_1_output = forward_convolution(initial_input_matrix, layer_1_filters, 1)
	layer_2_output = forward_max_pool(layer_1_output, 2)
	layer_3_output = forward_convolution(layer_2_output[1], layer_3_filters, 1)
	layer_4_output = forward_max_pool(layer_3_output, 2)
	final_output = layer_5.get_output(threed_to_oned(layer_4_output[1]))
	final_error = [final_output[i] - correct_output[i] for i in range(len(final_output))] # FUCK THIS IS WRONG OMFG AND ITS SO SIMPLE TOO!!!
	#																						no its good now :)

	final_error = layer_5.train(threed_to_oned(layer_4_output[1]), correct_output)[0]

	layer_4_error = backwards_max_pool(layer_4_output[0], 2, oned_to_threed(final_error, 4, 7, 7))
	layer_3_error = backwards_convolution(layer_2_output[1], layer_3_filters, 1, layer_4_error)
	layer_3_filters = layer_3_error[0]
	layer_2_error = backwards_max_pool(layer_2_output[0], 2, layer_3_error[1])
	layer_1_error = backwards_convolution(initial_input_matrix, layer_1_filters, 1, layer_2_error)
	layer_1_filters = layer_1_error[0]
	return final_output

last_100_results = []
for index, (image, label) in enumerate(zip(train_img, train_lbl)):
	# if label == 0 or label == 1:
		# for i in range(1):
	results = train( [np.multiply(np.reshape(image, (28, 28)), 1/256.0)], int(label))
	color = bcolors.FAIL
	best = results.index(max(results))
	if len(set(results)) >= 2:
		second_best_num = sorted(set(results))[-2]
		if best == int(label) and max(results) > second_best_num + 0.3:
			color = bcolors.OKGREEN
			last_100_results += [1]

		elif  best == int(label):
			color = bcolors.WARNING
			last_100_results += [0.5]
		else:
			last_100_results += [0]

	if len(last_100_results) > 100:
		del last_100_results[0]
	# content = urllib2.urlopen('http://unertech.com/secret_projects/conv%20net/something.py?progress=' + str(sum(last_100_results)/100.0))
	print color, label, ": " + ' '.join(['%.2f' % (k) for k in results])
	i += 1
	if i % 5 == 0:
		print "       0    1    2    3    4    5    6    7    8    9"
		

# print softmax([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])


# def get_data():
# 	input_data = [random.random() * 100, random.random()* 100, random.random() * 1000]

# 	output_data = 1 if input_data[0]* 10 - input_data[1]*4 + input_data[2]*3 > 1800 else 0

# 	return input_data + [output_data]

# layer_1_weights = make_random_matrix(9, 13)

# layer_2_weights = make_random_matrix(13, 9)

# layer_3_weights = make_random_matrix(9, 1)
# for i in range(10000):
# 	print i
# 	input_number = get_data()
# 	input_arr = input_number[:-1]
# 	input_arr += [1]
# 	# print len(input_arr)
# 	output_arr = [input_number[-1]]

# 	# pretty_print([[input_arr]])
# 	# time.sleep(1)
# 	layer_1_output = fully_connected_multiply(input_arr, layer_1_weights)
# 	layer_1_output[-1] = 1
# 	# pretty_print([[layer_1_output]])
# 	# time.sleep(1)
# 	layer_2_output = fully_connected_multiply(layer_1_output, layer_2_weights)
# 	layer_2_output[-1] = 1

# 	# pretty_print([[layer_2_output]])
# 	# time.sleep(1)

# 	layer_3_output = fully_connected_multiply(layer_2_output, layer_3_weights)
# 	layer_3_output[-1] = 1

# 	# pretty_print([[layer_3_output]])
# 	# time.sleep(1)

# 	# print "computing error..."
# 	error = [output_arr[0] - layer_3_output[0]]

# 	# pretty_print([[error]])
# 	# print layer_3_output, output_arr
# 	layer_3_error_output = fully_connected_backwards(layer_2_output, layer_3_weights, error)
# 	layer_3_weights = layer_3_error_output[0]
# 	layer_3_error = layer_3_error_output[1]
# 	# pretty_print([[layer_3_error]])
# 	# time.sleep(1)



# 	layer_2_error_output = fully_connected_backwards(layer_1_output, layer_2_weights, layer_3_error)
# 	layer_2_weights = layer_2_error_output[0]
# 	layer_2_error = layer_2_error_output[1]
# 	# pretty_print([[layer_2_error]])
# 	# time.sleep(1)

# 	layer_1_error_output = fully_connected_backwards(input_arr, layer_1_weights, layer_2_error)
# 	layer_1_weights = layer_1_error_output[0]
# 	layer_1_error = layer_1_error_output[1]
# 	# pretty_print([[layer_1_error]])
# 	# time.sleep(1)

# for i in range(100):
# 	input_number = get_data()
# 	input_arr = input_number[:-1] + [1]
# 	output_arr = [input_number[-1]]

# 	# pretty_print([[input_arr]])
# 	# time.sleep(1)
# 	layer_1_output = fully_connected_multiply(input_arr, layer_1_weights)
# 	layer_1_output[-1] = 1
# 	# pretty_print([[layer_1_output]])
# 	# time.sleep(1)
# 	layer_2_output = fully_connected_multiply(layer_1_output, layer_2_weights)
# 	layer_2_output[-1] = 1

# 	# pretty_print([[layer_2_output]])
# 	# time.sleep(1)

# 	layer_3_output = fully_connected_multiply(layer_2_output, layer_3_weights)
# 	# layer_3_output[-1] = 1

# 	# pretty_print([[layer_3_output]])
# 	# time.sleep(1)

# 	# print "computing error..."
# 	error = [output_arr[0] - layer_3_output[0]]

# 	# pretty_print([[error]])
# 	print layer_3_output, output_arr
# 	layer_3_error_output = fully_connected_backwards(layer_2_output, layer_3_weights, error)
# 	layer_3_weights = layer_3_error_output[0]
# 	layer_3_error = layer_3_error_output[1]
# 	# pretty_print([[layer_3_error]])
# 	# time.sleep(1)



# 	layer_2_error_output = fully_connected_backwards(layer_1_output, layer_2_weights, layer_3_error)
# 	layer_2_weights = layer_2_error_output[0]
# 	layer_2_error = layer_2_error_output[1]
# 	# pretty_print([[layer_2_error]])
# 	# time.sleep(1)

# 	layer_1_error_output = fully_connected_backwards(input_arr, layer_1_weights, layer_2_error)
# 	layer_1_weights = layer_1_error_output[0]
# 	layer_1_error = layer_1_error_output[1]
# 	# pretty_print([[layer_1_error]])
# 	# time.sleep(1)

# input_arr = [random.random() for i in range(8*49)]

# for i in range(1000):
# 	layer_5.train(input_arr, [0,0,0,0,0,0,1,0,0,0,0])
# 	print ' '.join(['%.2f' % (k) for k in layer_5.get_output(input_arr)])
# 	# time.sleep(0.1)
