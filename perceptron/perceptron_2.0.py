import time
import random


ideal_weights = [-10, 7, 14, .4, 1000, 82, 3]
weights = [0.5]*(len(ideal_weights)+1)
LEARNING_RATE = .001


# Predict 
# input 1D array of input_values
# 
# 
def predict(input_values): 
	# array[start:end]
	# array[-end]
	total = weights[-1]
	# (0-len(input)]
	# range(start, end, increment)
	# range(start, end)
	for i in range(len(input_values)): 
		total += input_values[i] * weights[i]

	# IF Expression
	prediction = 0
	if total >= 0: 
		prediction = 1

	# Ternary operator
	# return 1 if total >= 0 else 0

	return prediction

# Update (Gradient Descent)
# input 1D array of input_values
# input 1D arroy of weights
# 
# 
def update(input_values, correct): 
	global weights
	error = correct - predict(input_values)

	# Squared error
	# error = error*error

	# Update weights through data input
	for i in range(len(weights)-1): 
		weights[i] = weights[i] + LEARNING_RATE * error * input_values[i]

	# Update bias
	weights[-1] = weights[-1] + LEARNING_RATE * error * 1




















# def predict(input_values):
# 	total = 0
# 	for i in range(len(input_values)):
# 		total += input_values[i]*weights[i]
# 	return (1 if total > 0 else 0)


# def update(input_values, real_output):
# 	global weights
# 	input_values += [1]
# 	error = real_output - predict(input_values)
# 	for i in range(len(weights)):
# 		weights[i] += error*LEARNING_RATE*input_values[i]










































def get_input():
	return [random.randint(-100, 100) for i in range(len(ideal_weights))]

def get_correct_output(input_values):
	total = 0
	for i in range(len(input_values)):
		total += input_values[i]*ideal_weights[i]
	return (1 if total > 0 else 0)


for i in xrange(100000000):
	# changes += update(inputs[i%len(inputs)], outputs[i%len(outputs)])
	ii = get_input()
	update(ii, get_correct_output(ii))

	if i % 1 == 0:
		# print changes
		# changes = 0
		total_e = 0.0
		total_s = 0.0
		for i in range(100):
			ii = get_input()
			if get_correct_output(ii) == predict(ii):
				total_e += 1
			# print get_correct_output(ii)
			total_s += 1
		print total_e / total_s
		time.sleep(0.1)



