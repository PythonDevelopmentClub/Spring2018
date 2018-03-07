from random import randint

ps = [[1,2], [3,4], [5,6]]


# [5, 6, 2]
# 5 + 6x + 2x^2
# 2 + 4x

def get_value(function, x):
	return sum([function[n] * x**n for n in range(len(function))])


def fitness_function(function, points):
	total_error = 0
	for pt in points:
		total_error += abs(get_value(function, pt[0]) - pt[1])
	return total_error


def combined_two_genes(g1, g2):
	new = floor((g1 + g2)/2)
	mutation = floor(randint(-2,2)**2)
	return new + mutation


def recombined(function1, function2):
	length = combined_two_genes(len(function1), len(function2))

	new_function = [0] * length

	for i in range(0, len(new_function)):
		v1 = 0 if len(function1) <= i else function1[i]
		v2 = 0 if len(function2) <= i else function2[i]

		new_function[i] = combined_two_genes(v1, v2)

	return new_function


all_functions = [[], [], [], []]

def enviro_loop():

	best1 = all_functions[0]
	best2 = all_functions[1]

	for func in all_functions:

		if fitness_function(func) < fitness_function(best1):
			best2 = best1
			best1 = func
		elif fitness_function(func) < fitness_function(best2):
			best1 = func

	all_functions[0] = best1
	all_functions[1] = best2

	for i in range(2, len(all_functions)):
		all_functions[i] = recombined(best1, best2)


for i in range(100000000):
	enviro_loop()

