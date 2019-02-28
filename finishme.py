from math import floor
import random
import time
import matplotlib.pyplot as plt

ps = [] # the array of points

# creates x,y value pairs given a function and some x values
def make_pts(function, xs):
	pts = []
	for x in xs:
		pts += [[x, get_value(function, x)]]
	return pts


# given some function and an X value, return the y value
# a function will be in array form, for example, y = 5 + 4x + 3x^2 - 2x^3 would be:
#												    [5,  4,   3,     2]
def get_value(function, x):
	return sum([function[n] * x**n for n in range(len(function))])

# given a function and some points in the form [[x1, y1], [x2, y2], [x3, y3]], return how close the function was (you can just add up all the errors?)
def fitness_function(function, points):
	total_error = 0
	for pt in points:
		total_error += abs(get_value(function, pt[0]) - pt[1])
	return total_error

# combined two genes with a slight variation
def combined_two_genes(g1, g2):
	new = # FINISH ME, average the two ints g1 and g2
	mutation = # FINISH ME, randomly pick a positive or negative int (keep it between -4 and 4 :) 
	return new + mutation


def recombined(function1, function2):
	length = # FINISH ME, pick the larger length of the two, and have a chance of adding 1 or 2 

	new_function = [0] * int(length)

	for i in range(0, len(new_function)):
		v1 = 0 if len(function1) <= i else function1[i]
		v2 = 0 if len(function2) <= i else function2[i]

		new_function[i] = # FINISH ME what should we set the new gene to?

	return new_function

all_functions = [[6], [7],[6], [7],[6], [7],[6], [7]]


def enviro_loop():

	# FINISH ME find the best two functions

	all_functions[0] = best1
	all_functions[1] = best2

	print("best is:  ", fitness_function(best1, ps), best1)
	# print "the second best is:  ", fitness_function(best2, ps), best2

	# FINISH ME recombined the best functions to create the other ones


ps = make_pts([123,-80, 40, 25, 4], [1,4,6,2,3, 20, 10, 15, 17, 18, 19])
print(ps)

def display():
	xs = [a[0] for a in ps]
	ys = [a[1] for a in ps]
	plt.plot(xs, ys, marker='o', linestyle='', color='r', label='function')

	xs_of_function = [n / 10.0 for n in range(min(xs)*10, max(xs)*10+1)]
	ys_of_function = [get_value(all_functions[0], xs_of_function[i]) for i in range(len(xs_of_function))]
	# print(xs_of_function, "\n", ys_of_function)
	plt.plot(xs_of_function, ys_of_function)
	plt.pause(0.5)
	# plt.show()



for i in range(10000):
	enviro_loop()
	# if i % 10 == 0:
	display()




