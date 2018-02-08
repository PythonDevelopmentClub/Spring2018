from better_neural_network import Neural_Network

print "-----------------------------------------------------------"
print "welcome to the Python Development Club Machine Learning API"
print "-----------------------------------------------------------"

training_file_name = raw_input("what training dataset would you like to use? (file name)")
validation_file_name = raw_input("what validation dataset would you like to use? (file name)")


print "\n\n\n"
print """
valid algorithm options:

nn - neural network
pt - perceptron
"""
algorithm = raw_input("what algorithm would you like to use?")

if algorithm == "nn":
	pass
if algorithm == "pt":
	pass
