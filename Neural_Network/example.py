from better_neural_network import Neural_Network

# myNN = Neural_Network([4, 8, 4, 1])

# myNN.learn_from_file("normalized_data.txt", ["i","i","i","i","o"], 500)

# print myNN.validate_from_file("validation_data.txt", ["i","i","i","i","o"])

myNN = Neural_Network([2, 2, 1])

myNN.train([1,1], [0])

# print myNN.sigmoid(0.551072636233645, True)