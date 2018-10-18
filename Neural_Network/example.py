from better_neural_network import Neural_Network


myNN = Neural_Network([4, 8, 4, 1], "sam")



# Does train, then get_output and backpropogate based on error of output
myNN.learn_from_url("https://raw.githubusercontent.com/PythonDevelopmentClub/Spring2018/master/Neural_Network/normalized_data.txt", ["i","i","i","i","o"], 100)


# Test values
# print myNN.get_output([0.722,0.458,0.661,0.583])  #0
# print myNN.get_output([0.333,0.125,0.508,0.500])  #0
# print myNN.get_output([0.194,0.625,0.102,0.208])  #1
# print myNN.get_output([0.222,0.750,0.153,0.125])  #1

# myNN.save_data()

# myNN2 = Neural_Network([4, 8, 4, 1], "sam")

# myNN2.read_data()


print myNN.validate_from_url("https://raw.githubusercontent.com/PythonDevelopmentClub/Spring2018/master/Neural_Network/validation_data.txt", ["i","i","i","i","o"])


# Convolutional Stuff
