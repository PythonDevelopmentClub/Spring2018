from better_neural_network import Neural_Network

myNN = Neural_Network([4, 8, 4, 1])

myNN.learn_from_url("https://raw.githubusercontent.com/PythonDevelopmentClub/Spring2018/master/Neural_Network/normalized_data.txt", ["i","i","i","i","o"], 500)

print myNN.validate_from_url("https://raw.githubusercontent.com/PythonDevelopmentClub/Spring2018/master/Neural_Network/validation_data.txt", ["i","i","i","i","o"])
