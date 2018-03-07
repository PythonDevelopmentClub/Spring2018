from better_neural_network import Neural_Network

myNN = Neural_Network([4, 8, 4, 1], "sam")

myNN.learn_from_url("https://raw.githubusercontent.com/PythonDevelopmentClub/Spring2018/master/Neural_Network/normalized_data.txt", ["i","i","i","i","o"], 100)

myNN.save_data()

myNN2 = Neural_Network([4, 8, 4, 1], "sam")

myNN2.read_data()


print myNN2.validate_from_url("https://raw.githubusercontent.com/PythonDevelopmentClub/Spring2018/master/Neural_Network/validation_data.txt", ["i","i","i","i","o"])
