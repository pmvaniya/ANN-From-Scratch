from sources.Process import read_csv, encode, shuffle, split_data
from sources.ANN import initialize_network, train_network, predict, check_accuracy


# Data Loading and PreProcessing
data = read_csv("data/iris.csv")

lookup = {"setosa": 0, "versicolor": 1, "virginica": 2}
encode(data, lookup)

shuffle(data)

split_ratio = 0.8
train_set, test_set = split_data(data, split_ratio)


# Artificial Neural Network Model Training
n_inputs = len(train_set[0]) - 1
n_outputs = len(set(row[-1] for row in train_set))
learning_rate = 0.1
n_epochs = 100
network = initialize_network(n_inputs, 8, n_outputs)  # Initialize a neural network with specified parameters
train_network(network, train_set, learning_rate, n_epochs, n_outputs, verbose=False)  # Train the neural network


# Model Testing
print("Train Accuracy:", check_accuracy(network, train_set))  # Check accuracy on the training set
print("Test Accuracy:", check_accuracy(network, test_set))  # Check accuracy on the testing set

testit = [["23",4.6,3.6,1,0.2,"setosa"], ["63",6,2.2,4,1,"versicolor"], ["128",6.1,3,4.9,1.8,"virginica"]]
for i in testit:
    print(i, "=> Predicted Class:", predict(network, i[1:-1]))  # Make predictions for each sample
