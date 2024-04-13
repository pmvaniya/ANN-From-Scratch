import random  # Library for generating random numbers
import math  # Library for mathematical operations

random.seed(0)  # Set seed for reproducibility


def initialize_network(n_inputs, n_hidden, n_outputs):
    """
    Purpose: Initialize a neural network with random weights.

    Description of Input Parameters:
    - n_inputs: Number of input nodes.
    - n_hidden: Number of nodes in the hidden layer.
    - n_outputs: Number of output nodes.

    Description of Return Data:
    - network: Initialized neural network.

    Libraries Used: random
    """
    network = []
    hidden_layer = [{'weights': [random.uniform(-0.5, 0.5) for _ in range(n_inputs + 1)]} for _ in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random.uniform(-0.5, 0.5) for _ in range(n_hidden + 1)]} for _ in range(n_outputs)]
    network.append(output_layer)
    return network


def activate(weights, inputs):
    """
    Purpose: Calculate the activation of a neuron.

    Description of Input Parameters:
    - weights: List of weights for the neuron.
    - inputs: List of input values.

    Description of Return Data:
    - activation: Calculated activation value.

    Libraries Used: None
    """
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation


def sigmoid(activation):
    """
    Purpose: Compute the sigmoid activation function.

    Description of Input Parameters:
    - activation: Input value to the sigmoid function.

    Description of Return Data:
    - sigmoid_value: Result of the sigmoid function.

    Libraries Used: math
    """
    return 1.0 / (1.0 + math.exp(-activation))


def forward_propagate(network, row):
    """
    Purpose: Perform forward propagation through the neural network.

    Description of Input Parameters:
    - network: Neural network model.
    - row: Input data row.

    Description of Return Data:
    - inputs: Output values from the last layer.

    Libraries Used: None
    """
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def backward_propagate_error(network, expected):
    """
    Purpose: Backpropagate error and update neuron deltas.

    Description of Input Parameters:
    - network: Neural network model.
    - expected: Expected output values.

    Description of Return Data: None

    Libraries Used: None
    """
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j, neuron in enumerate(layer):
                errors.append(expected[j] - neuron['output'])
        for j, neuron in enumerate(layer):
            neuron['delta'] = errors[j] * neuron['output'] * (1.0 - neuron['output'])


def update_weights(network, row, learning_rate):
    """
    Purpose: Update weights of neurons based on backpropagated error.

    Description of Input Parameters:
    - network: Neural network model.
    - row: Input data row.
    - learning_rate: Learning rate for weight updates.

    Description of Return Data: None

    Libraries Used: None
    """
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += learning_rate * neuron['delta']


def train_network(network, train, learning_rate, n_epoch, n_outputs, verbose=False):
    """
    Purpose: Train the neural network using backpropagation.

    Description of Input Parameters:
    - network: Neural network model.
    - train: Training dataset.
    - learning_rate: Learning rate for weight updates.
    - n_epoch: Number of training epochs.
    - n_outputs: Number of output nodes.
    - verbose: Boolean flag for printing training progress.

    Description of Return Data: None

    Libraries Used: None
    """
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0] * n_outputs
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, learning_rate)
        if verbose:
            print(f'> epoch={epoch+1}, learning_rate={learning_rate}, error={sum_error}')


def predict(network, row):
    """
    Purpose: Make predictions using the trained neural network.

    Description of Input Parameters:
    - network: Trained neural network model.
    - row: Input data row for prediction.

    Description of Return Data:
    - prediction: Predicted class label.

    Libraries Used: None
    """
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


def accuracy_metric(actual, predicted):
    """
    Purpose: Calculate the accuracy metric for model evaluation.

    Description of Input Parameters:
    - actual: List of actual class labels.
    - predicted: List of predicted class labels.

    Description of Return Data:
    - accuracy: Accuracy percentage.

    Libraries Used: None
    """
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def check_accuracy(network, dataset):
    """
    Purpose: Check accuracy of the trained neural network on a dataset.

    Description of Input Parameters:
    - network: Trained neural network model.
    - dataset: Dataset for accuracy evaluation.

    Description of Return Data:
    - accuracy: Accuracy percentage.

    Libraries Used: None
    """
    predictions = []
    actual = [row[-1] for row in dataset]
    for row in dataset:
        prediction = predict(network, row)
        predictions.append(prediction)
    return accuracy_metric(actual, predictions)
