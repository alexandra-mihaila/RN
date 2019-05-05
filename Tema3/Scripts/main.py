import itertools
import random
import gzip
from queue import Queue

import numpy
import pickle


__author__ = "Mihaila Alexandra Ioana"
__version__ = "1.1"
__status__ = "Dev"

qiu = Queue()
max_accuracy = 0
max_learning_rate = 0
max_lambda = 0
max_mini_batch_size = 0


def load_db(file_name):
    try:
        with gzip.open(file_name, 'rb') as fin:
            train_set, valid_set, test_set = pickle.load(fin, encoding='latin1')
            return train_set, valid_set, test_set
    except Exception as error:
        print(error)
        return None, None, None


def get_vector_from_digit(y):
    vector = numpy.zeros((10, 1))
    vector[y] = 1.0

    return vector


def format_data():
    train_set, valid_set, test_set = load_db(r'..\Utils\mnist.pkl.gz')

    # transform input image to numpy array of size [784][1] (for easier calculation between layers)
    training_inputs = [numpy.reshape(x, (784, 1)) for x in train_set[0]]
    # digit-vector (e.g. 2 -> [0,0,1,0,0,0,0,0,0,0]
    training_results = [get_vector_from_digit(y) for y in train_set[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [numpy.reshape(x, (784, 1)) for x in valid_set[0]]
    validation_data = zip(validation_inputs, valid_set[1])

    test_inputs = [numpy.reshape(x, (784, 1)) for x in test_set[0]]
    test_data = zip(test_inputs, test_set[1])

    return training_data, validation_data, test_data


def validation_f():
    global qiu

    learning_rate = [i / 10 for i in range(1, 10)]
    lambda_ = learning_rate.copy()  # regularization parameter
    mini_batches_size = [i for i in range(5, 15)]
    cartesian_product = itertools.product(learning_rate, lambda_, mini_batches_size)

    for x in cartesian_product:
        qiu.put(x)


def test_cartesian_product():
    global training_data, test_data, validation_data, qiu, max_accuracy, max_mini_batch_size, max_lambda, \
        max_learning_rate

    number_of_iterations = 1
    n = 100

    while qiu.qsize():
        item = qiu.get()
        learning_rate = item[0]
        lambda_ = item[1]
        mini_batch_size = item[2]

        biases = [numpy.random.randn(y, 1) for y in [100, 10]]  # [[100][1],[10][1]]
        weights = [numpy.random.randn(y, x) / numpy.sqrt(x) for x, y in
                   [(784, 100), (100, 10)]]  # [[100][784],[10][100]]

        for iteration in range(number_of_iterations):
            random.shuffle(training_data)

            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                nabla_b = [numpy.zeros(b.shape) for b in biases]  # [[100][1] , [10][1]]
                # [[100][784],[10][100]] mini batch total weight gradient : tells us how to modify the weights that come into each neuron
                nabla_w = [numpy.zeros(w.shape) for w in weights]

                for x, y in mini_batch:
                    # gradienti pentru un input; delta_nabla_b si delta_nabla_w
                    bias_gradient, weight_gradient = backpropagation(x, y, biases, weights)

                    # gradienti pentru tot mini batch-ul (suma de 10 gradienti)
                    nabla_b = [nb + bg for nb, bg in zip(nabla_b, bias_gradient)]
                    nabla_w = [nw + wg for nw, wg in zip(nabla_w, weight_gradient)]

                weights = [
                    (1 - learning_rate * (lambda_ / n)) * w -
                    (learning_rate / len(mini_batch)) * nw for w, nw in zip(weights, nabla_w)
                ]
                biases = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(biases, nabla_b)]

            number_of_correct_outputs = test_accuracy(validation_data, biases, weights)
            accuracy = float(number_of_correct_outputs) / 100
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                max_learning_rate = learning_rate
                max_lambda = lambda_
                max_mini_batch_size = mini_batch_size

            print("Iteration {}: {} %".format(iteration + 1, accuracy))


def learn():
    global training_data, test_data

    learning_rate = 0.7
    lambda_ = 0.4
    number_of_iterations = 10
    mini_batch_size = 5
    n = 50000
    # training_data = list(training_data)
    # test_data = list(test_data)

    # biases-uri pentru al doilea layer si ultimul (distributie normala)
    biases = [numpy.random.randn(y, 1) for y in [100, 10]]
    # costurile pentru fiecare layer (distributie normala)
    weights = [numpy.random.randn(y, x) / numpy.sqrt(x) for x, y in [(784, 100), (100, 10)]]

    for iteration in range(number_of_iterations):
        random.shuffle(training_data)

        mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            nabla_b = [numpy.zeros(b.shape) for b in biases]  # [[100][1], [10][1]]
            # [[100][784], [10][100]] mini batch total weight gradient : tells us how to modify the weights that come into each neuron
            nabla_w = [numpy.zeros(w.shape) for w in weights]

            for x, y in mini_batch:
                bias_gradient, weight_gradient = backpropagation(x, y, biases, weights)  # gradients for just one input
                # gradients for the total mini batch (sum of 10 gradients)
                nabla_b = [nb + bg for nb, bg in zip(nabla_b, bias_gradient)]
                nabla_w = [nw + wg for nw, wg in zip(nabla_w, weight_gradient)]

            weights = [
                (1 - learning_rate * (lambda_ / n)) * w -
                (learning_rate / len(mini_batch)) * nw for w, nw in zip(weights, nabla_w)
            ]
            biases = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(biases, nabla_b)]

        number_of_correct_outputs = test_accuracy(test_data, biases, weights)
        accuracy = float(number_of_correct_outputs) / 100

        print("Iteration {}: {} %".format(iteration + 1, accuracy))


def backpropagation(x, y, biases, weights):

    # activari
    z1 = x
    activation_1 = z1

    z2 = numpy.dot(weights[0], activation_1) + biases[0]  # input - ul fiecarui neuron de la layer 2
    # activarea fiecarui neuron de la layer 2, sigmoid(w_layer2 * activation_layer1 + bias_layer2) = z2
    activation_2 = sigmoid(z2)

    z3 = numpy.dot(weights[1], activation_2) + biases[1]  # inputul fiecarui neuron de la layer 3
    # activaticarea fiecarui neuron de la layer 3, sigmoid(w_layer3* activation_layer2 + bias_layer3) = z3
    activation_3 = softmax(z3)


    # eroare pentru layer 3 (cross entropy)
    delta_23 = activation_3 - y
    bias_gradient23 = delta_23
    # gradienti pentru costuri de la layer 2 la layer 3
    weight_gradient23 = numpy.dot(delta_23, activation_2.transpose())

    # eroare pentru layer 2
    delta_12 = numpy.dot(weights[1].transpose(), delta_23) * sigmoid_prime(z2)
    bias_gradient12 = delta_12
    # gradienti pentru greutati de la layer 1 la layer 2
    weight_gradient12 = numpy.dot(delta_12, activation_1.transpose())

    bias_gradient = [bias_gradient12, bias_gradient23]
    weight_gradient = [weight_gradient12, weight_gradient23]

    return bias_gradient, weight_gradient  # gradient for the cost function C_x: delta_nabla_b, delta_nabla_w


def test_accuracy(test_data, biases, weights):
    number_of_correct_outputs = 0
    for x, y in test_data:
        # returns a vector [10] where each element(digit) has a probability of being the output
        output = find_output(x, biases, weights)
        digit = numpy.argmax(output)  # get the digit with the highest probability
        if digit == y:
            number_of_correct_outputs += 1

    return number_of_correct_outputs


def find_output(x, biases, weights):
    activation_2 = sigmoid(numpy.dot(weights[0], x) + biases[0])
    activation_3 = softmax(numpy.dot(weights[1], activation_2) + biases[1])

    return activation_3


def sigmoid(z):
    return 1.0 / (1.0 + numpy.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z):
    exp_sum = sum(numpy.exp(zk) for zk in z)
    return numpy.exp(z) / exp_sum


if __name__ == '__main__':
    training_data, validation_data, test_data = format_data()
    training_data = list(training_data)
    test_data = list(test_data)
    validation_data = list(validation_data)

    # finding the best constants for learning
    # validation_f()
    # print(qiu.qsize())
    # thds = list()
    # for _ in range(30):
    #     thread = Thread(target=test_cartesian_product)
    #     thread.start()
    #     thds.append(thread)
    #
    # for t in thds:
    #     t.join()
    # print(max_accuracy, max_learning_rate, max_lmbda, max_mini_batch_size)
    learn()
