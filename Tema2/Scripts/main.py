import gzip
from copy import deepcopy

import numpy
import pickle


__author__ = "Mihaila Alexandra Ioana"
__version__ = "1.1"
__status__ = "Dev"


def load_db(file_name):
    try:
        with gzip.open(file_name, 'rb') as fin:
            train_set, valid_set, test_set = pickle.load(fin, encoding='latin1')
            return train_set, valid_set, test_set

    except Exception as error:
        print(error)
        return None, None, None


def format_data(train_set, test_set):
    training_data = zip(train_set[0], train_set[1])
    test_data = zip(test_set[0], test_set[1])

    return training_data, test_data


def learn(train_set):
    learning_rate = 0.1
    nr_iter = 1
    all_classified = False
    bias = numpy.random.uniform(0, 1, 1)
    w = numpy.random.uniform(0, 1, 784)

    while not all_classified and nr_iter >= 0:
        all_classified = True
        for x, t in train_set:
            z = numpy.add(numpy.dot(w, x), bias)  # z = w * x + b
            output = 1 if z > 0 else 0  # classify the sample
            w = w + (t - output) * x * learning_rate
            bias = bias + (t - output) * learning_rate

            if output != t:
                all_classified = False

        nr_iter -= 1

    return w, bias


def set_perceptrons(train_set):
    perceptrons = []
    for digit in range(0, 10):
        t_set = [[x, 1] if y == digit else [x, 0] for x, y in deepcopy(train_set)]
        perceptrons += [learn(t_set)]

    return perceptrons


def start():
    print("[i] Loading data...")
    train_set, valid_set, test_set = load_db(r'..\Utils\mnist.pkl.gz')
    n = len(test_set[0])
    if not train_set:
        return

    train_data, test_data = format_data(train_set, test_set)
    perceptrons = set_perceptrons(train_data)

    goods = numpy.zeros((10,), dtype=int)
    total = numpy.zeros((10,), dtype=int)

    k = 1
    print("[i] Started:")
    for x, t in deepcopy(test_data):
        print("\r{} from {}".format(k, n), end='')
        max_value = -float("inf")
        index = 0
        for i in range(0, len(perceptrons)):
            perceptron = perceptrons[i]
            w = perceptron[0]
            bias = perceptron[1]
            z = numpy.add(numpy.dot(w, x), bias)
            if z > max_value:
                max_value = z
                index = i

        if index == t:
            goods[t] += 1
        total[t] += 1
        k += 1

    print('\nAccuracy: {}\n'.format(sum(goods) / float(len(test_set[0])) * 100))


if __name__ == '__main__':
    start()
