#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

epsilon = 1e-10
rng = np.random.RandomState(seed=0)


def sigmoid(a):
    return 1. / (1. + np.exp(-a))


def sigmoid_prime(a):
    return sigmoid(a) * (1. - sigmoid(a))


def tanh(a):
    return np.tanh(a)


def tanh_prime(a):
    return 1. - tanh(a) ** 2.


def ReLU(a):
    return np.maximum(a, np.zeros(a.shape))


def ReLU_prime(a):
    return (a > 0.).astype(a.dtype)


def softmax(a):
    return np.exp(a - np.max(a, axis=0)) / np.sum(np.exp(a - np.max(a, axis=0)), axis=0)


class NN(object):
    def __init__(self, sizes, g, g_prime):
        self.sizes = sizes
        self.g = g
        self.g_prime = g_prime
        self.Ws = [rng.uniform(-np.sqrt(6 / (x + y)), np.sqrt(6 / (x + y)), (y, x)) for x, y in
                   zip(self.sizes[:-1], self.sizes[1:])]
        self.bs = [np.zeros((y, 1)) for y in self.sizes[1:]]

    def train(self, data_train, data_valid, epoch=200, batch_size=10, alpha=0.1, lmbda=0., momentum=0., output=False):
        self.batch_size = batch_size
        n = len(data_train)
        train_cross_entropy_errors = np.zeros(epoch)
        valid_cross_entropy_errors = np.zeros(epoch)
        train_classification_errors = np.zeros(epoch)
        valid_classification_errors = np.zeros(epoch)
        self.update_W = [np.zeros(w.shape) for w in self.Ws]
        self.update_b = [np.zeros(b.shape) for b in self.bs]
        for i in range(epoch):
            rng.shuffle(data_train)
            mini_batches = [data_train[j:j + batch_size] for j in range(0, n, batch_size)]
            for mini_batch in mini_batches:
                self.SGD(mini_batch, alpha, lmbda, momentum)
            train_cross_entropy_errors[i], train_classification_errors[i] = self.predict(data_train, output=False)
            valid_cross_entropy_errors[i], valid_classification_errors[i] = self.predict(data_valid, output=False)
            if output or i + 1 == epoch:
                print("Epoch {0} completed".format(i + 1))
            if output:
                print("Training cross-entropy error  : {0:6.4f}  Training classification error  : {1:5.2f}%".format(
                    train_cross_entropy_errors[i], train_classification_errors[i]))
                print("Validation cross-entropy error: {0:6.4f}  Validation classification error: {1:5.2f}%".format(
                    valid_cross_entropy_errors[i], valid_classification_errors[i]))
        return [train_cross_entropy_errors, valid_cross_entropy_errors], [train_classification_errors,
                                                                          valid_classification_errors]

    def SGD(self, mini_batch, alpha, lmbda, momentum):
        nabla_W, nabla_b = self.backward_prop(mini_batch)
        self.update_W = [momentum * puW - alpha * (nW / self.batch_size + lmbda * W) for puW, nW, W in
                         zip(self.update_W, nabla_W, self.Ws)]
        self.update_b = [momentum * pub - alpha * (nb / self.batch_size) for pub, nb in zip(self.update_b, nabla_b)]
        self.Ws = [W + uW for W, uW in zip(self.Ws, self.update_W)]
        self.bs = [b + ub for b, ub in zip(self.bs, self.update_b)]

    def forward_prop(self, x, predict=False):
        h_x = x
        h_xs = [x]
        a_xs = []
        for l in range(len(self.sizes), 1, -1):
            a_x = np.dot(self.Ws[-l + 1], h_x) + self.bs[-l + 1]
            a_xs.append(a_x)
            if -l + 1 < -1:
                h_x = self.g(a_x)
            else:
                h_x = softmax(a_x)
            h_xs.append(h_x)
        if predict:
            return h_x
        else:
            return h_xs, a_xs

    def backward_prop(self, mini_batch):
        x, y = zip(*mini_batch)
        x = np.hstack(x)
        y = np.hstack(y)
        h_xs, a_xs = self.forward_prop(x)
        nabla_W = [np.zeros(w.shape) for w in self.Ws]
        nabla_b = [np.zeros(b.shape) for b in self.bs]
        delta = h_xs[-1] - y
        nabla_W[-1] = np.dot(delta, h_xs[-2].T)
        nabla_b[-1] = np.sum(delta, axis=1).reshape(self.bs[-1].shape)
        for l in range(2, len(self.sizes)):
            delta = np.dot(self.Ws[-l + 1].T, delta) * self.g_prime(a_xs[-l])
            nabla_W[-l] = np.dot(delta, h_xs[-l - 1].T)
            nabla_b[-l] = np.sum(delta, axis=1).reshape(self.bs[-l].shape)
        return nabla_W, nabla_b

    def predict(self, data, output=False):
        cross_entropy_error = self.cross_entropy_error(data)
        classification_error = self.classification_error(data)
        if output:
            print("Test cross-entropy error  : {0:6.4f}  Test classification error  : {1:5.2f}%".format(
                cross_entropy_error, classification_error))
        return cross_entropy_error, classification_error

    def cross_entropy_error(self, data):
        x, y = zip(*data)
        x = np.hstack(x)
        y = np.hstack(y)
        f_x = self.forward_prop(x, predict=True)
        cross_entropy = np.sum(-y * np.log(f_x + epsilon) - (1 - y) * np.log(1 - f_x + epsilon), axis=0)
        return np.sum(cross_entropy, axis=0) / len(data)

    def classification_error(self, data):
        x, y = zip(*data)
        x = np.hstack(x)
        y = np.hstack(y)
        f_x = np.argmax(self.forward_prop(x, predict=True), axis=0)
        y = np.argmax(y, axis=0)
        return sum(int(x != y) for (x, y) in zip(f_x, y)) / len(data) * 100
