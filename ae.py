#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

epsilon = 1e-10
rng = np.random.RandomState(seed=0)


def sigmoid(a):
    return 1. / (1. + np.exp(-a))


def sigmoid_prime(a):
    return sigmoid(a) * (1. - sigmoid(a))


class AE(object):
    def __init__(self, x_size, h_size):
        self.x_size = x_size
        self.h_size = h_size
        self.W = rng.normal(0, 0.1, (self.h_size, self.x_size))
        self.b = np.zeros((self.h_size, 1))
        self.c = np.zeros((self.x_size, 1))

    def train(self, data_train, data_valid, epoch=200, batch_size=10, alpha=0.1, lmbda=0., momentum=0., dropout=0.,
              output=False):
        self.batch_size = batch_size
        self.dropout = dropout
        n = len(data_train)
        train_cross_entropy_errors = np.zeros(epoch)
        valid_cross_entropy_errors = np.zeros(epoch)
        self.update_W = np.zeros(self.W.shape)
        self.update_b = np.zeros(self.b.shape)
        self.update_c = np.zeros(self.c.shape)
        for i in range(epoch):
            rng.shuffle(data_train)
            mini_batches = [data_train[j:j + batch_size] for j in range(0, n, batch_size)]
            for mini_batch in mini_batches:
                self.SGD(mini_batch, alpha, lmbda, momentum)
            train_cross_entropy_errors[i] = self.reconstruction_error(data_train)
            valid_cross_entropy_errors[i] = self.reconstruction_error(data_valid)
            if output or i + 1 == epoch:
                print("Epoch {0} completed".format(i + 1))
            if output:
                print("Training cross-entropy error  : {0:7.4f}".format(train_cross_entropy_errors[i]))
                print("Validation cross-entropy error: {0:7.4f}".format(valid_cross_entropy_errors[i]))
        return [train_cross_entropy_errors, valid_cross_entropy_errors]

    def SGD(self, mini_batch, alpha, lmbda, momentum):
        nabla_W, nabla_b, nabla_c = self.backward_prop(mini_batch)
        self.update_W = momentum * self.update_W - alpha * (nabla_W / self.batch_size + lmbda * self.W)
        self.update_b = momentum * self.update_b - alpha * (nabla_b / self.batch_size)
        self.update_c = momentum * self.update_c - alpha * (nabla_c / self.batch_size)
        self.W += self.update_W
        self.b += self.update_b
        self.c += self.update_c

    def forward_prop(self, x, dropout=0.):
        x_n = x * (rng.rand(*x.shape) < 1 - dropout).astype(x.dtype)
        a_x = np.dot(self.W, x_n) + self.b
        h_x = sigmoid(a_x)
        a_h = np.dot(self.W.T, h_x) + self.c
        x_h = sigmoid(a_h)
        return a_x, h_x, a_h, x_h

    def backward_prop(self, mini_batch):
        x, y = zip(*mini_batch)
        x = np.hstack(x)
        a_x, h_x, a_h, x_h = self.forward_prop(x, self.dropout)
        delta = x_h - x
        nabla_W = np.dot(delta, h_x.T).T
        nabla_c = np.sum(delta, axis=1).reshape(self.c.shape)
        delta = np.dot(self.W.T.T, delta) * sigmoid_prime(a_x)
        nabla_W += np.dot(delta, x.T)
        nabla_b = np.sum(delta, axis=1).reshape(self.b.shape)
        return nabla_W, nabla_b, nabla_c

    def reconstruction_error(self, data):
        x, y = zip(*data)
        x = np.hstack(x)
        a_x, h_x, a_h, x_h = self.forward_prop(x)
        cross_entropy = np.sum(-x * np.log(x_h + epsilon) - (1 - x) * np.log(1 - x_h + epsilon), axis=0)
        return np.mean(cross_entropy, axis=0)
