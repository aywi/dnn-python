#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

epsilon = 1e-10
rng = np.random.RandomState(seed=0)


def sigmoid(a):
    return 1. / (1. + np.exp(-a))


class RBM(object):
    def __init__(self, x_size, h_size):
        self.x_size = x_size
        self.h_size = h_size
        self.W = rng.normal(0, 0.1, (self.h_size, self.x_size))
        self.b = np.zeros((self.h_size, 1))
        self.c = np.zeros((self.x_size, 1))

    def train(self, data_train, data_valid, epoch=200, batch_size=10, alpha=0.1, lmbda=0., momentum=0., k=1,
              output=False):
        self.batch_size = batch_size
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
                self.SGD(mini_batch, alpha, lmbda, momentum, k)
            train_cross_entropy_errors[i] = self.reconstruction_error(data_train)
            valid_cross_entropy_errors[i] = self.reconstruction_error(data_valid)
            if output or i + 1 == epoch:
                print("Epoch {0} completed".format(i + 1))
            if output:
                print("Training cross-entropy error  : {0:7.4f}".format(train_cross_entropy_errors[i]))
                print("Validation cross-entropy error: {0:7.4f}".format(valid_cross_entropy_errors[i]))
        return [train_cross_entropy_errors, valid_cross_entropy_errors]

    def SGD(self, mini_batch, alpha, lmbda, momentum, k=1):
        x, y = zip(*mini_batch)
        x = np.hstack(x)
        h_x, h_x_neg, x_neg = self.CD_k(x, k)
        nabla_W = np.dot(h_x_neg, x_neg.T) - np.dot(h_x, x.T)
        nabla_b = np.sum(h_x_neg - h_x, axis=1).reshape(self.b.shape)
        nabla_c = np.sum(x_neg - x, axis=1).reshape(self.c.shape)
        self.update_W = momentum * self.update_W - alpha * (nabla_W / self.batch_size + lmbda * self.W)
        self.update_b = momentum * self.update_b - alpha * (nabla_b / self.batch_size)
        self.update_c = momentum * self.update_c - alpha * (nabla_c / self.batch_size)
        self.W += self.update_W
        self.b += self.update_b
        self.c += self.update_c

    def CD_k(self, x, k=1):
        h_x = sigmoid(np.dot(self.W, x) + self.b)
        h_x_neg = h_x
        for i in range(k):
            h_x_sample = (rng.rand(*h_x.shape) < h_x_neg).astype(h_x_neg.dtype)
            x_neg = sigmoid(np.dot(self.W.T, h_x_sample) + self.c)
            x_sample = (rng.rand(*x_neg.shape) < x_neg).astype(x_neg.dtype)
            h_x_neg = sigmoid(np.dot(self.W, x_sample) + self.b)
        return h_x, h_x_neg, x_neg

    def reconstruction_error(self, data):
        x, y = zip(*data)
        x = np.hstack(x)
        h_x = sigmoid(np.dot(self.W, x) + self.b)
        x_h = sigmoid(np.dot(self.W.T, h_x) + self.c)
        cross_entropy = np.sum(-x * np.log(x_h + epsilon) - (1 - x) * np.log(1 - x_h + epsilon), axis=0)
        return np.mean(cross_entropy, axis=0)
