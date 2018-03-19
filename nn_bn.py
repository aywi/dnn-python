#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


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
        self.Ws = [np.random.uniform(-np.sqrt(6 / (x + y)), np.sqrt(6 / (x + y)), (y, x)) for x, y in
                   zip(self.sizes[:-1], self.sizes[1:])]
        self.bs = [np.zeros((y, 1)) for y in self.sizes[1:]]
        self.gammas = [np.ones((y, 1)) for y in self.sizes[1:]]
        self.betas = [np.zeros((y, 1)) for y in self.sizes[1:]]
        self.temp_mus = [np.zeros((y, 1)) for y in self.sizes[1:]]
        self.temp_sigma2s = [np.ones((y, 1)) for y in self.sizes[1:]]
        self.epsilon = 0.0000001

    def train(self, data_train, data_valid, epoch=200, batch_size=10, alpha=0.1, lmbda=0., momentum=0.,
              batch_norm=False, output=False):
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        n = len(data_train)
        train_cross_entropy_errors, train_classification_errors = [], []
        valid_cross_entropy_errors, valid_classification_errors = [], []
        update_W = [np.zeros(w.shape) for w in self.Ws]
        update_b = [np.zeros(b.shape) for b in self.bs]
        update_gamma = [np.zeros(b.shape) for b in self.gammas]
        update_beta = [np.zeros(b.shape) for b in self.betas]
        for j in range(epoch):
            np.random.shuffle(data_train)
            mini_batches = [data_train[k:k + batch_size] for k in range(0, n, batch_size)]
            for mini_batch in mini_batches:
                update_W, update_b, update_gamma, update_beta = self.SGD(mini_batch, alpha, lmbda, momentum, update_W,
                                                                         update_b, update_gamma, update_beta)
            if self.batch_norm and self.batch_size > 1:
                for mini_batch in mini_batches:
                    self.running_mus = [np.zeros((y, 0)) for y in self.sizes[1:]]
                    self.running_sigma2s = [np.zeros((y, 0)) for y in self.sizes[1:]]
                    x, y = zip(*mini_batch)
                    x = np.hstack(x)
                    self.forward_prop(x, running=True)
            train_cross_entropy_error, train_classification_error = self.predict(data_train, output=False)
            train_cross_entropy_errors.append(train_cross_entropy_error)
            train_classification_errors.append(train_classification_error)
            valid_cross_entropy_error, valid_classification_error = self.predict(data_valid, output=False)
            valid_cross_entropy_errors.append(valid_cross_entropy_error)
            valid_classification_errors.append(valid_classification_error)
            if output or j + 1 == epoch:
                print("Epoch %d completed" % (j + 1))
            if output:
                print("Training cross-entropy error  : %6.4f  Training classification error  : %5.2f%%" % (
                    train_cross_entropy_error, train_classification_error * 100))
                print("Validation cross-entropy error: %6.4f  Validation classification error: %5.2f%%" % (
                    valid_cross_entropy_error, valid_classification_error * 100))
        return list(zip(train_cross_entropy_errors, train_classification_errors, valid_cross_entropy_errors,
                        valid_classification_errors))

    def SGD(self, mini_batch, alpha, lmbda, momentum, prev_update_W, prev_update_b, prev_update_gamma,
            prev_update_beta):
        nabla_W, nabla_b, nabla_gamma, nabla_beta = self.backward_prop(mini_batch)
        update_W = [lmbda * W + 1 / self.batch_size * nW + momentum * puW for W, nW, puW in
                    zip(self.Ws, nabla_W, prev_update_W)]
        self.Ws = [W - alpha * uW for W, uW in zip(self.Ws, update_W)]
        update_b = [1 / self.batch_size * nb + momentum * pub for nb, pub in zip(nabla_b, prev_update_b)]
        self.bs = [b - alpha * ub for b, ub in zip(self.bs, update_b)]
        update_gamma = [1 / self.batch_size * ngamma + momentum * pugamma for ngamma, pugamma in
                        zip(nabla_gamma, prev_update_gamma)]
        self.gammas = [gamma - alpha * ugamma for gamma, ugamma in zip(self.gammas, update_gamma)]
        update_beta = [1 / self.batch_size * nbeta + momentum * pubeta for nbeta, pubeta in
                       zip(nabla_beta, prev_update_beta)]
        self.betas = [beta - alpha * ubeta for beta, ubeta in zip(self.betas, update_beta)]
        return update_W, update_b, update_gamma, update_beta

    def forward_prop(self, x, running=False, predict=False):
        h_x = x
        h_xs = [x]
        a_xs = []
        BN_xs = []
        for l in range(len(self.sizes), 1, -1):
            a_x = np.dot(self.Ws[-l + 1], h_x) + self.bs[-l + 1]
            a_xs.append(a_x)
            if self.batch_norm and self.batch_size > 1:
                BN_x = self.BN(a_x, l, running=running, predict=predict)
            else:
                BN_x = a_x
            BN_xs.append(BN_x)
            if -l + 1 < -1:
                h_x = self.g(BN_x)
            else:
                h_x = softmax(BN_x)
            h_xs.append(h_x)
        if predict:
            return h_x
        else:
            return h_xs, a_xs, BN_xs

    def BN(self, a_x, l, running=False, predict=False):
        if not predict:
            mu = np.mean(a_x, axis=1).reshape(a_x.shape[0], 1)
            sigma2 = np.mean((a_x - mu) ** 2, axis=1).reshape(a_x.shape[0], 1)
            if running:
                self.running_mus[-l + 1] = np.hstack((self.running_mus[-l + 1], mu))
                self.running_sigma2s[-l + 1] = np.hstack((self.running_sigma2s[-l + 1], sigma2))
            else:
                self.temp_mus[-l + 1] = mu
                self.temp_sigma2s[-l + 1] = sigma2
            BN_x_hat = (a_x - mu) / np.sqrt(sigma2 + self.epsilon)
            BN_x = self.gammas[-l + 1] * BN_x_hat + self.betas[-l + 1]
        else:
            mu = np.mean(self.running_mus[-l + 1], axis=1).reshape(a_x.shape[0], 1)
            sigma2 = self.batch_size / (self.batch_size - 1) * np.mean(self.running_sigma2s[-l + 1], axis=1).reshape(
                a_x.shape[0], 1)
            BN_x = self.gammas[-l + 1] / np.sqrt(sigma2 + self.epsilon) * a_x + (
                    self.betas[-l + 1] - self.gammas[-l + 1] * mu / np.sqrt(sigma2 + self.epsilon))
        return BN_x

    def backward_prop(self, mini_batch):
        x, y = zip(*mini_batch)
        x = np.hstack(x)
        y = np.hstack(y)
        h_xs, a_xs, BN_xs = self.forward_prop(x)
        nabla_W = [np.zeros(w.shape) for w in self.Ws]
        nabla_b = [np.zeros(b.shape) for b in self.bs]
        nabla_gamma = [np.zeros(gamma.shape) for gamma in self.gammas]
        nabla_beta = [np.zeros(beta.shape) for beta in self.betas]
        delta = h_xs[-1] - y
        if self.batch_norm and self.batch_size > 1:
            delta, nabla_gamma, nabla_beta = self.BN_backward(a_xs, 1, delta, nabla_gamma, nabla_beta)
        nabla_W[-1] = np.dot(delta, h_xs[-2].T)
        nabla_b[-1] = np.sum(delta, axis=1).reshape(delta.shape[0], 1)
        for l in range(2, len(self.sizes)):
            delta = np.dot(self.Ws[-l + 1].T, delta) * self.g_prime(BN_xs[-l])
            if self.batch_norm and self.batch_size > 1:
                delta, nabla_gamma, nabla_beta = self.BN_backward(a_xs, l, delta, nabla_gamma, nabla_beta)
            nabla_W[-l] = np.dot(delta, h_xs[-l - 1].T)
            nabla_b[-l] = np.sum(delta, axis=1).reshape(delta.shape[0], 1)
        return nabla_W, nabla_b, nabla_gamma, nabla_beta

    def BN_backward(self, a_xs, l, delta, nabla_gamma, nabla_beta):
        mu = self.temp_mus[-l]
        sigma2 = self.temp_sigma2s[-l]
        nabla_gamma[-l] = np.sum(delta * (a_xs[-l] - mu) / np.sqrt(sigma2 + self.epsilon), axis=1).reshape(
            delta.shape[0], 1)
        nabla_beta[-l] = np.sum(delta, axis=1).reshape(delta.shape[0], 1)
        delta = (1. / self.batch_size) * self.gammas[-l] / np.sqrt(sigma2 + self.epsilon) * (
                self.batch_size * delta - np.sum(delta, axis=1).reshape(delta.shape[0], 1) - (a_xs[-l] - mu) / (
                sigma2 + self.epsilon) * np.sum(delta * (a_xs[-l] - mu), axis=1).reshape(delta.shape[0], 1))
        return delta, nabla_gamma, nabla_beta

    def predict(self, data, output=False):
        cross_entropy_error = self.cross_entropy_error(data)
        classification_error = self.classification_error(data)
        if output:
            print("Test cross-entropy error: %6.4f  Test classification error: %5.2f%%" % (
                cross_entropy_error, classification_error * 100))
        return cross_entropy_error, classification_error

    def cross_entropy_error(self, data):
        x, y = zip(*data)
        x = np.hstack(x)
        y = np.hstack(y)
        f_x = self.forward_prop(x, predict=True)
        cross_entropy = np.sum(np.nan_to_num(-y * np.log(f_x) - (1 - y) * np.log(1 - f_x)), axis=0)
        return np.sum(cross_entropy, axis=0) / len(data)

    def classification_error(self, data):
        x, y = zip(*data)
        x = np.hstack(x)
        y = np.hstack(y)
        f_x = np.argmax(self.forward_prop(x, predict=True), axis=0)
        y = np.argmax(y, axis=0)
        return sum(int(x != y) for (x, y) in zip(f_x, y)) / len(data)
