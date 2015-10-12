#!/usr/bin/env python
__author__ = 'Alexandra Vesloguzova, Peter Leontiev, Sergey Krivohatskiy'

import lasagne
import numpy as np
from lasagne import layers
from nolearn.lasagne import NeuralNet


class CNN:
    def __init__(self, features_cnt, num_targets):
        self.network = None
        self.features_cnt = features_cnt
        self.num_targets = num_targets

    def build(self, num_epochs=10):
        self.network = NeuralNet\
        (
            layers=[('input', layers.InputLayer),
                    ('conv', layers.Conv2DLayer),
                    ('output', layers.DenseLayer),
                    ],
            # layer parameters:
            input_shape=(None, 1, self.features_cnt, self.features_cnt),
            conv_num_filters=32,  # number of units in 'hidden' layer
            conv_filter_size=(16,16),
            output_nonlinearity=lasagne.nonlinearities.softmax,
            output_num_units=self.num_targets,  # target values for whales

            # optimization method:
            update=lasagne.updates.nesterov_momentum,
            update_learning_rate=0.01,
            update_momentum=0.9,

            max_epochs=num_epochs,
            verbose=1,

            use_label_encoder=True,
        )

    def fit(self, x_train, y_train):
        self.build()

        x_train = x_train[:, np.newaxis]
        self.network.fit(x_train, y_train)

    def predict_proba(self, x_test):
        x_test = x_test[:, np.newaxis]
        return self.network.predict_proba(x_test)