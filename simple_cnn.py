#!/usr/bin/env python
from fileinput import filename
__author__ = 'Alexandra Vesloguzova, Peter Leontiev, Sergey Krivohatskiy'

import lasagne
import numpy as np
import os
from lasagne import layers
from nolearn.lasagne import NeuralNet


class CNN:
    def __init__(self, features_cnt, num_targets, filename_to_dump="net.w"):
        self.network = None
        self.features_cnt = features_cnt
        self.num_targets = num_targets
        self.filename_to_dump = filename_to_dump
    
    @staticmethod
    def dump_network_weights(network, training_history, cnn=None):
        network.save_params_to(cnn.filename_to_dump)
        
    def _build(self):
        return NeuralNet\
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
            on_epoch_finished=CNN.dump_network_weights(cnn=self),
        )
        

    def build(self, num_epochs=10):
        self.network = _build()
        
        if os.path.exists(self.filename_to_dump):
            try:
                self.network.load_params_from(self.filename_to_dump)
            except:
                print("Cannot load network params due to errors")
                print("Initializing clean network...")
                self.network = _build()
        

    def fit(self, x_train, y_train):
        self.build()

        x_train = x_train[:, np.newaxis]
        self.network.fit(x_train, y_train)

    def predict_proba(self, x_test):
        x_test = x_test[:, np.newaxis]
        return self.network.predict_proba(x_test)