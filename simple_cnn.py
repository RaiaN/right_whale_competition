#!/usr/bin/env python
from fileinput import filename
__author__ = 'Alexandra Vesloguzova, Peter Leontiev, Sergey Krivohatskiy'

import lasagne
import numpy as np
import os
import time
from lasagne import layers
from nolearn.lasagne import NeuralNet


class CNN:
    def __init__(self, features_cnt, num_targets, 
                 num_epochs=10,
                 fresh_start=False,
                 dump_dir="network_weights/",
                 filename_to_dump="net.w"): 
        self.network = None
        self.features_cnt = features_cnt
        self.num_targets = num_targets
        
        self.num_epochs = num_epochs
        self.fresh_start = fresh_start
        
        self.dump_dir = dump_dir
        self.filename_to_dump = filename_to_dump
        
        try:
            os.makedirs(self.dump_dir)
        except:
            print("%s already exists" % self.dump_dir)
    
    
    def dump_network_weights(self, network, training_history):
        filename = time.strftime("%Y%m%d-%H%M%S.") + self.filename_to_dump
        network.save_params_to(os.path.join(self.dump_dir, filename))
        
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

            max_epochs=self.num_epochs,
            verbose=1,

            use_label_encoder=True,
            on_epoch_finished=[self.dump_network_weights],
        )
        
    def _find_latest_dump(self):
        entries = sorted((fn.split(".")[0] for fn in os.listdir(self.dump_dir)), reverse=True)
        latest_dump_filename = os.path.join(self.dump_dir, entries[0] + ".net.w")
        print("Latest dump: %s\n" % latest_dump_filename)
        return latest_dump_filename
    
    def _try_load_dump(self):
        try:
            self.network.load_params_from(self._find_latest_dump())
            print("Successfully loaded network params")
        except:
            print("Cannot load network params due to errors")
            print("Initializing clean network instead...")
            self.network = self._build()
        

    def build(self):
        self.network = self._build()
        if not self.fresh_start:
            self._try_load_dump()
        

    def fit(self, x_train, y_train):
        self.build()

        x_train = x_train[:, np.newaxis]
        self.network.fit(x_train, y_train)

    def predict_proba(self, x_test):
        x_test = x_test[:, np.newaxis]
        return self.network.predict_proba(x_test)