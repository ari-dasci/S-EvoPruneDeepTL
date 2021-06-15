#!/usr/bin/python3
from __future__ import division

import os
import numpy as np
from numpy.random import seed
#seed(1)
from tensorflow import set_random_seed
import keras
#set_random_seed(2)
#from keras.optimizers import SGD
#from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
#from keras.preprocessing.image import ImageDataGenerator as IDG
#from keras.models import Sequential,Model,model_from_json, load_model
#from keras.utils import np_utils
from keras import backend as K
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras import callbacks
from build_model import build_model_full, decode_chromosome, build_model, build_reference_model
from importlib import reload
from enum import Enum
import time
import random
import sys

def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

set_keras_backend("tensorflow")
keras.backend.set_image_data_format("channels_first")
keras.backend.set_image_dim_ordering('tf')
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
session = tf.Session(config=config)
K.set_session(session)


def getFeatures(model, images, batch_size):
    return model.predict(images, batch_size=batch_size)


class Optimization_type(Enum):
    "Optimization type of Network: By neurons or by connections"
    NEURONS = 1
    CONNECTIONS = 2


class Fitness():
    def __init__(self, size_input,first_layer, second_layer, data, bias=False,optimType=Optimization_type.NEURONS):
        self.size_input = size_input
        self.first_layer = first_layer
        self.second_layer = second_layer
        self.data = data
        self.optimType = optimType
        self.bias = bias

    def fitness(self, solution):
        if self.optimType == Optimization_type.NEURONS:
            assert len(solution[int(len(solution)/2):]) == self.first_layer and len(solution[:int(len(solution)/2)]) == self.second_layer
            #solution = np.repeat(solution, )#.tolist()
            #print("Solution repetida")
            #print(solution)
            #print(len(solution))
            print("Num unos: ", str(list(solution).count(1)))
        else:
            print("Para conexiones")
            assert len(solution[int(len(solution)/2):]) == self.size_input*self.first_layer and len(solution[:int(len(solution)/2)]) == self.first_layer * self.second_layer

        return self._fitness(sol=solution)


    def fitness_dense(self, drop=False):
        return self._fitness(sol=None, drop=drop)


    def _fitness(self, sol=None, drop=False):
        #fijamos las semillas al comienzo de la función fitness
        seed(1)
        set_random_seed(2)

        accs = 0
        accs_train = 0

        num_folds = len(self.data[1])
        print(num_folds)

        num_classes = self.data[5]

        #if num_folds == 1:
        #    num_classes = len(self.data[1].class_indices)
        #else:
        #    num_classes = len(self.data[1][0].class_indices)
        print("NUm classes: " + str(num_classes))

        for i in range(num_folds):
            print("Set "+ str(i), flush = True)
            history = []
            base_model = ResNet50(include_top=False, weights="imagenet", pooling="avg")

            if num_folds != 1:
                x,y = self.data[1][i].next()
                x_ = getFeatures(base_model, x,32)

                xVal,yVal = self.data[2][i].next()
                xVal_= getFeatures(base_model, xVal, 32)
            else:
                x,y = self.data[1].next()
                x_ = getFeatures(base_model, x, 32)

                xVal,yVal = self.data[2].next()
                xVal_= getFeatures(base_model, xVal,32)

            del x
            del xVal

            print("Train ", x_.shape, y.shape)
            print("Test ", xVal_.shape, yVal.shape)


            if sol is None:
                model = build_reference_model(x_.shape[1:], self.second_layer,
                                              num_classes, drop,
                                              self.first_layer,self.bias)
            else:
                first_sparse = sol[:int(len(sol)/2)]
                second_sparse = sol[int(len(sol)/2):]

                print(len(first_sparse))
                print(len(second_sparse))

                first_sparse = np.repeat(first_sparse,self.size_input)
                second_sparse = np.repeat(second_sparse,self.first_layer)

                first_matrix_connections = decode_chromosome(first_sparse, self.size_input)
                first_matrix_connections = first_matrix_connections.transpose()

                second_matrix_connections = decode_chromosome(second_sparse, self.first_layer)
                second_matrix_connections = second_matrix_connections.transpose()                

                print("Configuracion")
                print(first_matrix_connections.shape, flush = True)
                print(second_matrix_connections.shape, flush = True)
                print(first_matrix_connections, flush = True)
                print(second_matrix_connections, flush = True)
                model = build_model_full(first_matrix_connections,second_matrix_connections ,x_.shape[1:], num_classes,self.bias)

            callbacks_array = [callbacks.EarlyStopping(monitor='loss', min_delta=0.00001, patience=10, verbose=1, mode="min", restore_best_weights=True)]
            history = model.fit(x_, y, epochs=600, batch_size=32, verbose=0, callbacks=callbacks_array)
            
            print("Fit "+str(set)+
            " acc:" + str(history.history['acc'][-1])[:6] +
            " loss:"+ str(history.history['loss'][-1])[:6],flush = True
            )

            [test_loss, acc_test] = model.evaluate(xVal_,yVal)
            print(test_loss, acc_test)

            accs += acc_test
            accs_train += history.history['acc'][-1]
            K.clear_session()

        print("Accuracy cruzada: "+str(accs/num_folds), flush = True)
        acc = accs/num_folds
        print("Accuracy TRAIN: " + str(accs_train/num_folds), flush = True)

        # se fija semilla a un valor aleatorio
        new_seed = random.randrange(2**32-1)
        print("Nueva seed " + str(new_seed)) 
        seed(new_seed)
        return acc
    
    def get_problem_name(self):
        return self.data[0]
