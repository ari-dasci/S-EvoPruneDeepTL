# -*- coding: utf-8 -*-

from tensorflow_model_optimization.sparsity.keras import PolynomialDecay, strip_pruning
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np

import pruning_function as pf
from pruning_function import apply_pruning_to_layers as prunF
from create_model import create_pruneable_model
from LoadDataSet import LoadData
from Logger import Logger as log


class EvoPruningModel(object):
    def __init__(self, train_generator, test_generator, info,
                 pruneFC1=True, pruneFC2=True, final_sparsity=0.9,
                 batch_size=32, epochs=600):
        self.batch_size = batch_size
        self.epochs = epochs
        self.traingen = train_generator
        self.testgen = test_generator
        self.input_shape = info['input_shape']
        self.info = info
        self.callbacks = [EarlyStopping(monitor='loss', min_delta=0.00001, patience=10, verbose=1, mode="min", restore_best_weights=True)]
        self.prune_layers = []
        if pruneFC1:
            self.prune_layers.append('FC1')
        if pruneFC2:
            self.prune_layers.append('FC2')
        self.model_to_prune = create_pruneable_model(self.input_shape, info['num_classes'])
        num_batches = self.info['len_train']//self.batch_size
        if len(self.prune_layers) == 1:
            polynomial = PolynomialDecay(initial_sparsity=0.1,
                                         final_sparsity=final_sparsity,
                                         begin_step=0,
                                         end_step=num_batches * 5 * 5,
                                         frequency=num_batches * 5,
                                         power=3)
            self.pruning_conf = {'pruning_schedule': polynomial}
            model_to_prune, callbacks = prunF(self.model_to_prune, self.prune_layers, self.pruning_conf)

        elif len(self.prune_layers) == 2:
            polynomial_fc1 = PolynomialDecay(initial_sparsity=0.1,
                                             final_sparsity=final_sparsity[0],
                                             begin_step=0,
                                             end_step=num_batches * 5 * 5,
                                             frequency=num_batches * 5,
                                             power=3)
            polynomial_fc2 = PolynomialDecay(initial_sparsity=0.1,
                                             final_sparsity=final_sparsity[1],
                                             begin_step=0,
                                             end_step=num_batches * 5 * 5,
                                             frequency=num_batches * 5,
                                             power=3)
            self.pruning_conf_fc1 = {'pruning_schedule': polynomial_fc1}
            self.pruning_conf_fc2 = {'pruning_schedule': polynomial_fc2}
            model_to_prune, callbacks = prunF(self.model_to_prune, [self.prune_layers[0]], self.pruning_conf_fc1)
            model_to_prune, callbacks = prunF(model_to_prune, [self.prune_layers[1]], self.pruning_conf_fc2)

        self.callbacks += callbacks
        self.model = model_to_prune
        self._compile()
        self.raw_model = tf.keras.models.clone_model(self.model_to_prune)
        self.raw_model.set_weights(self.model_to_prune.get_weights())
        self.raw_model_weights = self.raw_model.get_weights()
        opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        self.raw_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        log.DebugSuccess('Raw model compiled')

    def _compile(self):
        opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        log.DebugSuccess('Model compiled')

    def set_original_weights(self):
        # Nos actualiza los pesos de la red raw con los pesos
        # despues de haberla entrenado
        self.raw_model.set_weights(self.raw_model_weights)
        log.DebugSuccess("Raw model weights updated")

    def train(self, epochs=26):
        log.DebugWarning('Starting network training...')
        dataset = self.traingen.next()
        res = self.model.fit(dataset[0], dataset[1], batch_size = self.batch_size,
                             epochs = epochs, callbacks = self.callbacks[1:])
        loss = res.history['loss'][-1]
        acc = res.history['accuracy'][-1]
        log.DebugInfo(f"Model train loss = {loss:.3f} and accuracy = {acc:.3f}")
        return {'loss': res.history['loss'][-1], 'accuracy': res.history['accuracy'][-1]}

    def train_without_pruning(self):
        log.DebugWarning('Starting network training...')
        dataset = self.traingen.next()
        res = self.raw_model.fit(dataset[0], dataset[1], batch_size = self.batch_size,
                                 epochs = self.epochs, callbacks = self.callbacks)
        loss = res.history['loss'][-1]
        acc = res.history['accuracy'][-1]
        log.DebugInfo(f"Model train loss = {loss:.3f} and accuracy = {acc:.3f}")
        self.raw_model_weights = self.raw_model.get_weights()
        return {'loss': res.history['loss'][-1], 'accuracy': res.history['accuracy'][-1]}

    def weight_prune(self, k):
        pruned_perc = dict()

        if type(k) == list:
            k = iter(k)
        else:
            k = iter([k]*len(self.prune_layers))

        for layer in self.raw_model.layers:
            if layer.name in self.prune_layers:
                w = layer.get_weights()
                new_w = np.array(pf.weight_prune(w[0], next(k)))
                layer.set_weights([new_w, w[1]])
                weights = layer.get_weights()
                a = np.product(weights[0].shape)
                b = len(weights[0][weights[0] == 0.0]) + len(weights[1][weights[1] == 0.0])
                pruned_perc[layer.name] = b/a*100
                log.DebugInfo(f"{b/a*100:.2f}% of the weights on layer {layer.name} were pruned")
        return pruned_perc

    def unit_prune(self, k, first=True):
        pruned_perc = dict()
        
        if type(k) == list:
            k = iter(k)
        else:
            k = iter([k]*len(self.prune_layers))
        
        for layer in self.raw_model.layers:
            if layer.name in self.prune_layers:
                w = layer.get_weights()
                new_w = np.array(pf.unit_prune(w[0], next(k), first))
                layer.set_weights([new_w, w[1]])
                weights = layer.get_weights()
                a = np.product(weights[0].shape)
                b = len(weights[0][weights[0] == 0.0]) + len(weights[1][weights[1] == 0.0])
                log.DebugInfo(f"{b/a*100:.2f}% of the weights on layer {layer.name} were pruned")
                pruned_perc[layer.name] = b/a*100
        return pruned_perc

    def test(self):
        dataset = self.testgen.next()
        model = strip_pruning(self.model_to_prune)
        opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        res = model.evaluate(dataset[0], dataset[1])
        log.DebugInfo(f"Model test loss = {res[0]:.3f} and accuracy = {res[1]:.3f}")
        return {'loss': res[0] , 'accuracy': res[1]}

    def test_raw_model(self):
        dataset = self.testgen.next()
        res = self.raw_model.evaluate(dataset[0], dataset[1])
        log.DebugInfo(f"Model test loss = {res[0]:.3f} and accuracy = {res[1]:.3f}")
        return {'loss': res[0] , 'accuracy': res[1]}

    def update_info(self, train_generator, test_generator, info):
        self.traingen = train_generator
        self.testgen = test_generator
        self.info = info

    def decay_prun_perc(self):
        output = {}
        for x, layer in enumerate(self.model.layers):
            if layer.name.split('_')[0] == "prune":
                weights = layer.get_weights()
                a = np.product(weights[0].shape)
                b = len(weights[0][weights[0] == 0.0]) + len(weights[1][weights[1] == 0.0])
                output[layer.name.split('_')[-1]] = b/a*100
                log.DebugInfo(f"{b/a*100:.2f}% of the weights on layer {layer.name} (layer index = {x}) were pruned")
        return output

    def __str__(self):
        """ Print number of pruned neurons on each pruneable layer """
        log.DebugWarning("This is an approximation making the asumption that pruned weights are equal to 0.0")
        return str(self.decay_prun_perc())

    def save(self, path='./summary/model.h5'):
        """ save the original model structure pruned"""
        if path.split('.')[-1] != 'h5':
            path += '.h5'
        log.DebugInfo(f"Savepath \"{path}\"")
        model_for_export = strip_pruning(self.model)
        tf.keras.models.save_model(strip_pruning(self.model), path, include_optimizer = False)
        log.DebugSuccess(f"Model saved to {path}")

    def save_raw_model(self, path='./summary/model.h5'):
        if path.split('.')[-1] != 'h5':
            path += '.h5'
        log.DebugInfo(f"Savepath \"{path}\"")
        tf.keras.models.save_model(self.raw_model, path, include_optimizer = False)
        log.DebugSuccess(f"Raw model saved to {path}")
