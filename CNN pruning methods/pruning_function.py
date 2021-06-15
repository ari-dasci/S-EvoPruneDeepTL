# -*- coding: utf-8 -*-

import tensorflow_model_optimization as tfmot
from keras.models import clone_model
from keras.models import Model
import numpy as np

from Logger import Logger as log


def apply_pruning_to_layers(model: type(Model),
                            layers_id: list,
                            pruning_conf: dict = None) -> (type(Model), list):

  str_types = all([isinstance(i, str) for i in layers_id])
  assert_msg =  " All layers to prune must be identified as string (layer name)!"
  assert str_types or layers_id is None, log.DebugError(assert_msg)

  prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
  pruning_conf = pruning_conf if pruning_conf is not None else dict()

  def cloning_function(layer):
      if layer.name in layers_id:
          log.DebugInfo(f"layer {layer.name} selected for pruning")
          return prune_low_magnitude(layer, **pruning_conf)
      else:
          return layer

  model_to_prune = clone_model( model, clone_function=cloning_function)
  log.DebugSuccess("Returned model requires to be recompiled!! Don't forget it ;D")

  callbacks = [tfmot.sparsity.keras.UpdatePruningStep(),
               tfmot.sparsity.keras.PruningSummaries(log_dir='./logs')]

  return (model_to_prune, callbacks)


#! Sacado de: https://github.com/yashkim77/Neural_Network_Pruning_Sparsification
def weight_prune(weight, k):
    '''
    Method to rank the individual weights in weight matrix W
    according to their magnitude (absolute value), and then set to zero the smallest
    k%.

    Arguments:
    weight -- weigths to prune
    k -- pruning percent

    Returns:
    weights  = pruned weigths
    '''
    weights = []
    weight_temp = weight
    weight_temp = np.absolute(weight_temp)
    weight_temp = np.sort(weight_temp, axis = None)
    threshold = weight_temp[int(k*weight_temp.size)] #Finding the threshold weight
    weight[(weight < threshold) & (weight > -threshold)] = 0 #Setting the lowest k% weights to 0
    weights.append(weight)
    return weights[0]


def unit_prune(weight, k, first=True):
    '''
    Method to rank the columns of a weight matrix according
    to their L2-norm and delete the smallest k%.
    
    Arguments:
    weight -- weigths to prune
    k -- pruning percent 
    
    Returns:
    weights  = pruned weigths
    '''
    axis = 0 if first else 1
    weights = []
    weight_temp = np.linalg.norm(weight, axis=axis) #Finding norm of each column
    sorted = np.sort(weight_temp)
    threshold = sorted[int(k*sorted.size)] #Finding threshold norm
    j=0
    for i in weight_temp:
        if(i<threshold):
            if first:
                weight[:, j] = 0 #set jth column in weight matrix to 0 because it's norm is less than that of threshold
            else:
                weight[j, :] = 0 #set jth row in weight matrix to 0 because it's norm is less than that of threshold
        j=j+1
    weights.append(weight)
    return weights[0]
