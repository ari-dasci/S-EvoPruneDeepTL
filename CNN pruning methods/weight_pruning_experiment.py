from time import time

from Model import EvoPruningModel
from Logger import Logger as log


def get_pruned_perc(model, prune_type, schedule, percentage):
    if prune_type == 'WEIGHT':
        pruned_perc = model.weight_prune(percentage)
        return pruned_perc
    elif prune_type == 'NEURON':
        if schedule == 'FIRST':
            pruned_perc = model.unit_prune(percentage)
            return pruned_perc
        else:
            pruned_perc = model.unit_prune(percentage, first=False)
            return pruned_perc


def PruningExp(model: EvoPruningModel, prune_type: str,
               percentage: list, schedule: str = 'FIRST', train_time=0) -> dict:
    log.DebugWarning("Weight pruning experiment runing...")
    log.DebugSuccess(f"Pruning {prune_type} with schedule: {schedule}")
    results = dict()

    # Fully connected 1 - con best fc1
    model.set_original_weights()
    model.prune_layers = ['FC1']
    t = time()
    pruned_perc = get_pruned_perc(model, prune_type, schedule, percentage['fc1'])
    t = time() - t
    res_test = model.test_raw_model()
    res_test['pruned_perc'] = pruned_perc
    res_test['time'] = train_time + t
    results['FC1_best_fc1' if prune_type == 'WEIGHT' or prune_type == 'NEURON' else 'FC1'] = res_test


    if prune_type == 'WEIGHT' or prune_type == 'NEURON':
        # Fully connected 1 - con best fc2
        model.set_original_weights()
        model.prune_layers = ['FC1']
        t = time()
        pruned_perc = get_pruned_perc(model, prune_type, schedule, percentage['fc2'])
        t = time() - t
        res_test = model.test_raw_model()
        res_test['time'] = train_time + t
        res_test['pruned_perc'] = pruned_perc
        results['FC1_best_fc2'] = res_test

    # Fully connected 2 - con best fc2
    model.set_original_weights()
    model.prune_layers = ['FC2']
    t = time()
    pruned_perc = get_pruned_perc(model, prune_type, schedule, percentage['fc2'])
    t = time() - t
    res_test = model.test_raw_model()
    res_test['time'] = train_time + t
    res_test['pruned_perc'] = pruned_perc
    results['FC2_best_fc2' if prune_type == 'WEIGHT' or prune_type == 'NEURON' else 'FC2'] = res_test

    if prune_type == 'WEIGHT' or prune_type == 'NEURON':
        # Fully connected 2 - con best fc1
        model.set_original_weights()
        model.prune_layers = ['FC2']
        t = time()
        pruned_perc = get_pruned_perc(model, prune_type, schedule, percentage['fc1'])
        t = time() - t
        res_test = model.test_raw_model()
        res_test['time'] = train_time + t
        res_test['pruned_perc'] = pruned_perc
        results['FC2_best_fc1'] = res_test

    # Both Layers
    model.set_original_weights()
    model.prune_layers = ['FC1', 'FC2']
    t = time()
    pruned_perc = get_pruned_perc(model, prune_type, schedule, percentage['both'])
    t = time() - t
    res_test = model.test_raw_model()
    res_test['pruned_perc'] = pruned_perc
    res_test['time'] = train_time + t
    results['BOTH'] = res_test

    return results
