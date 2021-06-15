# -*- coding: utf-8 -*-

import numpy as np
import json
try:
    import cPickle as pickle
except:
    import pickle

from Logger import Logger as log


def save_pickle(data, name: str, path: str ='./summary/'):
    with open(path+name, 'wb') as f:
        pickle.dump(data, f)
        f.close()
    log.DebugSuccess(f"Pickle file saved to {path + name}")


def load_pickle(path: str):
    try:
        with open(path, 'r') as f:
            data = pickle.load(f)
            f.close()
        return data
    except FileNotFoundError:
        log.DebugError(f"File {path} not found!")


def save_json(data, name: str, path: str ='./summary/'):
    with open(path+name, 'w') as f:
        f.write(json.dumps(data, indent=4))
        f.close()
    log.DebugSuccess(f"Json file saved to {path + name}")


def load_json(path: str):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            f.close()
        return data
    except FileNotFoundError:
        log.DebugError(f"File {path} not found!")


def results_mean(dict_results: dict, name: str, save_path: str = "./summary/"):
    mean_results = dict_results[0]
    types = ['WEIGHT','NEURON_FIRST','NEURON_SECOND','POLYNOMIAL_DECAY']
    schedule_full = ['FC1_best_fc1', 'FC1_best_fc2',
                     'FC2_best_fc2', 'FC2_best_fc1',
                     'BOTH']
    schedule_standard = ['FC1', 'FC2', 'BOTH']
    schedules = {'WEIGHT': schedule_full,
                 'NEURON_FIRST': schedule_full,
                 'NEURON_SECOND': schedule_full,
                 'POLYNOMIAL_DECAY': schedule_full}

    if len(dict_results) == 1:
        save_json(mean_results, name, save_path)
        return

    # La suma
    for i in dict_results[1:]:
        mean_results['original_train_time'] += i['original_train_time']
        for t in types:
            for s in schedules[t]:
                for k in mean_results[t][s].keys():
                    if type(mean_results[t][s][k]) == dict:
                        for pl in i[t][s][k].keys():
                            mean_results[t][s][k][pl] += i[t][s][k][pl]
                    else:
                        mean_results[t][s][k] += i[t][s][k]
    # La media
    num = len(dict_results)
    mean_results['original_train_time'] /= num
    for t in types:
        for s in schedules[t]:
            for k in mean_results[t][s].keys():
                if type(mean_results[t][s][k]) == dict:
                    for pl in i[t][s][k].keys():
                        mean_results[t][s][k][pl] /= num
                else:
                    mean_results[t][s][k] /= num
    save_json(mean_results, name, save_path)
