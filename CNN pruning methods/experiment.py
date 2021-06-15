# -*- coding: utf-8 -*-

import numpy as np

import saver
from Logger import Logger as log
from Model import EvoPruningModel
import directory_manager as dmanager
from LoadDataSet import LoadData, get_dataset_info
from weight_pruning_experiment import PruningExp
from time import time


def Experiment(dataset_name, dataset_dir,folder_name = None):
    if folder_name == None: folder_name = dataset_name
    dmanager.prepare_directory()
    dmanager.create_dir_in_summaries(folder_name)
    dmanager.create_dir_in_summaries(f"{folder_name}/models")

    info = get_dataset_info(dataset_name, dataset_dir)
    all_results = []

    for i in range(info['num_folds']):
        info = get_dataset_info(dataset_name, dataset_dir)
        log.DebugInfo(f"Fold {i+1}")


        if info['num_folds'] == 1:
            train_path = info['train_path']
            test_path = info['test_path']
        else:
            train_path = info['train_path'][i]
            test_path = info['test_path'][i]

        traingen, testgen, _info = LoadData(train_path, test_path,
                                           target_size=info['input_shape'][:-1],
                                           file_extension=info['file_format'],
                                           info = info)
        info.update(_info)

        # Empezamos entrenando la red hasta que Early Stopping diga que paremos
        model = EvoPruningModel(traingen, testgen, info, final_sparsity=[0.9, 0.9])
        
        # Tiempo de training
        training_time = time()
        res = model.train_without_pruning()
        training_time = time() - training_time
        
        test_res = model.test_raw_model()
        #model.save_raw_model(f"./summary/{folder_name}/models/notPruned-Fold_{i}.h5")
        saver.save_json({'train': res, 'test': test_res}, f"notPruned-Fold_{i}.json", path=f'./summary/{folder_name}/')

        pruning_percents = {'fc1': info['best_fc1'],
                            'fc2': info['best_fc2'], #esto era con fc2
                            'both': [info['best_both_fc1'], info['best_both_fc2']]}

        # Diferentes tipos de pruning aplicados a la misma red
        
        resw = PruningExp(model, 'WEIGHT', pruning_percents, train_time=training_time)
        #model.save_raw_model(f"./summary/{folder_name}/models/weightPruned-Fold_{i}.h5")
        
        resnf = PruningExp(model, 'NEURON', pruning_percents, 'FIRST', train_time=training_time)
        #model.save_raw_model(f"./summary/{folder_name}/models/neuronPrunedFirst-Fold_{i}.h5")

        
        resns = PruningExp(model, 'NEURON', pruning_percents, 'SECOND', train_time=training_time)
        #model.save_raw_model(f"./summary/{folder_name}/models/neuronPrunedSecond-Fold_{i}.h5")
        
        ## Polynomial Decay
        poly_res = {}

        fc1_poly = EvoPruningModel(traingen, testgen, info,
                                   pruneFC2 = False, final_sparsity=pruning_percents['fc1'] )
        fc1_poly_fc2 = EvoPruningModel(traingen, testgen, info,
                                   pruneFC2 = False, final_sparsity=pruning_percents['fc2'] )

        fc2_poly = EvoPruningModel(traingen, testgen, info,
                                   pruneFC1 = False, final_sparsity=pruning_percents['fc2'] )
        fc2_poly_fc1 = EvoPruningModel(traingen, testgen, info,
                                   pruneFC1 = False, final_sparsity=pruning_percents['fc1'] )
        
        both_poly = EvoPruningModel(traingen, testgen, info,
                                    final_sparsity=pruning_percents['both'])

        fc1_poly.model_to_prune.set_weights(model.raw_model_weights)
        fc1_poly_fc2.model_to_prune.set_weights(model.raw_model_weights)
        fc2_poly.model_to_prune.set_weights(model.raw_model_weights)
        fc2_poly_fc1.model_to_prune.set_weights(model.raw_model_weights)
        both_poly.model_to_prune.set_weights(model.raw_model_weights)

        # Pruning + entrenar 5 gens ... * 5 veces
        fc1_ttime = time()
        fc1_poly.train(26)
        fc1_ttime = time() - fc1_ttime

        fc1_ttime_fc2 = time()
        fc1_poly_fc2.train(26)
        fc1_ttime_fc2 = time() - fc1_ttime_fc2

        fc2_ttime = time()
        fc2_poly.train(26)
        fc2_ttime = time() - fc2_ttime

        fc2_ttime_fc1 = time()
        fc2_poly_fc1.train(26)
        fc2_ttime_fc1 = time() - fc2_ttime_fc1

        both_ttime = time()
        both_poly.train(26)
        both_ttime = time() - both_ttime

        respd1 = fc1_poly.test()
        pruned_perc = fc1_poly.decay_prun_perc()
        respd1['pruned_perc'] = pruned_perc
        respd1['time'] = training_time + fc1_ttime
        #fc1_poly.save(f"./summary/{folder_name}/models/PolynomialFC1-Fold_{i}.h5")
        poly_res['FC1_best_fc1'] = respd1
        
        respd1 = fc1_poly_fc2.test()
        pruned_perc = fc1_poly_fc2.decay_prun_perc()
        respd1['pruned_perc'] = pruned_perc
        respd1['time'] = training_time + fc1_ttime_fc2
        #fc1_poly_fc2.save(f"./summary/{folder_name}/models/PolynomialFC1(best_fc2)-Fold_{i}.h5")
        poly_res['FC1_best_fc2'] = respd1

        respd2 = fc2_poly.test()
        pruned_perc = fc2_poly.decay_prun_perc()
        respd2['pruned_perc'] = pruned_perc
        respd2['time'] = training_time + fc2_ttime
        #fc2_poly.save(f"./summary/{folder_name}/models/PolynomialFC2-Fold_{i}.h5")
        poly_res['FC2_best_fc2'] = respd2
        
        respd2 = fc2_poly_fc1.test()
        pruned_perc = fc2_poly_fc1.decay_prun_perc()
        respd2['pruned_perc'] = pruned_perc
        respd2['time'] = training_time + fc2_ttime_fc1
        #fc2_poly_fc1.save(f"./summary/{folder_name}/models/PolynomialFC2(best_fc1)-Fold_{i}.h5")
        poly_res['FC2_best_fc1'] = respd2

        respdboth = both_poly.test()
        pruned_perc = both_poly.decay_prun_perc()
        respdboth['pruned_perc'] = pruned_perc
        respdboth['time'] = training_time + both_ttime
        #both_poly.save(f"./summary/{folder_name}/models/PolynomialBoth-Fold_{i}.h5")
        poly_res['BOTH'] = respdboth
        

        res = {'WEIGHT': resw, 'NEURON_FIRST': resnf,
                'NEURON_SECOND': resns, 'POLYNOMIAL_DECAY': poly_res, 'original_train_time': training_time}
        saver.save_json(res, f"WeightPruning-Fold_{i}.json", f"./summary/{folder_name}/")

        all_results.append(res)

    saver.results_mean(all_results, f"{dataset_name}_results.json", f"./summary/{folder_name}/")
