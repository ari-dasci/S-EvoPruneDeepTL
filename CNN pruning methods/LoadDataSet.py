# -*- coding: utf-8 -*-

from keras.applications.resnet50 import preprocess_input as resnet_preproc
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator as IDG
from glob import glob

from Logger import Logger as log
import saver


def get_dataset_info(key,dataset_dir):
    dataset_info = saver.load_json(dataset_dir)[key]
    n_folds = dataset_info['num_folds']
    dataset_info['input_shape'] = (dataset_info['width'], dataset_info['height'],
                                   dataset_info['channels'])
    del dataset_info['width'], dataset_info['height'], dataset_info['channels']
    if n_folds > 1:
        train_paths = [f"{dataset_info['train_path']}partition{i}" for i in range(n_folds)] 
        test_paths = [f"{dataset_info['test_path']}partition{i}" for i in range(n_folds)] 
        dataset_info['train_path'] = train_paths
        dataset_info['test_path'] = test_paths
    return dataset_info


def LoadData(path_to_train: str, path_to_test: str,
             preprocessing_function: type(resnet_preproc) = resnet_preproc,
             target_size: tuple = (300, 300),
             file_extension: str = 'png', info: dict = None) -> (iter, iter, dict):
    """
        Load dataset from directory, apply a preprocessing function and return
        data generators for train and test

        Directories are expected to have the next tree distribution:
            main_dir_train (or test)/
            |_ class1/
               |-img1.png
               |-img2.png
               |-...
            |_ class2/
               |-img1.png
               |-img2.png
               |-...
            |_ ...

        return : train and test iterators returning the whole dataset as
                 (samples, classes) and a dictionary containing train
                 and test sizes
    """
    assert path_to_train is not None, log.DebugError('Missing path to train')
    assert path_to_test is not None, log.DebugError('Missing path to test')
    height, width = target_size

    path_to_train = path_to_train+'/' if path_to_train[-1] != '/' else path_to_train
    path_to_test = path_to_test+'/' if path_to_test[-1] != '/' else path_to_test
    if info is not None:
        try:
            len_train = info['len_train']
        except KeyError:
            len_train = len(glob(f"{path_to_train}*/*.{file_extension}", recursive=True))
        try:
            len_test = info['len_test']
        except KeyError:
            len_test = len(glob(f"{path_to_test}*/*.{file_extension}", recursive=True))
    else:
        len_train = len(glob(f"{path_to_train}*/*.{file_extension}", recursive=True))
        len_test = len(glob(f"{path_to_test}*/*.{file_extension}", recursive=True))

    classes = len(glob(f"{path_to_train}*"))

    log.DebugInfo(f"I\'ve found {len_train} images for train")
    log.DebugInfo(f"I\'ve found {len_test} images for test")

    generadorTrain = IDG(preprocessing_function=preprocessing_function)
    generadorTest = IDG(preprocessing_function=preprocessing_function)
    

    log.DebugInfo(f"I\'ve found {classes} different classes")
    
    iterator_train = generadorTrain.flow_from_directory(path_to_train,
                                                        batch_size=len_train,
                                                        target_size=(height, width),
                                                        shuffle=True)

    iterator_test = generadorTest.flow_from_directory(path_to_test,
                                                      batch_size=len_test,
                                                      target_size=(height, width),
                                                      shuffle=False)

    info = {'len_train': len_train, 'len_test': len_test, 'num_classes': classes}
    log.DebugSuccess("Data loaded successfully")
    return iterator_train, iterator_test, info
