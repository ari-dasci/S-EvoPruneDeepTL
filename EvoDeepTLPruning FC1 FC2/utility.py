import numpy as np
from numpy import array
import os, sys
import glob
from keras.preprocessing.image import ImageDataGenerator as IDG
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet

def read_dataset(data):
    generador = IDG(preprocessing_function = preprocess_input_resnet)
    generadorTest = IDG(preprocessing_function =  preprocess_input_resnet)

    # lectura de los datos
    config = data.readline().split(",")
    # en cofing tenemos todo lo necesario para leer el dataset
    print(config)
    name = str(config[0])
    #dir_train = "./"+str(config[1])
    #dir_test = "./"+str(config[2])
    dir_train = str(config[1])
    dir_test = str(config[2])
    #len_train = int(config[3])
    #len_test = int(config[4])

    #len_train = len([f for f in dir_train if os.path.isfile(os.path.join(dir_train
    #len_test = 
    num_folds = int(config[3])
    height = int(config[4])
    width = int(config[5])
    num_classes = int(config[6])
    it = []
    it_val = []

    print(dir_train)
    print(dir_test)

    if dir_train != "-":
        if num_folds == 1:
            len_train = count_files(dir_train)
            iterator_train = generador.flow_from_directory(dir_train,batch_size=len_train,target_size=(height,width),shuffle=True)
        else:
            for i in range(num_folds):
                #leemos su ruta final
                dir_final = dir_train+str(i)+"/"
                len_train = count_files(dir_final)
                print(len_train)
                print(dir_final)
                iterator = generador.flow_from_directory(dir_final,batch_size=len_train,target_size=(height,width),shuffle=True)
                it.append(iterator)

    if dir_test != "-":
        if num_folds == 1:
            len_test = count_files(dir_test)  
            iterator_test = generadorTest.flow_from_directory(dir_test,batch_size=len_test,target_size=(height,width),shuffle=False)
        else:
            for i in range(num_folds):
                dir_final = dir_test+str(i)+"/"
                len_test = count_files(dir_final)
                print(len_test)
                print(dir_final)
                iterator_val = generadorTest.flow_from_directory(dir_final,batch_size=len_test,target_size=(height,width),shuffle=False)
                it_val.append(iterator_val)

    if num_folds != 1:
        iterator_train = it
        iterator_test = it_val

    print(type(iterator_train))

    return [name,iterator_train,iterator_test,len_train,len_test,num_classes]

def count_files(folder):
    total = 0

    for root, dirs, files in os.walk(folder):
        total += len(files)

    return total
def read_population(file):
    population = []
    accuracies = []

    f = open(file,"r")
    count = 0
    line = f.readline().split(",")

    while True:
        population.append(line[0])
        accuracies.append(line[1])
        count += 1

        line = f.readline()

        if not line:
            break

        line = line.split(",")

    f.close()
    #print(count)
    return population,accuracies

def get_best_elements(num,population,accuracies):
    best_pop = []
    best_accs = []

    indexes = sorted(range(len(accuracies)), key=lambda i: accuracies[i],reverse=True)[:num]
    #print(indexes)

    for i in range(len(indexes)):
        best_accs.append(accuracies[indexes[i]])
        best_pop.append(population[indexes[i]])

    #print(best_accs)
    return best_pop,best_accs,indexes