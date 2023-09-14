#!/usr/bin/python3
from fitnessCKA import Fitness, Optimization_type
from SSbinaryGA import BGA
import sys
import time
from utility import read_dataset
import numpy as np
import argparse
from utility import read_population
from scipy.spatial import distance
from cka import *

# LECTURA DE PARAMETROS

parser = argparse.ArgumentParser(description="EvoPruneDeepTL", add_help=True)
parser.add_argument("--dataset", help="Config file for dataset", default="")
parser.add_argument("--configGA", help="Config for the GA", default="configGA.csv")
parser.add_argument("--run", help="Iteration number of EvoPruneDeepTL", type=int)
parser.add_argument("--extractor",help="Extractor", default="resnet50")


args = parser.parse_args()

file_dataset = args.dataset
file_params_ga = args.configGA
iteracion = args.run
extractor = args.extractor
#from_dir = args.features

# LECTURA DE DATOS
# Configuracion del dataset
# aqui leemos todo lo necesario para leer los datos

#file_dataset = sys.argv[1]


file = open(file_dataset,"r")
datos = read_dataset(file,extractor)


# ejecutamos el modelo genetico
# Parámetros para el GA
f = open(file_params_ga,"r")
config = f.readline().split(",")


# size de la poblacion
tam_pop = int(config[0])

# tipo: first = FS, second = pruning
tipo = str(config[1])

# input
input_size = int(config[2])

# priemra layer
primera = int(config[3])

# segunda layer (si hubiese)
segunda = int(config[4])

# max evaluaciones
max_evals = int(config[5])

# neuronas = 0, conexiones = 1
type_model = int(config[6])

# info sobre las capas
how_many = int(config[7])
which_sparse = int(config[8])
both_sparse = int(config[9])

info_layers = [how_many, which_sparse, both_sparse]

    
#iteracion = str(sys.argv[3])

print("Configuracion")
print(config)

a_guardar = str(tam_pop)+str(tipo)+str(input_size) + "-" + str(extractor) + "-" + str(iteracion)
print(a_guardar)
    
# EJECUCION DEL ALGORITMO
start_time = time.time()

if type_model == 0: # neuronas
    optimization_type = Optimization_type.NEURONS # no actualizamos la longitud de cromosoma pues es 512 para la primera capa y tambien para el GA
    longitud_cromosoma = -1

    if tipo == "second":
        if how_many == 1:
            longitud_cromosoma = primera
        else:
            if which_sparse == 1:
                longitud_cromosoma = primera
            elif which_sparse == 2:
                if both_sparse == 0:
                    longitud_cromosoma = segunda
                else:
                    longitud_cromosoma = primera+segunda
    else:
        longitud_cromosoma = input_size

    fitness = Fitness(input_size=input_size ,first_layer=primera,second_layer=segunda,data=datos, layer=tipo, num_layers = info_layers,optimType=optimization_type, extractor = extractor)

else: # conexiones
    optimization_type = Optimization_type.CONNECTIONS # en este caso, hay que decirle al fitness que la primera capa tiene 512 neuronas y al GA que la longitud del cromosoma es 512*512

    fitness = Fitness(input_size=input_size, first_layer=primera,second_layer=segunda,data=datos, layer = tipo, num_layers = info_layers,optimType=optimization_type, extractor = extractor)

    longitud_cromosoma = -1

    if how_many == 1:
    	longitud_cromosoma = input_size * primera
    else:
    	if which_sparse == 1:
    		longitud_cromosoma = input_size * primera
    	elif which_sparse == 2:
    		longitud_cromosoma = primera * segunda



## leemos el mejor de la ejecucion y la pop

if "OJOS" in datos[0]:
    dataset = "Ojos"
elif "RPS" in datos[0]:
    dataset = "RPS"
else:
    if "iconography" in datos[0]:
        dataset = "PinturasIconography"
    else:
        dataset = "Pinturas"

clase = datos[0][-1]

if tipo == "second":
    modelo = "both"
else:
    modelo = "fs"


if "iconography" in datos[0]:
    porcentaje = datos[0][-2:] 
    f = "./resultados"+dataset+"/"+"Icono"+porcentaje+"/"+modelo+"/"
else:
    f = "./resultados"+dataset+"/"+"Clase"+clase+"/"+modelo+"/"


f_mejor = f +"mejor_"+str(datos[0])+"_"+str(tipo)+"_"+str(iteracion)+".txt"
print(f_mejor)

archivo = open(f_mejor, "r")
line_mejor = archivo.readline().split(",")

#  OJOS_030first2048-resnet50-1.txt
#  OJOS_030second2048-resnet50-1.txt
f_pop = f + datos[0] + str(tam_pop)+str(tipo)+str(input_size)+"-"+str(extractor)+"-"+str(iteracion)+".txt"
print(f_pop)
pop, accs = read_population(f_pop)


# lo convertimos a lista de enteros
mejor = list(map(int, line_mejor[:-1]))

# escogemos el más cercano segun la distancia de Hamming
hamming_distances = []
new_pop = []

for element in pop:
    el = list(map(int,element))
    new_pop.append(el)
    hamming_distances.append(distance.hamming(el,mejor))

# quitarse a el mismo y calcular el mas cercano
min_val_idxs = [x for x in range(len(hamming_distances)) if hamming_distances[x] == min(hamming_distances)]
print(min_val_idxs)
print(min(hamming_distances))

new_pop.pop(min_val_idxs[0])
hamming_distances.pop(min_val_idxs[0])

print(len(hamming_distances))
print(len(new_pop))

# calculamos ahora el mas cercano
print("El minimo ahora es: ")
minimo = min(hamming_distances)
print(minimo)


min_val_idxs = [x for x in range(len(hamming_distances)) if hamming_distances[x] == min(hamming_distances)]
print(min_val_idxs)

cercanos = [new_pop[i] for i in range(len(new_pop)) if i in min_val_idxs][0]

# lo mismo para el mas cercano
print("Hamming minima: " + str(minimo))


if tipo == "first":
    # entrenamos y tomamos la matriz de pesos del mejor
    acc1, weights_best = fitness.fitness(mejor)
    print(weights_best.shape)
    
    # cogemos el más cercano
    acc2, weights_closest = fitness.fitness(cercanos)
    print(weights_closest.shape)

    # calculamos ahora el CKA linea y RBF de las matrices de pesos
    #print('Linear CKA, between best and closest: {}'.format(linear_CKA(weights_best, weights_closest)))
    #print('Linear CKA, between best and best: {}'.format(linear_CKA(weights_best, weights_best)))

    print('RBF Kernel CKA, between best and closest: {}'.format(kernel_CKA(weights_best, weights_closest)))
    print('RBF Kernel CKA, between best and best: {}'.format(kernel_CKA(weights_best, weights_best)))
else:
    _, weights_best = fitness.fitness(mejor)
    
    print(weights_best[0].shape)
    print(weights_best[1].shape)

    acc2, weights_closest = fitness.fitness(cercanos)
    
    print("CKA closest layer 1: {}".format(kernel_CKA(weights_best[0], weights_closest[0])))
    print("CKA closest layer 1: {}".format(kernel_CKA(weights_best[0], weights_best[0])))

    print("CKA closest layer 2: {}".format(kernel_CKA(weights_best[1], weights_closest[1])))
    print("CKA closest layer 2: {}".format(kernel_CKA(weights_best[1], weights_best[1])))


##################################### MODELO NO PRUNEADO

if tipo == "first":
    sin_pruning = [1] * 2048
else:
    sin_pruning = [1] * 1024


if tipo == "first":
    print("Hamming entre best y no pruning " + str(distance.hamming(mejor,sin_pruning)))
    _, weights_nopruning = fitness.fitness(sin_pruning)
    #print('Linear CKA, between best and no pruned: {}'.format(linear_CKA(weights_best, weights_nopruning)))
    #print('Linear CKA, between best and best: {}'.format(linear_CKA(weights_best, weights_best)))

    print('RBF Kernel CKA, between best and no pruned: {}'.format(kernel_CKA(weights_best, weights_nopruning)))
    print('RBF Kernel CKA, between best and best: {}'.format(kernel_CKA(weights_best, weights_best)))
else:
    print("Hamming entre best y no pruning " + str(distance.hamming(mejor,sin_pruning)))
    _, weights_nopruning = fitness.fitness(sin_pruning)

    print("CKA no pruned layer 1: {}".format(kernel_CKA(weights_best[0], weights_nopruning[0])))
    print("CKA no pruned layer 1: {}".format(kernel_CKA(weights_best[0], weights_best[0])))

    print("CKA no pruned layer 2: {}".format(kernel_CKA(weights_best[1], weights_nopruning[1])))
    print("CKA no pruned layer 2: {}".format(kernel_CKA(weights_best[1], weights_best[1])))
