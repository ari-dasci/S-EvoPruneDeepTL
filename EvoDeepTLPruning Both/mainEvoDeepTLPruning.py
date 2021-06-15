#!/usr/bin/python3
from fitnessConsecutive import Fitness, Optimization_type
from SSbinaryGA import BGA
import sys
import time
from utility import read_dataset
import numpy as np

# LECTURA DE DATOS
# Configuracion del dataset
# aqui leemos todo lo necesario para leer los datos

file_dataset = sys.argv[1]
file = open(file_dataset,"r")
datos = read_dataset(file)


# ejecutamos el modelo genetico
# Parámetros para el GA
file_params_ga = sys.argv[2]
f = open(file_params_ga,"r")
config = f.readline().split(",")

# size de la poblacion
tam_pop = int(config[0])

# longitud salida resnet
size_input = int(config[1])

#longitud de los cromosomas
longitud_cromosoma = int(config[2])

#second layer
siguiente_capa = int(config[3])
#prob de mutacion
p_m = float(config[4])
#max evaluaciones
max_evals = int(config[5])
#tipo de seleccion
select = int(config[6])
#tipo de cruce
crossover = int(config[7])
#init
init = int(config[8])

type_model = int(config[9])


iteracion = str(sys.argv[3])

print(config)
a_guardar = str(tam_pop)+str(longitud_cromosoma)+str(select)+str(crossover)+ str(init) + "-" + iteracion
print(a_guardar)

# EJECUCION DEL ALGORITMO
start_time = time.time()


if type_model == 0: # neuronas
    optimization_type = Optimization_type.NEURONS # no actualizamos la longitud de cromosoma pues es 512 para la primera capa y tambien para el GA
    fitness = Fitness(size_input = size_input,first_layer=longitud_cromosoma,second_layer=siguiente_capa,data=datos, optimType=optimization_type)
    test = BGA(pop_shape=(tam_pop,longitud_cromosoma*2), method=fitness, p_m=p_m, max_evals= max_evals, early_stop_rounds=None, verbose = None, maximum=True, cadena = a_guardar, select= select, crossover=crossover, init = init)
else: # conexiones
    optimization_type = Optimization_type.CONNECTIONS # en este caso, hay que decirle al fitness que la primera capa tiene 512 neuronas y al GA que la longitud del cromosoma es 512*512
    fitness = Fitness(size_input = size_input,first_layer=longitud_cromosoma,second_layer=siguiente_capa,data=datos, optimType=optimization_type)
    test = BGA(pop_shape=(tam_pop,longitud_cromosoma*siguiente_capa), method=fitness, p_m=p_m, 
            max_evals= max_evals, early_stop_rounds=None, verbose = None, maximum=True, cadena = a_guardar, select= select, crossover=crossover, init = init)

best_solution, best_fitness = test.run()

print(best_solution)
print(best_fitness)
print(list(best_solution).count(1))
print("Tiempo transcurrido: ", str(time.time()-start_time))


print("Evaluacion de la mejor solucion: ")
#print(type(best_solution))
#print(len(best_solution))
print("Accuracy mejor solucion: ", str(fitness.fitness(best_solution)))
