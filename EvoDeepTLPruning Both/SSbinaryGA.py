#!usr/bin/env python
import numpy as np
import random
from numpy import asarray
from numpy import savetxt
from utility import get_best_elements
import keras
from scipy.spatial.distance import hamming
from fitnessConsecutive import Fitness

class BGA():
    def __init__(self, pop_shape, method,cadena, select, crossover,init,elitism = True, p_m=0.2, max_evals = 1000, early_stop_rounds=None, verbose = None, maximum=True):
        self.pop_shape = pop_shape
        self.method = method
        self.cadena = cadena
        self.evaluaciones = 0
        self.max_evals = max_evals
        self.p_c = 1
        self.p_m = p_m
        self.pop = np.zeros(pop_shape)
        self.fitness = np.zeros(pop_shape[0])
        self.early_stop_rounds = early_stop_rounds
        self.verbose = verbose
        self.maximum = maximum
        self.memory = dict()
        self.elitism = elitism
        self.selection_type = select
        self.crossover_type = crossover
        self.init = init


    def bin2dec(self, element):
        return int("".join(str(x) for x in element), 2)


    def update_memory(self):
        for i in range(self.pop_shape[0]):
            self.memory[self.bin2dec(self.pop[i])] = self.fitness[i]

    def evaluation(self, pop):
        """
        Computing the fitness of the input popluation matrix.
        Args:
            p: The population matrix need to be evaluated.
        """
        evaluaciones = 0
        resultados = []

        for i in range(len(pop)):
            ind = pop[i]
            code_ind = self.bin2dec(ind)
            
            ##print(code_ind)
            if code_ind in self.memory:
                resultados.append(self.memory[code_ind])
            else:
                acc = self.method.fitness(ind)
                
                f = open(self.method.get_problem_name()+self.cadena+".txt", "a+")

                for c in ind:
                    f.write(str(c))

                f.write(",")
                f.write(str(acc))
                f.write("\n")
                f.close()

                evaluaciones += 1
                # calculamos el accuracy final
                resultados.append(acc)
            
        return np.array(resultados), evaluaciones

    def initialization(self):
        num_individuos = self.pop_shape[0]
        unidades = [i for i in range(self.pop_shape[1])]
        
        print(len(unidades))
        poblacion = []

        for i in range(num_individuos):
            if self.init == 1:
                p_1 = 0.3
            elif self.init == 2:
                p_1 = 0.4
            elif self.init == 3:
                p_1 = 0.5
            elif self.init == 4:
                p_1 = 0.6
            elif self.init == 5:
                p_1 = 0.7
            elif self.init == 6:
                p_1 = random.uniform(0.3,0.7)
            else:
                p_1 = random.uniform(0,1)

                while p_1 == 0:
                    p_1 = random.uniform(0,1)

            num_unos = round(p_1*self.pop_shape[1])
            print("Numero unos: " + str(num_unos))

            individuo = np.zeros(self.pop_shape[1], dtype=np.bool)

            # seleccionamos todos los 1
            indexes = np.random.permutation(unidades)[:num_unos]
            individuo[indexes] = True
            poblacion.append(individuo)


        poblacion = [list(map(int,poblacion[i])) for i in range(self.pop_shape[0])]
        self.pop= np.array(poblacion)
        
        print(self.pop.shape)
        self.fitness, evaluations = self.evaluation(self.pop)
        self.evaluaciones += evaluations

        #print("Poblacion inicializada")
        self.update_memory()
        #print("Memoria actualizada")

    def crossover(self, ind_0, ind_1):
        """
        Single point crossover.
        Args:
            ind_0: individual_0
            ind_1: individual_1
        Ret:
            new_0, new_1: the individuals generatd after crossover.
        """
        assert(len(ind_0) == len(ind_1))

        point = np.random.randint(len(ind_0))
        new_0 = np.hstack((ind_0[:point], ind_1[point:]))
        new_1 = np.hstack((ind_1[:point], ind_0[point:]))

        assert(len(new_0) == len(ind_0))
        return new_0, new_1

    def crossover_1(self, ind_0, ind_1):
        """
        Crossover en el que el gen de cada nuevo elemento se toma 1 a 1 según los padres
        :param ind_0:
        :param ind_1:
        :return: dos hijos
        """

        assert(len(ind_0) == len(ind_1))
        new_0 = []
        new_1 = []

        indexes = np.random.rand(len(ind_0)) < 0.5

        new_0 = np.where(indexes, ind_0, ind_1)
        new_1 = np.where(indexes, ind_1, ind_0)

        return np.array(new_0), np.array(new_1)

    def selection_nam(self):
        size_pop = self.pop_shape[0]
        parent1, *candidates = np.random.permutation(size_pop)[:4]
        dist = [hamming(self.pop[parent1,:],self.pop[c,:]) for c in candidates]
        parent2 = candidates[np.argmax(dist)]

        print("Parent 1: " + str(parent1))
        print("Candidates: " + str(candidates))
        print(dist)
        print("Parent 2: " + str(parent2))

        return parent1, parent2

    def mutation(self, indi):
        """
        Simple mutation.
        Arg:
            indi: individual to mutation.
        """
        point = np.random.randint(len(indi))
        indi[point] = 1 - indi[point]
        return indi


    def rws(self, size, fitness):
        """
        Roulette Wheel Selection.
        Args:
            size: the size of individuals you want to select according to their fitness.
            fitness: the fitness of population you want to apply rws to.
        """
        if self.maximum:
            fitness_ = fitness
        else:
            fitness_ = 1.0 / fitness
#         fitness_ = fitness
        idx = np.random.choice(np.arange(len(fitness_)), size=size, replace=True,
               p=fitness_/fitness_.sum()) # p
        return idx

    def processPopulation(self):
        list_idx = list(np.argsort(-self.fitness))

        best_value = self.fitness[list_idx[0]]
        best_num_ones = list(self.pop[list_idx[0],:]).count(1)
        best_index = 0

        idx = 1
        length = len(list_idx) # es la misma longitud que el vector de fitness

        next_value = self.fitness[list_idx[idx]]
        next_num_ones = list(self.pop[list_idx[idx],:]).count(1)


        # si no coinciden el primero con el segundo es porque el primero es mejor que el segundo
        if next_value == best_value: # si son el mismo valor, miramos cual de ellos tiene un menor numero de unos
            if next_num_ones < best_num_ones: # si el segundo elemento tiene menos 1's, como tiene el mismo fitness, nos quedamos con ese como mejor elemento
                best_index = 1

        idx = idx + 1

        while next_value == best_value and idx < length:
            next_value = self.fitness[list_idx[idx]]
            next_num_ones = list(self.pop[list_idx[idx],:]).count(1)

            if next_value == best_value:
                if next_num_ones < best_num_ones:
                    best_index = idx

            idx = idx + 1
            
        return best_index

    def run(self):
        """
        Run the genetic algorithm.
        Ret:
            global_best_ind: The best indiviudal during the evolutionary process.
            global_best_fitness: The fitness of the global_best_ind.
        """
        global_best = 0
        self.initialization()

        print("Despues inicializaion")
        print(type(self.fitness))
        best_index = self.processPopulation()
        #best_index = np.argsort(-self.fitness)[0]
        global_best_fitness = self.fitness[best_index]
        global_best_ind = self.pop[best_index, :]
        
        self.evaluaciones = self.pop_shape[0]

        print(self.pop)
        print(self.fitness)
        print(best_index)
        print(global_best_fitness)
        count = 0
        it = 0

        print("Evals ", str(self.evaluaciones))
        print("Max evals ", str(self.max_evals))

        while (self.evaluaciones < self.max_evals):
            next_gene = []
            print("ITERACION " + str(it+1))
            print("Evaluaciones: " + str(self.evaluaciones))
            #print("Tam pop " + str(self.pop_shape[0]))

        
            # seleccion de padres
            if self.selection_type == 0:
                i, j = self.rws(2, self.fitness) # choosing 2 individuals with rws.
            else:
                i,j = self.selection_nam()

            indi_0, indi_1 = self.pop[i, :].copy(), self.pop[j, :].copy()

            # crossover
            if np.random.rand() < self.p_c:
                if self.crossover_type == 0:
                    indi_0, indi_1 = self.crossover(indi_0, indi_1)
                elif self.crossover_type == 1:
                    indi_0, indi_1 = self.crossover_1(indi_0, indi_1)

            if np.random.rand() < self.p_m:
                indi_0 = self.mutation(indi_0)
                indi_1 = self.mutation(indi_1)

            next_gene.append(indi_0)
            next_gene.append(indi_1)

            children_pop = np.array(next_gene)
            children_fitness, evaluaciones = self.evaluation(children_pop)
            print("Evaluacion de hijos acabada")
            self.update_memory()
            self.evaluaciones += evaluaciones

            # debemos ahora quedarnos con los 50 mejores elementos
            # tenemos en children_pop y en children_fitness los elementos y sus fitness resp y...
            # en self.pop y self.fitness la poblacion de 50.
            sorted_pop, sorted_accs, indices = get_best_elements(len(self.pop), self.pop, self.fitness)

            insert = [False, False]

            for i in range(len(children_fitness)):
                acc_hijo = children_fitness[i]
                hijo = children_pop[i]
                longitud = len(sorted_accs)
                hijo_insertado = False
                last_two = sorted_accs[-2:]

                if not insert[1]:
                    if acc_hijo > last_two[1]:
                        sorted_accs[longitud-1] = acc_hijo
                        sorted_pop[longitud-1] = hijo
                        insert[1] = True
                        hijo_insertado = True

                if not insert[0]:
                    if acc_hijo > last_two[0] and not hijo_insertado:
                        sorted_accs[longitud-2] = acc_hijo
                        sorted_pop[longitud-2] = hijo
                        insert[0] = True

            self.pop = np.array(sorted_pop)
            self.fitness = np.array(sorted_accs)

            print(self.fitness)

            # actualizamos los datos del mejor tras ordenar
            best_index = self.processPopulation()
            global_best_fitness = self.fitness[best_index]
            global_best_ind = self.pop[best_index,:]

            print(best_index)
            print(self.fitness[best_index])
            print(self.pop[best_index,:])

            
            if self.elitism:
                worst_index = np.argsort(-self.fitness)[-1]
                self.pop[worst_index, :] = global_best_ind
                self.fitness[worst_index] = global_best_fitness

                
            it += 1
            print("Poblacion: ",self.pop)
            print("Fitness: ",self.fitness)
            print("Mejor pop: " + str(global_best_fitness))
            print("Num unos primera mitad: " + str(list(global_best_ind)[:512].count(1)))
            print("Num unos segunda mitad: " + str(list(global_best_ind)[512:].count(1)))

            print("Global index: " + str(best_index))
            print("Mejor pop: " +str(self.pop[best_index,:]))
            print("Mejor fitness: " + str(self.fitness[best_index]))
            #save_population()
        
        print('\n Solution: {} \n Fitness: {} \n Evaluation times: {}'.format(global_best_ind, global_best_fitness, self.evaluaciones))
        return global_best_ind, global_best_fitness
