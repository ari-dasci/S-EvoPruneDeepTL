import numpy as np
from sparse_layer import Sparse
import keras
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout

def decode_chromosome(element, n):
    final = [element[i * n:(i + 1) * n] for i in range((len(element) + n - 1) // n)]
    return np.array(final)

def extend_chromosome(element, rep):
    longitud = len(element)
    final = []

    for i in range(longitud):
        content = element[i]
        repeated_content = [content] * rep 
        final.append(repeated_content)

    # devolvemos el vector
    flat_list = []
    for sublist in final:
        for item in sublist:
            flat_list.append(item)

    return flat_list


def build_model(connections, shape,num_classes,bias):
    #print("El bias es... ", bias)
    sparse = Sparse(adjacency_mat=connections, activation = "relu", use_bias = bias)

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape= shape))
    model.add(sparse)
    model.add(Dense(num_classes, activation='softmax'))
    
    opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary(), flush = True)

    return model

def build_reference_model(shape,second_layer,num_classes,drop,first_layer,bias):
    #print("Creando modelo")
    print("El bias es... ", bias)
    model = Sequential()
    model.add(Dense(first_layer, activation='relu', input_shape=shape)) # antes 512

    if drop >= 0:
        if drop == 1:
            model.add(Dropout(0.5))
        
    model.add(Dense(second_layer, activation='relu', use_bias = bias))
    model.add(Dense(num_classes, activation='softmax'))

    opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary(), flush = True)

    return model
