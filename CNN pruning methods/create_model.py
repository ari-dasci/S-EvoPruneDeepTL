# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import ResNet50
from keras.models import Model


def create_pruneable_model(input_shape : tuple, num_classes : int, fc1_input : int = 512,
                           fc2_input : int = 512) -> type(Model):
        """ Create custom model using ResNet50 as feature extractor  
        
        input_shape : Expected input data shape
        num_classes : Number of classes, used as output neurons
        fc1_input, fc2_input : Number of neurons on each Dense layer
        """
        ModelResNet50 = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
        ModelResNet50.trainable = False
        flattened = Flatten()(ModelResNet50.output)
        fc1 = Dense(fc1_input, activation="relu", name="FC1")(flattened)
        fc2 = Dense(fc2_input, activation="relu", name="FC2")(fc1)
        out = Dense(num_classes, activation="softmax", name="outFC")(fc2)
        model = Model(inputs=ModelResNet50.input, outputs=out)
        return model
