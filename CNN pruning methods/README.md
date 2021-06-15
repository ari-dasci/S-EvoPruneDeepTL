# EvoPruning

Para hacer funcionar el trastochisme este en principio será suficiente con que pogas todos los Datasets bajo una carpeta llamada Datos (junto con todos los .py) y añadas la info correspondiente al archivo .json. Tiene la siguiente forma:
```json
    "RPS":
    {
        "height" : 300,
        "width" : 300,
        "channels" : 3,
        "num_folds" : 1,
        "train_path" : "./Datos/rps/rps/",
        "test_path" : "./Datos/rps/rps-test-set/",
        "file_format" : "png",
        "best_fc1" : 0.98,
        "best_fc2" : 0.98,
        "best_both_fc1": 0.605,
        "best_both_fc2": 0.637,
        "len_train": 2520,
        "len_test": 372
    }
```

Si quieres se puede añadir el `len_train` y `len_test` para que no haga falta tener todas las imagenes con la misma extension. También importante que los valores que pongamos en `best_fc1`, `best_fc2` y `best_both_{fc1/fc2}` se correspondan a los del paper.

Explico tambien un poquito qué experimentos se realizan:
* Para todos los tipos de pruning (`WEIGHTS`,`NEURONS (FIRST/SECOND)`,`POLYNOMIAL_DECAY`) se evalua su funcionamiento pruneando la capa FC1 (Esta seria la parte de la salida de 2048 con la primera de 512), la FC2 (conexiones entre las dos de 512) y BOTH (tanto FC1 como FC2)
* Los resultados he decidido guardarlos en formato JSON para que sea super easy leerlos. Los resultados se guardan en `summary`;
    *  Hay un json dentro de summary que tiene la media de todos los runs
    * Bajo el nombre de cada DATASET_RUN tienes todo lo demás: Modelos y json separados.
    * He hecho que todos los approaches pruneen un modelo con los mismos pesos, asi me parecia mas fair. Es decir, el modelo se entrena una vez y se prunea con todos los approaches, asi que de lujo, no hay que andar entrenando mil modelos diferentes xD
