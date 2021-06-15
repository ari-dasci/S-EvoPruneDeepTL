# S-EvoDeepTLPruning

This is the official repository of EvoDeepTLPruning: Evolutionary Algorithm for Deep Transfer Learning by Pruning Neurons in Dense Layers

## Code

The implementation of EvoDeepTLPruning is divided in the following folders:

   * EvoDeepTLPruning FC1 FC2: the folder contains the python files for the one layer approaches.
   * EvoDeepTLPruning Both: this folders contains the python files for the both layer approach.
   * CNN pruning methods: contains the implementation of the compared CNN pruning methods in the paper.
   * configs: contains the configuration files for each analyzed dataset in the paper.
   * convergence images: it contains the images for the convergence of some used datasets.
  
## Datasets

The used datasets in this paper can be downloaded from:

  * SRSMAS: https://sci2s.ugr.es/CNN-coral-image-classification
  * RPS: https://www.tensorflow.org/datasets/catalog/rock_paper_scissors
  * LEAVES: https://www.tensorflow.org/datasets/catalog/citrus_leaves
  * PAINTING: https://www.kaggle.com/thedownhill/art-images-drawings-painting-sculpture-engraving
  * PLANTS: https://github.com/pratikkayal/PlantDoc-Dataset
  * CATARACT: https://www.kaggle.com/jr2ngb/cataractdataset

## Results

EvoDeepTLPruning is able to optimize sparse layers using a genetic algorithm, giving a neural scheme as it is shown.

![Image0](https://github.com/ari-dasci/S-EvoDeepTLPruning/tree/main/images/sparseRepresentation.png)
<img src="https://github.com/ari-dasci/S-EvoDeepTLPruning/tree/main/images/sparseRepresentation.png" width="100" height="100">

The following table shows the results of EvoDeepTLPruning when the comparison is made against CNN pruning methods.

![Image0](https://github.com/ari-dasci/S-EvoDeepTLPruning/blob/main/images/resultsEvoDeepTLPruningCNN.png)


## Convergence Plots

We show some of the convergence plots taken from our experiments:

