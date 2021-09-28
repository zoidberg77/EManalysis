<h1 align="center">
  <b>EManalysis</b><br>
</h1>

<p align="center">
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.8-ff69b4.svg" /></a>
    <a href= "https://pytorch.org/">
      <img src="https://img.shields.io/badge/PyTorch-1.8-2BAF2B.svg" /></a>
    <a href= "https://github.com/zudi-lin/pytorch_connectomics/blob/master/LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-blue.svg" /></a>
</p>

<p align="center">
  <img width="100%" height="225" src="https://github.com/frommwonderland/EManalysis/blob/main/resources/croped_gt_5_em_220.png">
</p>

<hr/>

<p align="justify">
One of the major challenges in neuroscience is to understand the functional and structural foundation that is underlying in the brain and which is ultimately leading to human understanding of intelligence. One approach to resolve this problem is the emerging field of <b>connectomics</b> where neuroscience and artificial intelligence intertwine in order to analyze neuronal connections. A connectome is a complete map of a neuronal system, comprising all neuronal connections between its structures. The term connectome implies completeness of all neuronal connections, in the same way as a genome is a complete listing of all nucleotide sequences. The goal of connectomics is to create a complete representation of the brain’s wiring. The important role of network architecture as a structural substrate for the functioning of the brain constitutes the main reason for the field of connectomics.

<p align="justify">
Based on electronic microscopy (EM) data the goal of this repository is enabling analysis on human brain tissue, particularly on mitochondria. Besides having algorithms that enable dense segmentation and alignment, the need for classification and clustering is key. Therefore, this software enables to cluster mitochondria (or any segments you want to cluster) automatically without relying on powerful computational resources.
</p>


## Installation
Create a new conda environment:
```
conda create -n py3_torch python=3.8
source activate py3_torch
```

Download and install the package:
```
git clone https://github.com/frommwonderland/EManalysis
cd EManalysis
pip install --upgrade pip
pip install -r requirements.txt
```

## Environment

The code is developed and tested under the following configurations.

- Hardware: 1-8 Nvidia GPUs with at least 12G GPU memory (change ```SYSTEM.NUM_GPU``` accordingly based on the configuration of your machine)
- Software: CentOS Linux 7.4 (Core), CUDA>=11.0, Python>=3.8, PyTorch>=1.8.0, YACS>=0.1.8

## Dataset
The framework relies on both EM data and its related groundtruth mask of the segments (e.g. mitochondria) you want to cluster. The output are images with relabeled segments.

## Structure
The structure of the project is the following:
Within the folder *analyzer* all the relevant code is stored for running the project, whereas *configs* holds various .yaml files for adjusting the relevant parameters and which specification or mode will be executed. The *datasets* folder stores  all relevant dataset files and further prepared datasets based on the original data (e.g. input volumes for the Autoencoder). Furthermore, there is a *features* storage where are computed features are stored for later usage. The *models* folder keeps all corresponding model saved. Additionally, within the *outputs* folder all labeled clustering results are stored.

## Usage
Running the software after the installation is simply put by
```
python main.py --cfg configs/process.yaml
```

This software uses different features for the clustering process and these are computed in separate ways. The main components are the Deep Learning Architectures for learning theses features. There are three different frameworks available: **Variational Autoencoder** ([VAE](https://github.com/AntixK/PyTorch-VAE)), **Autoencoder based on Point Clouds** ([PtC-Ae](https://arxiv.org/abs/1612.00593)) and **Constrastive Learning** ([CL](https://arxiv.org/abs/2002.05709)). The latent space is used respectively as representational features. In order to use these frameworks, some steps have to be performed beforehand: *Preprocessing*, *Training* \& *Inference*

After the computation of the learned features, the clustering process can be performed.

### Step 1: Preprocessing
Before training the various models, a preprocessing step has to be performed in order to adjust the data in a right way (consistent input size and format).
For perfoming **preprocessing**, altering the configuration file will do the job:
  ``` yaml
  MODE:
    PROCESS: 'preprocessing' || 'ptcprep'
  ```
- **'preprocessing'** will create a dataset which holds various 64 x 64 x 64 input volumes computed based on a combination of EM and label data.
- **'ptcprep'** will create a dataset which transforms every unique segment into a point cloud.

### Step 2: Training Process
After the preparation of the trainig data, **training** different frameworks is done by  altering the configuration file in the following way:
  ``` yaml
  MODE:
    PROCESS: 'train' || 'ptctrain' || 'cltrain'
  ```
- **'train'**: Starts the training process of the Variational Autoencoder
- **'ptctrain'**: Executes training an Autoencoder based on Point Clouds.
- **'cltrain'**: Performs training based on the Constrastive Learning setup.

### Step 3: Inference
Inference for computing the latent representation which will be used as features is done by adjusting the configuration file as:
  ``` yaml
  MODE:
    PROCESS: 'infer' || 'ptcinfer' || 'clinfer'
  ```
  - **'infer'**: Inferring the features learned by the Variational Autoencoder
  - **'ptcinfer'**: Inferring the features learned by an Autoencoder based on Point Clouds.
  - **'clinfer'**: Inferring the features learned by a Constrastive Learning setup.


### Hyperparameter Tuning
For all three frameworks different options can be adjusted which are shown below:
``` yaml
AUTOENCODER:
  ARCHITECTURE: 'unet_3d'
  TARGET: (64, 64, 64)
  EPOCHS: 5
  BATCH_SIZE: 2
  LATENT_SPACE: 100
  MAX_MEAN: 0.001
  MAX_VAR: 0.001
  MAX_GRADIENT: 1.0
  LARGE_OBJECT_SAMPLES: 2
PTC:
  BATCH_SIZE: 1
  EPOCHS: 25
  LR: 0.0001
  WEIGHT_DECAY: 0.0001
  LATENT_SPACE: 512
  RECON_NUM_POINTS: 5000
  SAMPLE_SIZE: 4096
  INPUT_DATA: 'datasets/mouseA/pts.h5'
SSL:
  BATCH_SIZE: 16
  EPOCHS: 25
  OPTIMIZER_LR: 0.05
  OPTIMIZER_WEIGHT_DECAY: 0.0005
  OPTIMIZER_MOMENTUM: 0.9
  K_KNN: 5
  USE_PREP_DATASET: 'datasets/mouseA/mito_samples.h5'
```
For more information, please check out the file */analyzer/config/config.py* where all default values are set. Please check the section structure and adjust the paths of where different information and savings are stored, so the program is able to find certain files.

### Main Step: Clustering stage
For running the clustering, apply
``` yaml
MODE:
  PROCESS: ''
```
Furthermore, you have to can set the algorithm and the features, you want to use. You find all the algorithms that are usable in the .yaml example below, the default is *'kmeans'*.

You find also all the possible features. Please fill the list with all features, that should be used for clustering.
- [**'sizef'**, **'distf'**, **'circf'**, **'slenf'**]: These are all features computed with traditional Computer Vision algorithms. They are extracted on the fly by the program.
- [**'shapef'**, **'ptcf'**, **'clf'**]: Extracted by VAE, PTCAE & CL accordingly. Please note that these features have to be present in *features/shapef.h5*

It is also possible to weight the features by applying a weighted term to the features, please adapt this list accordingly. N_CLUSTER allows to adjust the number of clusters, that should be found. By GENERATE_MASKS you can tell the program is output labels (images) should be produced.
``` yaml
CLUSTER:
  ALG: 'kmeans' || 'affprop' || 'specCl' || 'aggloCl' || 'dbscan' || 'hdbscan'
  FEAT_LIST: ['sizef', 'distf', 'circf', 'slenf', 'shapef', 'ptcf', 'clf']
  WEIGHTSF: [1, 1, 1, 1, 1]
  N_CLUSTER: 5
  GENERATE_MASKS: True
```

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/frommwonderland/EManalysis/blob/main/LICENSE) file for details.
