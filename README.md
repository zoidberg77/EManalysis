<h1 align="center">
  <b>EManalysis</b><br>
</h1>

The field of Connectomics aims to reconstruct the wiring diagram of the brain by mapping the neural connections at a cellular level. Based on electronic microscopy (EM) data the goal of this repository is enabling analysis on human brain tissue, particularly on mitochondria. Besides having algorithms that enable dense segmentation and alignment, the need for classification and clustering is key. Therefore, this software enables to cluster mitochondria (or any segments you want to cluster) automatically without relying on powerful computational resources.

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

## Dataset
The framework relies on both EM data and its related groundtruth mask of the segments (e.g. mitochondria) you want to cluster. The output are images with relabeled segments.

## Usage
Running the software after the installation is simply put by
```
python main.py --cfg configs/process.yaml
```
but there are a few points to consider. This software uses different features for the clustering process and these are computed in separate ways. The main component of the framework is the Variational Autoencoder ([VAE](https://github.com/AntixK/PyTorch-VAE)) where the latent space is used as representational features. In order to use the VAE, it is has to be trained beforehand. So the framework is consisting as two separate parts, where one is the training phase of the VAE and the other is the actual clustering. You have to make sure to tell the program which part you want to run.

### Part 1: Training the Variational Autoencoder
Before training the VAE, you have preprocess the data. This has two main reasons: Consistent input size & individual training samples.

Preprocessing is done by either ...
- altering the configuration file
``` yaml
MODE:
  PROCESS: 'preprocessing'
```
- from the command line by adding ``` --mode preprocessing ```

Training the VAE is done by either ...
- altering the configuration file
``` yaml
MODE:
  PROCESS: 'train'
```
- from the command line by adding ``` --mode train```

Hyperparameter Tuning
``` yaml
AUTOENCODER:
  ARCHITECTURE: 'unet_3d'
  REGION_LIMIT: None
  CHUNKS_CPU: 4
  UPPER_BOUND: 100000
  LOWER_BOUND: 100
  TARGET: (128, 128, 128)
  EPOCHS: 5
  BATCH_SIZE: 2
  OUTPUT_FOLDER: 'features/'
  FEATURES: ['shape', 'texture']
  LATENT_SPACE: 100
  LOG_INTERVAL: 10
```

### Part 2: Clustering stage


## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/frommwonderland/EManalysis/blob/main/LICENSE) file for details.
