## Improving Camera Pose Estimation via the Use of Deep Generative Models

Final project in Deep Generative Models course.

In this study, we assessed the efficacy of deep generative models in producing novel samples for camera pose localization. 
By utilizing accurately positioned captured images, we aimed to train a generative model to create new viewpoints within the given scene, 
ultimately enhancing the performance of the camera pose localization model. 
Through our assessment of various architectures, we have ascertained that the VQ-VAE architecture exhibits superior performance
 in reconstructing input images compared to other architectures. 
 However, when presented with additional input containing the relative pose between input and target images, the model was unable to effectively reconstruct 
 the target image, instead generating an image of poor quality. 
 We hypothesize that the restricted nature of the provided relative pose information renders it insufficient for the task of synthesizing novel scenes.
 As a potential avenue for future research, more powerful deep generative models, such as latent diffusion-based models, could be explored as they possess stronger generative capabilities.

### Repository Overview 

This code implements:

1. Training of a multiple architectures for novel view relative pose generation
2. Testing code

### Prerequisites

In order to run this repository you will need:

1. Python3 (tested with Python 3.7.7, 3.8.5), PyTorch
2. Set up dependencies with ```pip install -r requirements.txt```
3. Download the [Cambridge Landmarks](http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset) dataset and the [7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) dataset

### Usage
Training 
```
python train_vae.py --bPoseCondition=0
```
Testing
```
python train_vae.py --mode=test --checkpoint_path=out_vae/vae_model.pth --bPoseCondition=0
```