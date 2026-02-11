# Deepfake-Detection
This repository is the official implementation of our paper "Towards Multi-Modal Forgery Representation Learning for AI-Generated Video Detection and Localization".
<img width="1945" height="857" alt="main" src="https://github.com/user-attachments/assets/ac2a9b19-1bd7-4309-a2cd-07f9e67d3d3a" />

## 1. Setup
### 1. Visual
See [visual setup](./visual/README.md#1-environment-setup)
### 2. Audio
See [audio setup](./audio/README.md#1-environment-setup)

## 2. Dataset Preparation
### 1. AV-Deepfake1M++
Follow the instruction from https://huggingface.co/datasets/ControlNet/AV-Deepfake1M-PlusPlus
### 2. FakeAVCeleb
Follow the instruction from https://github.com/DASH-Lab/FakeAVCeleb

1. Download both the dataset into the `./visual/dataset/` folder.
2. The final file tree should look like this:
```
.visual/dataset
|_ AV-Deepfake1M-PlusPlus
|   |_ train
|   |   |_ train
|   |       |_ lrs3
|   |       |_ silent_videos
|   |       |_ vox_celeb_2
|   |_ val
|   |   |_ val
|   |       |_ lrs3
|   |       |_ silent_videos
|   |       |_ vox_celeb_2
|   |_ train_metadata.json
|   |_ val_metadata.json
|
|_ FakeAVCeleb
    |_ FakeVideo-FakeAudio
    |_ FakeVideo-RealAudio
    |_ RealVideo-FakeAudio
    |_ RealVideo-RealAudio
    |_ meta_data.csv
```
**Note**: See [audio's dataset preparation](./audio/README.md#3-dataset-preparation) for additional steps.
## 3. Checkpoints
https://huggingface.co/anonymous-researcher-3407/Deepfake-Detection/
## 4. Training and Inference
### 1. Visual
See [visual](./visual/README.md#4-train)
### 2. Audio
See [audio](./audio/README.md#4-training-and-inference)
### 3. Fusion
See [fusion](./visual/README.md#4-train)
