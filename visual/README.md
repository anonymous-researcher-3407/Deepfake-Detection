# Deepfake Detection (Visual Branch)

This repository contains the implementation for the visual branch of our multi-modal deepfake detection framework. It is built upon the [MM-Det](https://github.com/SparkleXFantasy/MM-Det) framework.

## 1. Environment Setup
```
conda create -n visual python=3.10
conda activate 
pip install -r requirements.txt
cd LLaVA
pip install -e ".[train]"
pip install flash-attn==2.5.8 --no-build-isolation
```
## 2. Checkpoints
```
./visual/weights
|_ fusion.ckpt
|_ visual.ckpt
|_ vit.pth
|_ vqvae.pt
```
- [fusion.ckpt](https://huggingface.co/anonymous-researcher-3407/Deepfake-Detection/resolve/main/fusion.ckpt)
- [visual.ckpt](https://huggingface.co/anonymous-researcher-3407/Deepfake-Detection/resolve/main/visual.ckpt)
- [vit.pth](https://huggingface.co/anonymous-researcher-3407/Deepfake-Detection/resolve/main/vit.pth)
- [vqvae.pt](https://huggingface.co/anonymous-researcher-3407/Deepfake-Detection/resolve/main/vqvae.pt)
## 3. Train and Inference
### 1. Visual
```
# 1. Extract frames from videos
python prepare_dataset.py --data-root ./dataset/(AV-Deepfake1M-PlusPlus or FakeAVCeleb) --dataset av1m

# 2. Reconstruct frames to amplify generative traces
python prepare_reconstructed_dataset.py

# 3. Expose multi-modal forgery via LMM
python prepare_mm_representations.py

# 4. Train
python train.py --ckpt-dir ./checkpoints
```
Parameters:
`--data-root`: ./data/AV-Deepfake1M-PlusPlus or ./data/FakeAVCeleb
`--dataset`: `av1m` or `fakeavceleb`

### 2. Fusion
```
# 1. Predict score from visual
python predict.py --ckpt-path ./checkpoints/trained_visual_checkpoint.ckpt

# 2. Predict score from audio

# 3. Train
python fusion_train.py --ckpt-dir ./checkpoints
```

### 3. Inference
```
# 1. Predict score from visual
python predict.py --ckpt-path ./checkpoints/trained_visual_checkpoint.ckpt

# 1. Predict final score
python fusion_predict.py --ckpt-path ./checkpoints/trained_fusion_checkpoint.ckpt
```

`--ckpt-path`: can be our provide pre-trained weights or your own trained checkpoint