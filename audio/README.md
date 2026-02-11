# Audio Deepfake Detection (Audio Branch)

This repository contains the implementation for the audio branch of our multi-modal deepfake detection framework. It is built upon the [PartialSpoof](https://github.com/nii-yamagishilab/PartialSpoof) framework.

**License Note:** This project utilizes code from PartialSpoof and s3prl. Please refer to the `LICENSE` files in each submodule/directory for specific licensing details.

## 1. Environment Setup

The SSL model used in this project is based on [s3prl](https://github.com/s3prl/s3prl).

### Prerequisites
*   Python >= 3.6
*   `sox` (Install via your OS package manager, e.g., `sudo apt install sox` or `conda install -c conda-forge sox`)

### Installation

1.  **Install s3prl** (Included in `modules/s3prl`):
    ```bash
    pip install -e modules/s3prl
    ```

2.  **Install Fairseq**:
    ```bash
    pip install fairseq@git+https://github.com/pytorch/fairseq.git@f2146bdc7abf293186de9449bfa2272775e39e1d#egg=fairseq
    ```

3.  **Dependencies & Troubleshooting**:
    You may encounter version conflicts with `omegaconf`, `numpy`, or `fairseq` as some dependencies are older. We have provided a full environment export in `environment.yml` for reference.

    If you run into issues, please check `environment.yml` to match specific package versions.

## 2. Prepare Checkpoints

You will need two checkpoints for the model.

1.  **Wav2Vec 2.0 Checkpoint**:
    *   Download: [huggingface](https://huggingface.co/anonymous-researcher-3407/Deepfake-Detection/blob/main/w2v_large_lv_fsh_swbd_cv.pt)
    *   Place it in: `modules/ssl_pretrain/`
    *   Expected filename: `w2v_large_lv_fsh_swbd_cv.pt`

2.  **gMLP / Trained Model Checkpoint**:
    *   Download: [huggingface](https://huggingface.co/anonymous-researcher-3407/Deepfake-Detection/blob/main/trained_network.pt)
    *   Place it in: `03multireso/multi-reso/01/`

## 3. Data Preparation

To train or infer, the system requires three components:
1.  **Audio files** (`.wav`)
2.  **Filename list** (`.lst`)
3.  **Groundtruth labels** (`.npy` containing 10ms segment labels)

### Using `prepare_data.py`
We provide a utility script to convert AV-Deepfake1M++ metadata into the required format.

```bash
python database/prepare_data.py \
    --json_path /path/to/metadata.json \
    --video_folder /path/to/videos \
    --output_dir database/ \
    --dataset_name dev
```

**Label Definition:**
*   **1**: Real / Bonafide
*   **0**: Fake / Spoof

The script processes the video files, extracts audio, handles chunking for long files, and generates the corresponding 10ms segment labels stored in an `.npy` file.

## 4. Training and Inference
Setup paths
```bash
./env.sh
```

Navigate to the experiment directory:
```bash
cd 03multireso/multi-reso/01
```

### Training
To train the model:
```bash
bash 00_run.sh 1
```

### Inference
To run inference on the test/dev set:
```bash
bash 00_run.sh 2
```

## 5. Results & Next Steps

The inference process will output **6 `.pkl` files**. Each file represents the segment-level predictions for a specific temporal resolution.

These outputs are designed to be integrated with the Visual Branch (Spatio-Temporal branch and LMM branch) for the final Multi-Modal Fusion stage.
