# 🧠 Brain MRI Segmentation — U-Net + Streamlit

FLAIR abnormality segmentation on the LGG Brain MRI dataset using a PyTorch U-Net, served via a Streamlit web app.

---

## Project Structure

```
brain_mri_app/
├── app.py            ← Streamlit UI
├── model.py          ← U-Net architecture + losses + metrics
├── dataset.py        ← LGG dataset loader with augmentation
├── train.py          ← Training script (CLI)
├── requirements.txt
└── checkpoints/      ← Saved model weights (created on first run)
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Download the LGG dataset from Kaggle:
```
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
```
Extract it so you have a `kaggle_3m/` folder with 110 patient sub-directories.

### 3. Train the model
```bash
python train.py \
  --data  ./kaggle_3m \
  --epochs 50 \
  --batch  16 \
  --img_size 256
```

Training flags:
| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `./kaggle_3m` | Path to dataset root |
| `--epochs` | 50 | Number of training epochs |
| `--batch` | 16 | Batch size |
| `--lr` | 1e-4 | Initial learning rate |
| `--img_size` | 256 | Input resolution (square) |
| `--bilinear` | False | Use bilinear upsampling in decoder |
| `--bce_weight` | 0.5 | Balance BCE vs Dice loss |
| `--resume` | None | Path to checkpoint to resume from |

Best checkpoint → `checkpoints/best_model.pt`  
Training log  → `checkpoints/train_log.csv`

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

Open `[http://localhost:8501](https://brain-mri-segmentation-satzuwvqxo53qxgkhyx52o.streamlit.app/)` in your browser.

> **Note:** The app ships with a randomly-initialised model for demo purposes.  
> Replace the `load_model()` function in `app.py` to load your trained weights:
> ```python
> @st.cache_resource
> def load_model():
>     model = UNet(in_channels=3, out_channels=1)
>     ckpt  = torch.load("checkpoints/best_model.pt", map_location="cpu")
>     model.load_state_dict(ckpt["model"])
>     model.eval()
>     return model
> ```

---

## Model

**U-Net** (Ronneberger et al., 2015)

| Component | Details |
|-----------|---------|
| Encoder | 4 × DoubleConv + MaxPool (64 → 512 channels) |
| Bottleneck | DoubleConv 1024 channels |
| Decoder | 4 × Upsample + skip concat + DoubleConv |
| Output | 1×1 Conv → Sigmoid |
| Loss | 0.5 × BCE + 0.5 × Dice |
| Optimiser | Adam (lr=1e-4, wd=1e-5) |
| Scheduler | CosineAnnealingLR |

**Expected metrics after 50 epochs (256×256):**
- Val Dice ≈ 0.88–0.92
- Val IoU  ≈ 0.80–0.86

---

## Dataset

- **Source:** The Cancer Imaging Archive (TCIA) via Kaggle
- **Patients:** 110
- **Modality:** FLAIR MRI (3 RGB channels combined)
- **Annotations:** Manual binary tumour masks
- **License:** CC BY-NC-SA 4.0

---

## References

1. Buda et al. (2019). *Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm.* Computers in Biology and Medicine.
2. Mazurowski et al. (2017). *Radiogenomics of lower-grade glioma.* Journal of Neuro-Oncology.
3. Ronneberger et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
