# Duality Offroad Desert Segmentation (PyTorch)

Production-style semantic segmentation starter for the **Duality AI Offroad Desert Segmentation** challenge: **10 classes**, configurable **segmentation_models_pytorch** models (DeepLabV3+ or U-Net), ImageNet-pretrained encoders, Albumentations augmentations, mixed precision on CUDA, and full train / test / predict scripts.

## Class definitions

| ID | Class           |
|----|-----------------|
| 0  | Trees           |
| 1  | Lush Bushes     |
| 2  | Dry Grass       |
| 3  | Dry Bushes      |
| 4  | Ground Clutter  |
| 5  | Flowers         |
| 6  | Logs            |
| 7  | Rocks           |
| 8  | Landscape       |
| 9  | Sky             |

## Folder layout (project)

```
duality_segmentation/
├── train.py
├── test.py
├── predict.py
├── dataset.py
├── model.py
├── metrics.py
├── utils.py
├── visualize.py
├── config.yaml
├── requirements.txt
├── README.md
├── checkpoints/      # best_model.pth written here
├── outputs/          # predictions & overlays from test.py
└── runs/             # training_log.csv & curve plots
```

## Expected dataset layout

Place data next to the project (or set `dataset.root_dir` in `config.yaml`):

```
dataset/
├── Train/
│   ├── images/
│   └── masks/
├── Val/
│   ├── images/
│   └── masks/
└── testImages/
```

- **Training never uses `testImages/`** — only `Train/` and `Val/`.
- Images: `.png`, `.jpg`, `.jpeg`.
- Pairs are matched by **filename stem** (e.g. `scene42.jpg` ↔ `scene42.png`).

### Masks: grayscale or RGB

- **Grayscale**: if pixel values are already **0–9**, they are treated as class IDs. Unknown values become **255** (ignored in loss). You can set an explicit map in `config.yaml` under `mask_encoding.grayscale_value_to_class`.
- **RGB**: set `mask_encoding.rgb_palette` to a list of **10** `[R, G, B]` colors in class order (0 → 9). The loader matches **both RGB and BGR** so OpenCV-loaded masks align with your palette. **Update the palette** to match your dataset’s exact colors.

## Setup (macOS / Linux)

```bash
cd duality_segmentation
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** Install a PyTorch build that matches your machine ([pytorch.org](https://pytorch.org/get-started/locally/)). The `requirements.txt` pins minimum versions; CUDA wheels are usually installed separately.

## Train

```bash
python train.py --config config.yaml
```

**Before training**, the script checks that dataset directories exist, prints counts of train/val pairs and test images, and prints **unique values / colors** from the first training mask so you can confirm encoding.

**Outputs:**

- `checkpoints/best_model.pth` — best validation **mIoU**
- `runs/training_log.csv` — epoch, learning rate, train/val loss, val mIoU, pixel accuracy
- `runs/loss_curves.png`, `runs/miou_curve.png`

## Test (inference on `testImages/`)

```bash
python test.py --config config.yaml --checkpoint checkpoints/best_model.pth
```

**Outputs:**

- `outputs/predictions/<stem>_pred.png` — class index mask (0–9)
- `outputs/overlays/<stem>_overlay.png` — BGR overlay on the original image

## Predict (single file or folder)

```bash
python predict.py --checkpoint checkpoints/best_model.pth --input path/to/image_or_folder --output_dir outputs/predict_single --show
```

`--show` saves a small matplotlib panel per image (image / prediction / overlay).

## Configuration

Edit `config.yaml` for paths, model (`architecture`, `encoder`, `pretrained` / `encoder_weights`), image size, batch size, scheduler, augmentations, and mask color / grayscale maps.

## Metrics (implementation)

- **Pixel accuracy** (valid pixels only; ignore index 255 excluded)
- **Per-class IoU** and **mean IoU (mIoU)** from the confusion matrix

---

## Commands (copy-paste)

```bash
pip install -r requirements.txt
python train.py --config config.yaml
python test.py --config config.yaml --checkpoint checkpoints/best_model.pth
```

Run these from the `duality_segmentation` directory (with your virtual environment activated).
