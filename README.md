# UI Element Detector

Semester project: detect UI elements in Android app screenshots (RICO) and compare six
CNN techniques. This file is updated after every completed step.

**Status:** steps 0–5 done (code). Models not trained yet — training runs on Google Colab.

---

## Pipeline

```
Preprocessing  ->  Split 70:20:10  ->  Training  ->  Testing  ->  Evaluation
```

Two stages:

1. **Detection** — YOLO26n finds element boxes on the full screenshot.
2. **Classification** — five CNNs classify each detected crop into one of 15 classes.

15 classes: Layout, Text, Image, Button, Toolbar, ProgressBar, List, WebView, Tab,
Drawer, Ad, Card, ScrollView, Checkbox, ViewPager.

No image generation. Screenshots are never modified; crops are made on the fly.

---

## Repository layout

```
├── README.md                 this file
├── AGENTS.md                 working rules for the coding agent
├── requirements.txt
├── configs/                  one yaml per model (all training hyper-parameters)
├── data/
│   ├── raw/rico/             RICO screenshots + JSON (not in git)
│   └── yolo/                 labels, split, class_index, data.yaml (not in git)
├── scripts/
│   ├── class_mapping/        raw class -> 15 classes
│   ├── preprocessing/        JSON -> YOLO boxes
│   ├── split/                70:20:10 split
│   ├── models/               5 CNN definitions + registry
│   ├── training/             crop dataset, train_cnn, train_yolo
│   ├── testing/              (step 6)
│   └── evaluation/           (step 7)
├── notebooks/colab_train.ipynb
└── results/                  per-step reports, checkpoints, metrics
```

---

## Step by step

### Step 0 — Setup (done)

Python 3.14 virtualenv with `requirements.txt` (torch 2.14.1+cpu, ultralytics 8.4.184,
timm, scikit-learn, opencv, streamlit). Training happens on Colab because the local
GPU is a GTX 960M with 2 GB and the venv is CPU-only.

RICO extracted to `data/raw/rico/unique_uis/combined/`: **66,261 screenshots + 66,261
JSON view hierarchies**.

### Step 1 — Class mapping (done)

Scanned all 66,261 JSON files: **40,014 distinct raw Android class names, 6,371,389
view nodes**.

`scripts/class_mapping/class_map.py` maps raw classes to the 15 classes. Rules are
applied to the simple class name, specific classes first:

* `EditText` before `Text`, `TabLayout` before `Layout`, `ImageButton` before `Image`
  — a generic rule never steals a specific class.
* Dropped: structural (`View`, `ViewStub`, `ViewStubCompat`, `DecorView`), `Other`,
  and the low-count classes Input, SeekBar, Dialog, VideoView, Spinner.
* `ImageButton`, `RadioButton`, `Switch` -> Button. Ad web views -> Ad.

`test_map.py` covers 54 tricky cases (`TableLayout` must be Layout, not Tab;
`ViewSwitcher` must not become Button; ...). All pass.

Outputs: `results/class_mapping/raw_class_counts.csv`, `mapped_class_counts.csv`.

### Step 2 — Preprocessing to YOLO (done)

`scripts/preprocessing/build_yolo_labels.py` reads each JSON, maps classes, checks
boxes and writes one YOLO label file per screenshot.

* Boxes sticking out of the screenshot are **clipped**, not dropped.
* Boxes with reversed coordinates (`x2 < x1`) or zero area are **dropped** — about
  half of all RICO nodes, which is why box counts are roughly half the node counts.
* Every box with positive area is kept (no minimum size).
* Screenshots with no surviving box are skipped.

Result: **66,041 label files, 1,915,620 boxes (29.0 per image)**, 220 images dropped.

| class | boxes | share |
|---|---|---|
| Layout | 993,162 | 51.85% |
| Text | 412,781 | 21.55% |
| Image | 213,601 | 11.15% |
| Button | 78,649 | 4.11% |
| Toolbar | 61,263 | 3.20% |
| List | 36,054 | 1.88% |
| ScrollView | 19,996 | 1.04% |
| Drawer | 19,522 | 1.02% |
| Tab | 15,467 | 0.81% |
| ViewPager | 15,442 | 0.81% |
| ProgressBar | 13,494 | 0.70% |
| Card | 12,910 | 0.67% |
| Ad | 11,660 | 0.61% |
| WebView | 7,718 | 0.40% |
| Checkbox | 3,901 | 0.20% |

Imbalance is **255x** (Layout vs Checkbox), handled with class-weighted loss.

Verified: 0 malformed lines, all coordinates inside [0,1], and an overlay drawn on a
real screenshot lines up with the drawer, cards and ad banner.

### Step 3 — Split 70:20:10 (done)

`scripts/split/make_split.py`, seed 42, **by image** (all boxes of an image stay
together):

| split | images | share | boxes |
|---|---|---|---|
| train | 46,228 | 70% | 1,339,701 |
| test | 13,208 | 20% | 383,955 |
| val | 6,605 | 10% | 191,964 |

Screenshots are **hard-linked** into `data/yolo/images/` (no extra disk). Every class
keeps the same proportion in every split.

Per-class folders: `data/yolo/class_index/<split>/<ClassName>/images.txt` lists the
images containing that class (15 folders × 3 splits, no duplicated images).

`data.yaml` loads through Ultralytics `check_det_dataset`.

### Step 4 — Models (done)

| model | params | weights |
|---|---|---|
| DenseNet121 | 6.97 M | ImageNet (timm `densenet121.ra_in1k`) |
| Xception41 | 24.95 M | ImageNet (timm `xception41.tf_in1k`) |
| SE-ResNet50 | 26.07 M | ImageNet (timm `legacy_seresnet50.in1k`) |
| PyramidNet-110 | 1.77 M | trained from scratch |
| Residual Attention Network | 21.75 M | trained from scratch |
| YOLO26n (detector) | 2.57 M | COCO |

PyramidNet and Residual Attention Network are reimplemented from their papers because
no pretrained weights exist. Both are trained from scratch, so lower scores are
expected and will be reported as they are.

### Step 5 — Training pipeline (code done, training not run yet)

* `scripts/training/crops.py` — crops boxes from the screenshots on the fly; nothing
  is written to disk. Train crops are augmented, val crops only resized.
* `scripts/training/train_cnn.py` — AdamW + cosine schedule, label smoothing, weight
  decay, **class-weighted cross-entropy** (inverse frequency), early stopping on
  validation macro-F1, best checkpoint saved. Writes `best.pt`, `last.pt`,
  `metrics.csv`, `history.json`, `val_per_class.json` into `results/<model>/`.
* `scripts/training/train_yolo.py` — Ultralytics wrapper with the same seed and
  protocol, saves into `results/<model>/`.
* `configs/*.yaml` — every hyper-parameter per model.

Smoke runs (CPU, 1 epoch, tiny subset) passed for all six models; they only prove the
code runs. RAN needed a fix for odd feature-map sizes, and PyramidNet was initially
2.3 B parameters because of a wrong channel-growth rule (paper: width grows from 16 to
16+alpha across the network).

Known limitation: Ultralytics has no per-class weights (only a scalar `cls_pw`), so
class weighting applies to the classifier stage.

---

## How to run

**Local smoke test** (CPU, no weights download):

```bash
python scripts/models/check_models.py --no-pretrained
python scripts/training/train_cnn.py --model densenet121 --epochs 1 --max-crops 200 \
    --val-per-class 20 --batch-size 16 --no-pretrained --out-suffix _smoke
```

**Real training (Colab, T4):**

```bash
python scripts/run_all_colab.py --data-zip /content/drive/MyDrive/ui-data.zip --zip-out /content/drive/MyDrive/ui-results.zip
```

or open `notebooks/colab_train.ipynb` and run the cells in order (it does the same thing
step by step). Useful flags: `--models densenet121 xception41` (subset), `--skip-smoke`,
`--no-yolo`, `--smoke-only`, `--epochs`, `--batch-size`.

**Rebuild everything from raw RICO:**

```bash
python scripts/class_mapping/scan_raw_classes.py
python scripts/class_mapping/count_mapped_classes.py
python scripts/preprocessing/build_yolo_labels.py
python scripts/split/make_split.py
```

---

## Results

Not available yet — models are trained on Colab in the next step. Results will be
added here as they arrive, per class, with no cherry-picking.
