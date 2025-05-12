# GMT: Graph Matching Transformer

**GMT** (Graph Matching Transformer) is a PyTorch-based framework for matching and aligning 2D curves (graphs) using rich geometric embeddings and a cross-attention Transformer architecture. It supports four model variants—`tiny`, `small`, `medium`, and `large`—to scale computational complexity and capacity.

---

## Key Features

- **Multi-Geometry Support**: Generates and processes sinusoids, circles, ellipses, and random polylines.
- **Curvature & Ray Embeddings**: Computes curvature, ray distances, incidence angles, and hit flags for each point.
- **Index & Initial Shift Embedding**: Includes normalized index, curvature, and initial displacement as features.
- **Cross-Attention Transformer**: Two-stream self-attention on target & baseline, followed by cross-attention for fine-grained alignment.
- **Variants**: Four predefined configurations (`tiny`, `small`, `medium`, `large`) with adjustable `d_model`, depth, and feed-forward dimensions.
- **Metal/CUDA/CPU**: Auto-selects MPS (Apple Silicon), CUDA, or CPU device.
- **Visualizations**: Built-in training loss curves, inference progression plots, and error distribution histograms.

---

## Repository Structure

```text
weights/                 # Weights folder
README.md
train.py                 # Entry-point for training all variants
infer.py                 # CLI for inference and mapping extraction
gmt/                     # Core package
  __init__.py
  variants.py            # Model configurations
  utils.py               # Geometry & resampling utilities
  embeddings.py          # Ray-segment embedding functions
  dataset.py             # ThreadedRayDataset & helpers
  model.py               # Transformer definitions
  trainer.py             # Training loop and checkpointing
experiment.ipynb         # Jupyter notebook demo
LICENSE
requirements.txt         # Python dependencies
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/raildart/gmt.git
cd gmt

# (Optional) Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Training All Variants

```bash
python train.py \
  --epochs 30 \
  --batch_size 64 \
  --lr 5e-5
```

This will train `tiny`, `small`, `medium`, and `large` sequentially and save checkpoints as `GMT_<variant>.pth`.

### Running Inference with External Geometries

```bash
python infer.py \
  --variant medium \
  --external path/to/geoms.npz \
  --samples 5 \
  --batch_size 16 \
  --save
```

This loads your own `.npz` with `baseline` and `target` arrays, runs the model, plots 5 sample alignments, and saves `mappings_medium.npz`.

---

## Model Variants & Performance

Below is a summary of each variant’s architecture along with its final test MSE (mean squared error). Replace the placeholder MSE values with your actual results.

| Variant | d_model | Layers | FF Dim | Dropout | Test MSE |
| ------- | ------: | -----: | -----: | ------: | -------: |
| tiny    |     128 |      2 |    256 |    0.10 |   0.0034 |
| small   |     256 |      3 |    512 |    0.15 |   0.0028 |
| medium  |     512 |      4 |   1024 |    0.20 |        X |
| large   |     768 |      5 |   1536 |    0.20 |        X |

### Mean Squared Error (MSE)

The **Mean Squared Error (MSE)** is our primary training and evaluation metric. For a single predicted sequence $\hat{\mathbf{y}} = [\hat{y}_1, \hat{y}_2, \dots, \hat{y}_N]$ and its ground-truth sequence $\mathbf{y} = [y_1, y_2, \dots, y_N]$, the MSE is computed as:

$$
\mathrm{MSE}(\mathbf{y}, \hat{\mathbf{y}}) \;=\; \frac{1}{N} \sum_{i=1}^{N} \bigl(y_i - \hat{y}_i\bigr)^{2}.
$$

In our setting, each sequence consists of 2-D displacements for $N$ resampled points, so we actually average over both dimensions:

$$
\mathrm{MSE} = \frac{1}{N}\sum_{i=1}^{N}\Bigl[(\Delta x_i - \widehat{\Delta x}_i)^2 + (\Delta y_i - \widehat{\Delta y}_i)^2\Bigr].
$$

During training, we report the **batch-averaged** MSE each epoch, and at the end we compute the **dataset-wide** MSE by averaging over all samples. Lower MSE indicates that the model’s predicted alignment shifts more closely match the true geometric offsets.

---

## API Usage

```python
from gmt.dataset import ThreadedRayDataset
from gmt.model import ComplexCrossTransformer
from gmt.trainer import train
from gmt.variants import define_variants

# Create dataset
ds = ThreadedRayDataset(num_samples=5000, max_workers=8)
feat_dim = ds.tgt_feats.shape[-1]

# Choose a variant
variant = 'medium'
model = ComplexCrossTransformer(tgt_dim=feat_dim, base_dim=3, variant=variant)

# Train
dtrained_model = train(ds, model, variant=variant, epochs=20, batch_size=64, lr=5e-5)
```

---

## License

This project is licensed under the [MIT License](LICENSE).
