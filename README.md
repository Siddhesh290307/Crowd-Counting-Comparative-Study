# Crowd Counting Experiments on Shanghai TechA

This repository contains three crowd‑counting experiments using the Shanghai TechA dataset.  
Each experiment explores a different architectural strategy:  

1. **CSRNet with an MLP bridge** – a density‑map regression model that adds a channel‑wise MLP between the frontend and backend.  
2. **Patch‑based classification + regression** – a two‑stage approach that first classifies patches into density levels, then applies specialised counters.  
3. **Router‑based hybrid density estimation** – a ResNet‑18 router selects between a CNN and a ViT counter (with soft fusion for uncertain patches).  

All models are implemented in PyTorch and evaluated using Mean Absolute Error (MAE).

---

## Dataset: Shanghai TechA

- **Part A** of the ShanghaiTech dataset contains 482 images with highly varying crowd densities.  
- We use the official training/test split (300 training, 182 testing).  
- Ground‑truth density maps are generated using Gaussian kernels (σ adapted to head sizes).

---

## Experiment 1: CSRNet with MLP Bridge

### Architecture

We modify the standard CSRNet by inserting a small **MLP bridge** (implemented as 1×1 convolutions) between the frontend and the backend.

```python
class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Frontend: VGG16 first 10 conv layers (pretrained, unfrozen)
        vgg = models.vgg16(weights='IMAGENET1K_V1')
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])

        # MLP Bridge: 1x1 convs = channel-wise MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(512, 512, 1), nn.ReLU(),
            nn.Conv2d(512, 512, 1), nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(512, 512, 1), nn.ReLU(),
        )

        # Backend: Dilated convolutions
        self.backend = nn.Sequential(
            nn.Conv2d(512, 256, 3, dilation=2, padding=2), nn.ReLU(),
            nn.Conv2d(256, 128, 3, dilation=2, padding=2), nn.ReLU(),
            nn.Conv2d(128, 64,  3, dilation=2, padding=2), nn.ReLU(),
            nn.Conv2d(64,  1,   1)
        )
    ...
```

### Why Add an MLP?

The MLP (applied channel‑wise) acts as a learnable feature transformation that refines the features extracted by VGG before the dilated backend.

- It introduces non‑linearity and feature re‑weighting, allowing the network to better capture fine‑grained patterns.
- The added dropout helps regularisation, preventing overfitting on the relatively small training set.

### Results

| Metric | Value |
|--------|-------|
| MAE | ~130 |
| Parameters | 16.9M |

#### Why is the MAE relatively low?

- Pretrained VGG features provide a strong starting point.
- The MLP bridge increases model capacity without dramatically increasing parameters.
- Dilated convolutions preserve spatial resolution, which is crucial for density estimation.

---

## Experiment 2: Patch‑Based Classification + Regression

This experiment splits each image into 128×128 patches and processes them in two stages.

### 1. Patch Classification

A lightweight CNN classifies each patch into one of three density classes:

- **Empty** (count < 3)
- **Low density** (3 ≤ count < 30)
- **High density** (count ≥ 30)

```python
class PatchClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64,3)
    ...
```
**Accuracy: 78.5%** on patch classification.

## Patch Routing Visualization

![Patch Routing Map](assets/patch_routing_map.png)

### 2. Density Regression

Separate counters are trained for low‑density and high‑density patches:

- **CNNCounter** – a small CNN that outputs a scalar count (used for low‑density patches).
- **ViTCounter** – a simple ViT‑like model that flattens the patch and uses MLPs (used for high‑density patches).
- **Empty patches** are assigned a count of 0.

### Results

| Stage | Metric | Value |
|-------|--------|-------|
| Patch classification | Accuracy | 78.49% |
| Full‑image regression | MAE | 204.21 |

The relatively high MAE suggests that the coarse classification and simple counters struggle to capture the continuous nature of crowd density.

---

## Experiment 3: Router‑Based Hybrid Density Estimation

To overcome the limitations of the patch‑based approach, we design a router‑guided hybrid model that dynamically selects the appropriate counter per patch.

### Architecture

**Router:** A ResNet‑18 trained to classify patches into the same three density classes (empty / low / high).
- **Accuracy: 99.38%** (much higher than the patch classifier in Exp. 2).

**Counters:**
- **SimpleCNN** – a lightweight CNN that outputs a density map (used for low‑density patches).
- **FastViTDensity** – a ViT‑tiny that outputs a density map (used for high‑density patches).
  - This ViT is trained with a low learning rate (1e‑6) and gradient clipping to prevent exploding gradients.

**Fusion:** For patches where the router has low confidence (<0.75), we blend the outputs of both counters using learnable weights.

```python
class HybridDensity(nn.Module):
    def forward(self, x, confidence_threshold=0.75):
        # Router predictions
        with torch.no_grad():
            class_logits = self.router(x)
            confs = F.softmax(class_logits, dim=1)
            max_confs, classes = confs.max(dim=1)

        # Compute both counters (always, for gradient flow)
        cnn_out = self.cnn(x)
        vit_out = self.vit(x)

        # Routing logic...
        # (empty, use_cnn, use_vit, or fusion)
```

### Results

| Model | MAE (full image) |
|-------|-----------------|
| CNN only | 184.87 |
| ViT only | 74.77 |
| Hybrid | 91.23 |

## Hybrid Model Training MAE

![Hybrid Model Training MAE](assets/training_mae.png)

#### Observations

- The ViT alone achieves the lowest MAE (74.77), indicating its superior ability to model global context.
- The hybrid performs worse than the ViT alone – likely because the router sometimes selects the weaker CNN for patches that could be better handled by the ViT, or because the fusion mechanism is not optimal.
- Nevertheless, the hybrid model demonstrates the potential of using a router to combine specialised experts.

---

## Summary of Results

| Experiment | Model / Approach | MAE |
|------------|-----------------|-----|
| 1 | CSRNet + MLP | ~130 |
| 2 | Patch‑based classification + regression | 204.21 |
| 3 | Router‑based hybrid | 91.23 (hybrid) / 74.77 (ViT only) |

### Key Takeaways

- Adding an MLP bridge to CSRNet improves feature refinement and yields a reasonable MAE (130).
- A simple patch‑based classification + regression approach is insufficient for this challenging dataset.
- A well‑trained ViT can achieve strong performance (74.77 MAE) on Shanghai TechA.
- Router‑guided mixture of experts is promising, but careful tuning of the router threshold and fusion weights is necessary to fully exploit the benefits of both counters.

---

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- timm (for ViT models)
- tqdm

Install dependencies:

```bash
pip install torch torchvision timm tqdm
```

---

## Usage

Detailed training and evaluation scripts are provided in the accompanying notebooks.
To reproduce the results:

1. Download the Shanghai TechA dataset and place it in `data/ShanghaiTech/part_A/`.
2. Run the notebooks (or scripts) for each experiment.
3. Checkpoints and logs will be saved automatically.

---

## Acknowledgements

This work was conducted as part of a crowd‑counting study using the Shanghai TechA dataset.
We thank the authors of CSRNet, ViT, and ResNet for their open‑source implementations.