# MeliVision

**Automated optical microscopy for pollen-grain classification using Deep Metric Learning (DML) with a DINOv2 Vision Transformer.**

**Affiliation:** FLAME³ — Fundamental of Light, Applied optics and Microscale engineering, Experiments and Explorations, Universidad de Concepción, Chile.

This repository accompanies a scientific manuscript prepared for *Scientific Reports* (Nature Portfolio). It provides the source code, configuration, and documentation required to reproduce the localisation, embedding, training, evaluation, and end-to-end inference pipelines described in the paper.

| Component | Role |
|-----------|------|
| **Localisation (production)** | Dense ViT (DINOv2) + configurable coarse-to-fine strategy + quality gates — full-image inference |
| **Localisation (preprocessing)** | U²-Net (SOD) + morphology — generation of `.seg` annotations |
| **DML** | AnalogyNet (trained ViT) learns 128D embeddings on S^127 and classifies by slice |
| **Camera inference** | `PollenViTDetector` localises → `FullImageClassifier` + best model classifies |

---

## Table of Contents

- [Scientific Context](#scientific-context)
- [Repository Purpose (Scientific Reports)](#repository-purpose-scientific-reports)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Full System Pipeline](#full-system-pipeline)
- [Stage 1 — Optical Acquisition](#stage-1--optical-acquisition)
- [Stage 2 — Grain Detection](#stage-2--grain-detection)
- [Stage 3 — .seg Format and On-The-Fly Dataset](#stage-3--seg-format-and-on-the-fly-dataset)
- [Stage 4 — Preprocessing and Augmentations](#stage-4--preprocessing-and-augmentations)
- [Stage 5 — Embedding Model (AnalogyNet + DINOv2)](#stage-5--embedding-model-analogynet--dinov2)
- [Stage 6 — Training (Sliced MS Loss)](#stage-6--training-sliced-ms-loss)
- [Declarative Configuration and Alignment Plan](#declarative-configuration-and-alignment-plan)
- [Stage 7 — Evaluation and Metrics](#stage-7--evaluation-and-metrics)
- [Stage 8 — Post-Training Analysis](#stage-8--post-training-analysis)
- [Stage 9 — End-to-End Inference](#stage-9--end-to-end-inference)
- [Stage 10 — Explainability (DINOv2 Attention)](#stage-10--explainability-dinov2-attention)
- [Experimental Results](#experimental-results)
- [Software Architecture](#software-architecture)
- [Data Availability](#data-availability)
- [Code Availability](#code-availability)
- [Citation](#citation)
- [References](#references)
- [Academic Documentation](#academic-documentation)

---

## Scientific Context

### Melissopalynology

**Melissopalynology** — the study of pollen content in honey — is essential for geographic and botanical origin certification. Manual analysis requires direct optical microscopy (>30 min/sample) with high operator dependence (Louveaux et al., 1978). This system automates the full process: detection, classification, and counting of pollen grains.

### Why Deep Metric Learning?

Unlike softmax classifiers with a fixed number C of output neurons:

| Property | Softmax Classifier | Deep Metric Learning |
|----------|--------------------|----------------------|
| New species | Full re-training | Add reference embeddings only |
| Scalability | O(C) final neurons | Extensible vector database |
| Few-shot | Many samples per class | Works with few references |
| Interpretability | Abstract logits | Distances in metric space |
| Generalisation | Seen classes only | Zero-shot to unseen classes |

### Dataset

The system supports multi-class pollen datasets. The recommended stratified partition is **70% train** · **15% val** · **15% test**.

---

## Repository Purpose (Scientific Reports)

This repository is structured so that readers, referees, and independent researchers can:

1. Inspect the exact software stack and declarative configuration (`config.yaml`).
2. Reproduce training, evaluation, and end-to-end inference for the paper-associated code version.
3. Access installation instructions, operating-system assumptions, dependencies, and demo entry points.
4. Locate **Data Availability** and **Code Availability** statements consistent with Nature Portfolio / *Scientific Reports* policies.

Custom code central to the conclusions of the manuscript is maintained under version control in this repository. Authors should mint a DOI for the paper-specific release (e.g. Zenodo / institutional archive) and cite that frozen version in the manuscript **Code Availability** statement.

---

## System Requirements

| Item | Specification |
|------|---------------|
| **Operating system** | Windows 10/11, Linux, or macOS |
| **Language** | Python ≥ 3.10 |
| **Deep learning** | PyTorch ≥ 2.0 (CUDA recommended for training) |
| **GPU (recommended)** | NVIDIA GPU with ≥ 8 GB VRAM for DINOv2 ViT-S/14 training |
| **CPU-only** | Supported for inference demos; training is substantially slower |
| **Disk** | Space for dataset images, `.seg` annotations, and `runs/` artefacts |

---

## Installation

```bash
git clone https://github.com/Nahzap/MeliVision.git
cd MeliVision
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

DINOv2 weights are downloaded via `torch.hub` on first use. U²-Net weights under `models/weights/` are fetched automatically when required.

Validate declarative configuration:

```bash
python -c "from src.utils.config_validator import validate_config; import yaml; validate_config(yaml.safe_load(open('config.yaml')))"
```

Optional smoke test (pooling / slice layout):

```bash
python scripts/smoke_test_pooling_slice.py --config test_config.yaml
```

---

## Quick Start

**GUI (PyQt5):**

```bash
python app.py
```

**Training (DML):**

```bash
python scripts/train.py --config config.yaml
```

**Post-training analysis (9 stages):**

```bash
python scripts/post_training.py --run runs/<run_id>
```

**End-to-end gallery regeneration / metrics:**

```bash
python scripts/regenerate_fullimage_folder.py
python scripts/analyze_gallery_folder.py
```

Expected outputs include annotated full-field images, per-species counts, Grad-CAM / attention heatmaps, and JSON/Markdown reports under `runs/`.

---

## Full System Pipeline

```
╔══════════════════════════════════════════════════════════════════╗
║                   FULL PIPELINE (10 STAGES)                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  ┌───────────────────┐                                            ║
║  │ MICROSCOPE IMAGE  │  Bright-field optics, CCD/CMOS sensor      ║
║  │ (H × W × 3)       │  Köhler illumination, 60× objective        ║
║  └─────────┬─────────┘                                            ║
║            │                                                      ║
║            ▼                                                      ║
║  STAGE 2: U²-Net (preprocessing only → .seg)                     ║
║  │  Image → U²-NETP (320×320) → Saliency                          ║
║  │  → Adaptive threshold → Morphology                              ║
║  │  → Contours → Filtering (area, circ.)                          ║
║  │  → N detections {bbox, contour, prob}                          ║
║  └─────────┬──────────────────────────────────────────────────────║
║            │                                                      ║
║            ▼                                                      ║
║  │ STAGE 3: .seg annotation (plain text)                          ║
║  │  ZERO extra images on disk                                     ║
║  │  Format: idx|cls|bbox|saliency|contour                         ║
║  └─────────┬──────────────────────────────────────────────────────║
║            │                                                      ║
║            ▼                                                      ║
║  │ STAGES 4–5: Crop + Mask + Augment + Embed                      ║
║  │  .seg + img → Square crop + padding                            ║
║  │  → Contour mask → Neutral grey bg (128)                        ║
║  │  → Albumentations (geom+colour+micro)                          ║
║  │  → ImageNet normalisation                                      ║
║  │                                                                 ║
║  │  DINOv2 ViT-S/14 (21M params)                                  ║
║  │    Layers 3+11 → masked multi-spec                             ║
║  │    (mean+max+std) → R^1152 × 2                                 ║
║  │  Projection LF/HF → 64+64 + L2-Cat                             ║
║  │  Embedding e-hat ∈ S^127 (128D)                                ║
║  └─────────┬──────────────────────────────────────────────────────║
║            │                                                      ║
║            ▼                                                      ║
║  │ STAGES 6–8: Sliced MS + MLRC evaluation + post-training        ║
║  │ STAGE 9: Inference — dense ViT coarse-fine → ViT classifies    ║
║  └─────────┬──────────────────────────────────────────────────────║
║            │                                                      ║
║            ▼                                                      ║
║  RESULT: {species → count} + annotated image + heatmaps           ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Stage 1 — Optical Acquisition

### Optical System

| Parameter | Specification |
|-----------|---------------|
| **Microscope** | Compound optical, bright field |
| **Objectives** | 60× (oil immersion) |
| **Illumination** | Köhler (uniform field) |
| **Sensor** | Digital CCD/CMOS, ≥ 1920×1080 |
| **Preparation** | Erdtman acetolysis or direct glycerine mount |

### Diagnostic Pollen-Grain Structures

| Structure | Description | Taxonomic relevance |
|-----------|-------------|---------------------|
| **Exine** | Outer sporopollenin layer | Species-specific ornamentation |
| **Apertures** | Colpi (furrows) and pores | Number and arrangement define pollen types |
| **Shape** | Spheroidal, prolate, oblate | Polar/equatorial axis ratio |
| **Ornamentation** | Reticulate, echinate, psilate, etc. | Species-level diagnosis |
| **Size** | 10–200 μm | Family/genus-level diagnosis |

### Optical Challenges

| Challenge | Effect | Compensating augmentation |
|-----------|--------|---------------------------|
| Limited depth of field | Partial defocus | Blur (kernel ≤ 3) |
| Illumination variation | Brightness gradients | ColorJitter, CLAHE |
| Preparation artefacts | Bubbles, residues | CoarseDropout (fill=128) |
| Random orientation | Polar / equatorial views | Rotate ±180°, Flip H/V |
| Grain overlap | Partial occlusions | Circularity filtering |
| Electronic noise | Shot + read noise | GaussNoise |

---

## Stage 2 — Grain Detection

> **Dual role:** U²-Net generates `.seg` annotations during preprocessing; **dense ViT** (`localization_detector: vit_dense`) is the active detector for full-image inference.  
> **Species classification** is performed by the best AnalogyNet + slice-aware model, not by the detector.

### U²-Net — Preprocessing and Legacy Backend

U²-Net (Qin et al., 2020) is a salient object detection (SOD) network that produces pixel-level saliency maps without bounding-box annotations.

```
RGB image (320×320)
    │
    ▼
Encoder: 6 RSU stages (Residual U-blocks)
    │     RSU = nested U-Net inside each block
    │     Multi-scale context at block level
    │     320×320 → 160×160 → ... → 20×20
    │
    ▼
Decoder: 5 RSU stages (skip connections)
    │     6 fused side outputs
    │
    ▼
Saliency P(x,y) ∈ [0, 1]   (salient-object probability)

Variants:
  U²-NET  : 44M params, 176 MB — maximum accuracy
  U²-NETP : 1.1M params, 4.7 MB — real-time (default)
```

### Post-Processing

```
Saliency P(x,y)
  → Adaptive threshold: T = μ(P) + k·σ(P),  k=0.30,  T ∈ [0.15, 0.50]
  → Binarisation: B(x,y) = 1 if P(x,y) > T
  → Morphology: Closing + Opening (elliptical 3×3 kernel)
  → findContours(EXTERNAL)
  → Filtering:
      Area ∈ [1000, 120000] px²
      Circularity: C = 4πA/P² ≥ 0.3
  → N DetectedGrain: {bbox, contour, saliency, centroid}
```

### Dense ViT — Active Detector (E2E Inference)

**PollenViTDetector** (`vit_dense_detector.py`) uses a DINOv2 backbone with a dense segmentation head. It produces pixel-level saliency maps at 504 px (letterbox), outperforming U²-Net in recall on small grains and dense fields.

**Coarse-to-fine strategy** (`coarse_fine_detection.py`, `detection_strategy: coarse_fine`):

| Stage | Function |
|-------|----------|
| **Coarse** | Single global pass over the image (or `max_load_side`) |
| **Supplement** | Sliding-window + saliency peaks for missed grains |
| **Verify** | Local re-detection per candidate |
| **Quality gates** | Pre-classification filters (`detection_quality.py`) |

**Configurable quality gates** (no hard-coded thresholds outside config):

| Gate | Purpose |
|------|---------|
| `reject_edge_partial` | Reject grains partially cut at the image border |
| `split_multi_peak_enabled` | Split fused blobs (watershed on saliency) |
| `reject_multi_peak_area_frac` | Reject large multi-peak boxes (fusion) |
| `classification_bbox` | Tight crop aligned with `.seg` training (≠ display `bbox`) |

**References:** Oquab et al., 2024 (DINOv2); Felzenszwalb et al., 2010 (cascade); Veta et al., 2016 (tiling in microscopy).

---

## Stage 3 — .seg Format and On-The-Fly Dataset

### .seg Annotation Format

Each image has an associated plain-text `.seg` file (same stem, different extension):

```
# Generated by: PollenGrainDetector (U2-NETP)
# Image: Acaena_sp_001.png | Dimensions: 1920x1080
# Params: threshold: 0.3 | adaptive_k: 0.3 | min_area: 1000
#
# grain_index | class_index | bbox_x,y,w,h | saliency | contour
0 | 0 | 234,156,87,92 | 0.891 | 234,180,238,168,245,160,...
1 | 0 | 512,340,65,71 | 0.847 | 512,360,516,348,524,342,...
```

**Design principle: ZERO additional images on disk.** Crops are generated on-the-fly from the original image + `.seg` coordinates.

### SegmentedGrainDataset

```
Initialisation:
  Scan root/{class}/*.seg → register each grain as a sample
  LRU cache of 64 original images

__getitem__(idx):
  1. Read original image (LRU cache)
  2. Extract square crop with 20% padding
  3. Apply contour mask → `imagenet_neutral` background
  4. Resize to 252×252
  5. Apply transforms (Albumentations or torchvision)
  6. Return (tensor, class_idx) or (tensor, class_idx, mask)
```

### ImageNet-Neutral Background: Mathematical Rationale

After ImageNet normalisation (μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]), the neutral colour in normalised space is the zero vector. Pixels outside the contour contribute minimally to masked pooling — consistent between train and inference (`mask_background: imagenet_neutral`).

---

## Stage 4 — Preprocessing and Augmentations

### Albumentations Pipeline (Training)

| # | Group | Transform | Optical justification |
|---|-------|-----------|----------------------|
| 1 | **Scale** | RandomResizedCrop(scale=0.8–1.0, p=0.5) | Magnification variation |
| 2 | **Geom.** | HorizontalFlip (p=0.5) + VerticalFlip (p=0.5) | No preferred orientation |
| 3 | **Geom.** | Rotate(±180°, p=0.8, fill=128) | Arbitrary slide rotation |
| 4 | **Geom.** | Affine(translate ±8%, scale 0.9–1.1, fill=128) | Variable field position |
| 5 | **Colour** | ColorJitter(B=0.3, C=0.6, S=0.2, H=0.1, p=0.8) | Staining variability |
| 6 | **Micro** | RandomGamma(γ∈[80,120], p=0.5) | CCD sensor non-linearity |
| 7 | **Micro** | GaussNoise(σ∈[10,50]/255, p=0.5) | Electronic noise |
| 8 | **Micro** | Blur(kernel≤3, p=0.3) | Limited depth of field |
| 9 | **Micro** | OpticalDistortion(0.1, p=0.3) | Seidel aberrations |
| 10 | **Micro** | CLAHE(clip=2, grid 8×8, p=0.5) | Imperfect Köhler illumination |
| 11 | **Domain** | RGBShift(±20, p=0.3) | White-balance drift |
| 12 | **Domain** | CoarseDropout(1–3 holes, fill=128, p=0.15) | Occlusions (bubbles) |
| 13 | **Norm** | Normalize(μ_ImageNet, σ_ImageNet) + ToTensorV2 | DINOv2 compatibility |

**All borders/holes filled with neutral background** — consistent with `imagenet_neutral` in train and inference.

**Validation:** Only `Resize(252) → Normalize → ToTensor` (no augmentation).

---

## Stage 5 — Embedding Model (AnalogyNet + DINOv2)

### DINOv2 ViT-S/14 Backbone + Masked Pooling

**DINOv2** (Oquab et al., 2023): SSL foundation model on LVD-142M. This project **does not use the CLS token** for metric learning; it aggregates **patch tokens** from intermediate layers with multi-spectral statistics and the contour mask.

```
Image (3 × H × W)  +  mask aligned with augmentations
    │
    ▼
┌────────────────────────────────────────┐
│ Patch Embedding (14×14) + 12 blocks    │
│  GLOBAL self-attention at every layer  │
│  (grey bg ≈ 0 after ImageNet norm)     │
├────────────────────────────────────────┤
│ get_intermediate_layers(n=[3, 11])     │
│  layer 3  → texture / low semantics    │
│  layer 11 → semantics / pre-final      │
├────────────────────────────────────────┤
│ Masked multi-spectral pooling          │
│  per layer: mean + max + std           │
│  only patches with mask > threshold    │
│  DropToken (train, configurable p)     │
│  → feat_lf, feat_hf ∈ R^(3×384)=R^1152 │
├────────────────────────────────────────┤
│ projection_lf / projection_hf (MLP)    │
│  1152 → 256 → 256 → 64 per branch      │
├────────────────────────────────────────┤
│ L2-Cat-L2 → e-hat ∈ S^127 (128D)       │
└────────────────────────────────────────┘
    │
    ▼
Sliced MS Loss partitions 128D into S slices (default 4×32D)
```

**Note:** the image is **not** multiplied by the mask before the ViT (positional artefacts). Filtering is **post-attention**, in the pooling stage. See audit in `docs/20260210_1750_AUDITORIA_REGRESION_MASKED_POOLING.md`.

### Pooling ↔ Spectral Slices

| Stage | What it does | Effect on discriminative capacity |
|-------|--------------|-----------------------------------|
| **Pooling** | Compresses N patches → rich vector (mean/max/std, 2 layers) | Preserves intra-grain morphology (texture, heterogeneity) |
| **Projection** | Compresses 1152D → 128D | Learns which features enter metric space |
| **S slices** | Partitions 128D; independent MS+LDM per slice | Raises effective spectral rank ≈ S×(C−1) vs collapse to C−1 |

The N slices **do not replace** good pooling: they amplify metric geometry **if** each slice’s dimensions correspond to coherent signal sources. The [implementation plan](docs/2026-06-11_PLAN_IMPLEMENTACION_POOLING_SLICE_ALIGNED.md) defines the `slice_native` layout to align layers, pooling, and S-partition.

### Why DINOv2 Instead of a CNN?

| Property | CNN (ResNet-50) | DINOv2 ViT-S/14 |
|----------|-----------------|-----------------|
| Receptive field | Local → global (by depth) | **Global from layer 1** |
| Pretraining | Supervised (1.3M) | **SSL (142M images)** |
| Emergent segmentation | No | **Yes** (attention heads) |
| Parameters | 23.5M | **21M** (more efficient) |
| Transfer learning | Good | **Superior** (universal features) |

### L2-Normalisation Property

On S^{d−1} (‖a‖=‖b‖=1):

```
‖a − b‖_2² = 2(1 − cos(a, b))
```

Euclidean distance and cosine similarity are monotonically equivalent.

### Supported Backbones

| Backbone | Features | Params | Source |
|----------|----------|--------|--------|
| ResNet-18 | 512 | 11.7M | torchvision |
| ResNet-50 | 2048 | 23.5M | torchvision |
| ConvNeXt V2 Tiny | 768 | 28.6M | timm (FCMAE+IN22k) |
| **DINOv2 ViT-S/14** | **1152** (post-pool) / 384 (token) | **21M** | **torch.hub (default)** |
| DINOv2 ViT-B/14 | 768 | 86M | torch.hub |
| DINOv2 ViT-S/14 + Reg | 384 | 21M | torch.hub (registers) |
| DINOv2 ViT-B/14 + Reg | 768 | 86M | torch.hub (registers) |

**Embedding spaces:** Euclidean (S^{d−1}) or Hyperbolic (Poincaré ball, Ermolov et al., CVPR 2022).

---

## Stage 6 — Training (Sliced MS Loss)

Training prioritises morphologically hard pairs via **integrated soft mining** in each slice (see [Hard Example Mining → Soft Mining](#hard-example-mining--soft-mining-integrated-in-sliced-ms)); explicit miners (OHEM / triplet) remain legacy for `arcface`/`triplet`.

### Loss Function (default: `sliced_ms`)

The current pipeline uses **Sliced Multi-Similarity Loss with Learnable Dynamic Margins** (LDM-MS per subspace). It partitions the 128D embedding into **S=4 disjoint 32D subspaces** and applies MS Loss (Wang et al., CVPR 2019) independently in each slice, with learnable per-class margins.

Motivation: mitigate **Neural Collapse** (Papyan et al., PNAS 2020) and raise the effective spectral rank of the embedding space (Ko & Gu, CVPR 2020; slicing inspired by Sanakoyeu et al., CVPR 2019).

```
L_sliced = (1/S) Σ_{s=1..S} L_MS^{(s)}(normalize(z^{(s)}), y)
```

where z^{(s)} is slice s of the embedding and each slice is L2-normalised before cosine similarity.

| Aspect | Global 128D k-NN | Sliced MS + slice-aware inference |
|--------|------------------|-----------------------------------|
| Train/infer geometry | May diverge | Coherent per slice |
| Dimensional collapse | Tendency to simplex (C−1) | S independent simplexes |
| Inference classification | Centroids / 128D k-NN | Mean cosine per slice vs representatives |

**Alternative losses:** `sliced_proxy` (SG-Softmax per slice), `hybrid_proxy_ms`, Triplet, ArcFace, Proxy Anchor.

Documentation: `docs/2026-06-09_INFORME_MODIFICACIONES_PIPELINE_MULTI_SLICE.md` (slices + inference) and `docs/2026-06-11_PLAN_IMPLEMENTACION_POOLING_SLICE_ALIGNED.md` (configurable pooling + slice-native alignment).

### Multi-Similarity Loss (Base per Slice)

**Multi-Similarity Loss** (Wang et al., CVPR 2019) jointly exploits positive and negative relations of each sample with respect to the full batch:

```
L_MS = (1/m) Σ_i {
  (1/α) log[ 1 + Σ_{k∈P_i} exp(−α(S_ik − λ)) ]
  + (1/β) log[ 1 + Σ_{k∈N_i} exp( β(S_ik − λ)) ]
}
```

where S_ik = ⟨f_θ(x_i), f_θ(x_k)⟩, P_i = positives, N_i = negatives, α=2, β=50, λ=0.5.

| Aspect | Triplet Loss | Multi-Similarity Loss |
|--------|--------------|----------------------|
| Relations / batch | O(n³) triplets | All pairs: O(n²) |
| Mining | External (semi-hard) | Integrated (soft mining) |
| Gradients | Sparse | Dense |
| Convergence | Slow | Fast |

### Hard Example Mining → Soft Mining Integrated in Sliced MS

A design goal from the start was to **focus training on the hardest morphological separations** — positive/negative pairs at the margin boundary, confusable melissopalynology classes. That idea was implemented in two phases:

| Phase | Mechanism | Compatible loss | Status |
|-------|-----------|-----------------|--------|
| **v1 (Mar 2026)** | `HardNegativeMiner` (OHEM) + optional triplet miners | `arcface`, `triplet` | Legacy — `src/training/hard_mining.py` |
| **v2 (current)** | **Soft mining** via MS LogSumExp + LDM per slice | `sliced_ms` (default) | **Active** — replaces external miner |

#### v1 — Explicit Miners (Legacy)

Two blocks in `config.yaml`, both **disabled** with `sliced_ms`:

```yaml
training:
  mining:          # Triplet Mining — semihard / hard / distance_weighted (pytorch_metric_learning)
    enabled: false
  hard_mining:     # OHEM — top-k% highest-loss samples (Shrivastava et al., CVPR 2016)
    enabled: false
```

- **`training.hard_mining`:** computes per-sample loss, selects the hardest top-k%, averages only those. Used in ArcFace runs (Mar 2026, ratio 0.3).
- **`training.mining`:** triplet miners (`TripletMarginMiner`). Useful with Triplet Loss; automatically skipped with ArcFace.

#### v2 — Mining Integrated per Slice (Current Pipeline)

With **`sliced_ms`**, the external miner **does not participate** (OHEM auto-disabled; triplet mining incompatible with LDM). Hard-pair detection occurs **inside** each 32D subspace:

1. **MS soft mining:** LogSumExp with β=50 exponentially weights **nearby negatives** (implicit hard negatives); α modulates distant positives. All batch pairs contribute — dense gradients, no graph truncation.
2. **Per-class, per-slice LDM:** learnable `pos_margin` / `neg_margin` adapt difficulty **per species** in each subspace.
3. **S=4 slices:** difficulty is distributed across 4 independent geometries → spectral rank ≈ S × (C−1) vs collapse to one simplex.

> **Design note:** using hard miners (`indices_tuple`) with MS/LDM **distorts** soft mining and favours dimensional collapse (`src/losses/ldm_loss.py`). Explicit mining was therefore removed from the default path when adopting `sliced_ms`.

**For Methods in the paper:** describe as *implicit hard-pair weighting via Multi-Similarity LogSumExp across S orthogonal subspaces with learnable dynamic margins* — not as OHEM or external triplet mining.

**Additional supported losses:** Triplet Margin, Proxy Anchor (CVPR 2020), ArcFace (CVPR 2019), SG-Softmax (AAAI 2023).

### Sampling: M-Per-Class Balanced Sampler

```
P=6 classes/batch × K=12 samples/class = 72 images/batch

Relations per batch:
  Positive pairs: K(K−1)/2 × P = 66 × 6 = 396
  Negative pairs: K² × P(P−1)/2 = 144 × 15 = 2160
  → MS Loss exploits ALL pairs → dense gradients
```

### Optimisation Configuration

| Component | Technique | Parameters |
|-----------|-----------|------------|
| **Optimizer** | AdamW | lr=10⁻⁴, weight_decay=5×10⁻⁴ |
| **Scheduler** | Cosine Annealing | T_max=30, η_min=10⁻⁶ |
| **Warmup** | Linear ramp | 3 epochs from 10⁻⁵ |
| **Mixed Precision** | AMP (float16/32) | ~50% less VRAM, ~30% faster |
| **Gradient Clipping** | Max norm | 5.0 |
| **Early Stopping** | `validation_metric` (default mAP@R) | Patience=20, restore best |

### SOTA Regularisation (2022–2024)

| Technique | Paper | Status in trainer |
|-----------|-------|-------------------|
| **NIR** | Roth et al., CVPR 2022 | Optional |
| **HIER** | Kim et al., CVPR 2023 | Optional |
| **Anti-collapse** | Wang & Isola, ICML 2020 | Optional (`sota.anti_collapse`) |
| **Self-Distillation** | Zeng et al., 2024 | Optional |
| Uniformity / VICReg / disentanglement (GUI) | — | Not implemented — validator rejects if enabled |

### Training Artefacts

| Artefact | Description |
|----------|-------------|
| `slice_representatives.pt` | **Primary inference** — per-slice representatives `(S, C, D_s)` |
| `class_centroids.pt` | Global centroids (reference / legacy) |
| `reference_embeddings.pt` | Train-set embeddings + labels |
| `training_metrics.csv` | Epoch-by-epoch metrics |
| `best_model.pth` | Checkpoint with embedded training configuration |

**Principle:** evaluation never recomputes products of the training phase.

---

## Declarative Configuration and Alignment Plan

### Principle: Nothing Hard-Coded

Every architectural decision must live in `config.yaml` and be validated at startup (`src/utils/config_validator.py`). Values that were previously fixed in code and move to config:

| Parameter | Current code | Config destination |
|-----------|--------------|--------------------|
| ViT layers | `[3, 11]` | `model.pooling.layers` |
| Statistics | mean+max+std | `model.pooling.statistics` |
| DropToken | `p=0.3` | `model.pooling.drop_token` |
| Patch mask threshold | `0.01` | `model.pooling.masked.threshold` |
| Projection layout | Dual LF/HF | `model.projection.layout: slice_native` |
| S slices | `num_slices: 4` | already in `training.loss.params` |

### Central Parameters (Current)

```yaml
model:
  embedding_dim: 128      # divisible by num_slices
training:
  loss:
    type: sliced_ms
    params:
      num_slices: 4       # S — controls spectral rank
grain_detection:
  localization_detector: vit_dense
  detection_strategy: coarse_fine
  masked_pooling: true
```

### Implementation Roadmap (2026-06-11)

| Phase | Deliverable | Priority |
|-------|-------------|----------|
| **0** | Reproducible baseline + hardcode inventory | High |
| **1** | `model.pooling` schema + extended validator | High |
| **2** | Unified `PoolingModule` + `EmbeddingExtractor` | **Critical** |
| **3** | `slice_native` projection (dims ↔ slices) | High |
| **4** | XAI/Grad-CAM same forward as train | Medium |
| **5** | Ablation matrix (GeM, layers, S, masking) | Medium |
| **6** | Checkpoint migration + docs | Low |

Full plan with formulae, acceptance criteria, and 2022–2026 references:  
**[`docs/2026-06-11_PLAN_IMPLEMENTACION_POOLING_SLICE_ALIGNED.md`](docs/2026-06-11_PLAN_IMPLEMENTACION_POOLING_SLICE_ALIGNED.md)**

Implementation log (2026-06-11): **`docs/2026-06-11_IMPLEMENTACION_POOLING_SLICE_CHANGELOG.md`** — smoke test: `python scripts/smoke_test_pooling_slice.py --config test_config.yaml`

### Resolved Gaps (2026-06-12)

- Checkpoint and report aligned with `validation_metric` (mAP@R).
- Validation `val_loss` uses `prenorm` under sliced_ms.
- Post-training separates slice-aware vs 1-NN LOO metrics.
- Full-image inference with dense ViT coarse-fine (`PollenViTDetector`) + configurable quality gates.
- Validator rejects unimplemented regularisers.

Full register: [`docs/2026-06-12_17-44-00_REGISTRO_DML_PIPELINE_REFERENCIAS.md`](docs/2026-06-12_17-44-00_REGISTRO_DML_PIPELINE_REFERENCIAS.md)

---

## Stage 7 — Evaluation and Metrics

| Metric | Definition | Role |
|--------|------------|------|
| **Checkpoint** | `training.validation_metric` (default **mAP@R**) | Best-model selection (Musgrave et al., ECCV 2020) |
| **R@1, mAP@R, NMI, F1-macro** | 1-NN LOO MLRC on embeddings | Standard retrieval metrics |
| **slice_accuracy** | Mean cosine per slice vs representatives | Deployment classifier (Sanakoyeu et al., CVPR 2019) |
| **effective_rank** | Components for 95% variance on val spectrum | Anti-collapse monitor |
| **val_loss** | Auxiliary P×K (aligned with prenorm in sliced_ms) | Curves; not checkpoint criterion |

**Do not mix** slice_accuracy with R@1 in paper tables — they are distinct protocols.

---

## Stage 8 — Post-Training Analysis

Automated **9-stage** pipeline on the best model:

| # | Stage | Output |
|---|-------|--------|
| 1 | Training curves (loss, recall, lr) | `loss_curves.png`, `recall_curves.png` |
| 2 | 2D + 3D t-SNE of embeddings | `tsne_2d.png`, `tsne_3d.png` |
| 3 | Confusion matrix (kNN) | `confusion_matrix.png` |
| 4 | Intra/inter distance distribution | `distance_distribution.png` |
| 5 | Inference preview | `inference_preview.png` |
| 6 | **Attention heatmap gallery** | `gradcam_gallery/*.png` |
| 7 | Per-class inference gallery | `inference_gallery/*.png` |
| 8 | Full-image inference (test) | `fullimage_inference/*.png` |
| 9 | Automatic Markdown report | `training_report.md` |

---

## Stage 9 — End-to-End Inference

Two-stage pipeline (localise → identify), standard in automated microscopy (Litjens et al., 2017):

```
Camera image (any resolution)
  │
  ├── PollenViTDetector (dense ViT + coarse-fine + quality gates) → N grains
  │     coarse → supplement (SW) → verify → split → NMS → quality filter
  │
  └── For each grain (every detection is classified; no silent drop):
        Crop with classification_bbox + padding → Contour mask
        → imagenet_neutral bg → Val transform
        → AnalogyNet(mask if masked_pooling) → e-hat ∈ S^127
        → SliceAwareClassifier vs slice_representatives.pt
  │
  └── Result: annotated image + species counts + heatmaps
      (Grad-CAM with classification_bbox)
```

| Stage | Module | Task | Trains? |
|-------|--------|------|---------|
| Localisation | `PollenViTDetector` + `coarse_fine_detection` | Where the grain is | Yes (separate detector) |
| Quality gates | `detection_quality.py` | Filter partials/fusions | No (config) |
| Identification | AnalogyNet + `SliceAwareClassifier` | Which species | Yes (best checkpoint) |

Active config (`grain_detection`):

```yaml
localization_detector: vit_dense
detection_strategy: coarse_fine
sliding_window_enabled: true
max_load_side: 1536
nms_iou_threshold: 0.5
crop_padding: 0.2
detector_bbox_expand_frac: 0.0      # tight crop = .seg training
max_area_frac: 0.15
reject_edge_partial: true
split_multi_peak_enabled: true
reject_multi_peak_area_frac: 0.055  # multi-peak only, not by size
drop_low_confidence_detections: false
```

**Alternative backends:** `u2net` (classic SOD), `vit` (legacy ViT attention, `ModelGrainDetector`).

**E2E validation scripts:**

```bash
python scripts/regenerate_fullimage_folder.py   # Regenerate val/test gallery
python scripts/analyze_gallery_folder.py        # E2E metrics → gallery_analysis.json
```

**Literature review and change log:** [`docs/2026-06-13_REVISION_BIBLIOGRAFICA_Y_PIPELINE_E2E.md`](docs/2026-06-13_REVISION_BIBLIOGRAFICA_Y_PIPELINE_E2E.md)

---

## Stage 10 — Explainability (DINOv2 Attention)

### CLS-to-Patch Self-Attention

DINOv2 self-attention maps naturally segment the object of interest (Caron et al., ICCV 2021). Correct method for ViTs (GradCAM was designed for CNNs):

```
Hook on blocks[-1].attn.qkv → Extract Q, K
  → A = softmax(QK^T / √d_k)         (B, 6 heads, 325, 325)
  → A_cls = A[:, :, 0, 1:]            CLS → 324 patches
  → Average over heads → (18, 18)
  → Bicubic interpolation → (252, 252)
  → Heatmap [0,1]: red=grain, blue=background
```

### Gradient-Weighted Attention (Task-Specific)

For metric learning, each attention head is weighted by its contribution to classification (Chefer et al., CVPR 2021):

```
Forward with gradients → cos_sim(embedding, target_slice_aware) → backward
  → Per-head importance: h_j = mean(|∂sim/∂QKV_j|)
  → Weighted attention: A_weighted = Σ_j (h_j · A_cls_j)
  → Shows which patches contribute to CORRECT CLASSIFICATION
```

### GradCAM vs Native Attention

| Property | GradCAM (CNNs) | DINOv2 Attention |
|----------|----------------|------------------|
| Designed for | CNNs (Selvaraju, 2017) | **Transformers (Caron, 2021)** |
| Reliability on ViTs | Low (diffuse/errors) | **High (emergent property)** |
| Dependency | pytorch_grad_cam | **PyTorch hooks only** |
| Task-specific | Yes | **Yes (gradient weighting)** |

The system auto-detects backbone type and selects the correct method.

---

## Experimental Results

### Classification (Isolated DML, `.seg` Crops)

| Component | Configuration |
|-----------|---------------|
| **Backbone** | DINOv2 ViT-S/14 (21M parameters) |
| **Embedding** | 128D on S^127 |
| **Loss** | `sliced_ms` (4×32D) + LDM |
| **Inference** | `SliceAwareClassifier` + per-slice representatives |
| **Isolated accuracy (val)** | ~97.5% |

### End-to-End Inference (Val Gallery, 46 Images)

Run: `runs/2026-06-13_09-50-06` · Detector: `runs/2026-06-13_03-02-32/detector/best_detector.pth`

| Metric | Value |
|--------|-------|
| **E2E legacy** (correct classifications / total detections) | **89.5%** (85/95) |
| **Detection recall** (annotated `.seg` with IoU ≥ 0.3) | **95.1%** (77/81) |
| **Classification on matched** | **90.8%** |
| Gevuina avellana / Anthemis cotula / Eucryphia | **100%** |

The gap between isolated classifier (~97%) and E2E (~90%) mainly reflects **localisation** errors, not ViT discriminative capacity.

DML evaluation metrics: R@K, NMI, mAP@R, F1-macro (MLRC protocol, Musgrave et al., ECCV 2020).

---

## Software Architecture

```
MeliVision/
├── app.py                              # Entry point (GUI / CLI)
├── config.yaml                         # Full declarative configuration
│
├── src/                                # Scientific core
│   ├── models/
│   │   ├── analogy_net.py              # AnalogyNet: backbone + pooling + projection
│   │   ├── pooling.py                  # Configurable PoolingModule
│   │   ├── embedding_extractor.py      # Canonical forward path
│   ├── inference/
│   │   └── slice_classifier.py         # SliceAwareClassifier + representatives
│   │   ├── hyperbolic.py               # Poincaré Ball (hyperbolic embedding)
│   │   └── u2net/                      # U²-Net for saliency
│   │       └── u2net.py                # U²-Net / U²-NETP architecture
│   │
│   ├── grain_detection/                # Detection pipeline
│   │   ├── vit_dense_detector.py       # PollenViTDetector (active backend)
│   │   ├── coarse_fine_detection.py    # Coarse → supplement → verify
│   │   ├── detection_quality.py        # Configurable quality gates
│   │   ├── grain_crop_utils.py         # Unified crops/masks train↔infer
│   │   ├── detected_grain.py           # DetectedGrain + classification_bbox
│   │   ├── detection_profiles.py       # sensitive/balanced/strict presets
│   │   ├── salient_detector.py         # U²-Net wrapper (SOD)
│   │   ├── grain_detector.py           # PollenGrainDetector (U²-Net)
│   │   ├── model_grain_detector.py     # Legacy backend (ViT attention)
│   │   ├── seg_format.py               # .seg parser/writer
│   │   ├── annotator.py                # SegmentationAnnotator (batch)
│   │   └── full_image_classifier.py    # E2E camera pipeline
│   │
│   ├── data/
│   │   ├── grain_dataset.py            # SegmentedGrainDataset (on-the-fly)
│   │   ├── dataset.py                  # MielDataset (legacy ImageFolder)
│   │   └── transforms_enhanced.py      # SOTA augmentations (Albumentations)
│   │
│   ├── training/
│   │   ├── trainer_enhanced.py         # Trainer: AMP, mining, EMA, logging
│   │   └── ema_teacher.py              # Self-Distillation EMA (2024)
│   │
│   ├── losses/
│   │   ├── nir_regularizer.py          # NIR (CVPR 2022)
│   │   ├── hier_regularizer.py         # HIER (CVPR 2023)
│   │   ├── sg_softmax.py               # SG-Softmax (AAAI 2023)
│   │   └── dada_wrapper.py             # DADA (AAAI 2024)
│   │
│   ├── inference/
│   │   ├── knn_classifier.py           # k-NN classifier
│   │   └── faiss_index.py              # FAISS index
│   │
│   ├── utils/
│   │   ├── inference_utils.py          # Attention maps, GradCAM, embeddings
│   │   └── metrics_utils.py            # R@K, NMI, confusion matrix
│   │
│   └── visualization/                  # Visualisation tools
│
├── scripts/
│   ├── train.py                        # Full training
│   ├── post_training.py                # Post-training analysis (9 stages)
│   ├── regenerate_fullimage_folder.py  # Regenerate E2E gallery + Grad-CAM
│   └── analyze_gallery_folder.py       # E2E metrics on gallery
│
├── lib/                                # Graphical interface (PyQt5)
│   ├── gui_app.py                      # Main window (5 tabs)
│   ├── config_editor.py                # Tab: visual config.yaml editor
│   ├── dataset_manager_gui.py          # Tab: data manager (split)
│   ├── contour_analysis_tab.py         # Tab: contour analysis (U²-Net)
│   ├── inference_tab.py                # Tab: inference (single/batch)
│   └── styles.py                       # Style system
│
├── models/weights/                     # U²-Net weights (auto-download)
├── data/processed/{train,val,test}/    # Partitioned data (symlinks)
├── runs/                               # Per-experiment results
├── docs/                               # Technical documentation
└── tests/                              # Unit tests (pytest)
```

### Technology Stack

| Layer | Technology | Function |
|-------|------------|----------|
| **Deep Learning** | PyTorch ≥ 2.0 | Base framework |
| **Metric Learning** | pytorch-metric-learning ≥ 2.0 | Losses, miners, samplers |
| **Vision Transformer** | DINOv2 (torch.hub) | ViT-S/14 backbone |
| **Object Detection** | Dense ViT + U²-Net (custom) | Grain localisation (SOD) |
| **Augmentation** | Albumentations 2.0 | Augmentation pipeline |
| **Vector search** | FAISS | Indices for k-NN |
| **Visualisation** | matplotlib, seaborn | Plots and heatmaps |
| **Dim. reduction** | scikit-learn (t-SNE) | Embedding visualisation |
| **GUI** | PyQt5 | Graphical interface |
| **Processing** | OpenCV, Pillow | Image I/O and processing |
| **Testing** | pytest | Unit tests |

### Graphical Interface (GUI)

5 tabs: **Configuration** (visual YAML editor) → **Data Manager** (train/val/test split) → **Contour Analysis** (U²-Net explorer, click→crop) → **Training** (async, progress bar) → **Inference** (single/batch from checkpoint).

---

## Data Availability

Datasets supporting the findings of this study are managed under FLAME³, Universidad de Concepción. Where participant confidentiality or material-transfer restrictions do not apply, the minimal dataset required to interpret, replicate, and build upon the reported results will be deposited in a public repository with accession codes / DOIs listed in the manuscript **Data Availability** section.

Until public deposition is complete, processed layout conventions and local paths are described under `data/` (see also `data/README.md`). Editors and referees may request controlled access consistent with *Scientific Reports* policy.

---

## Code Availability

The custom source code central to the conclusions of this work is available in this GitHub repository:

- **Repository:** https://github.com/Nahzap/MeliVision  
- **Version associated with the manuscript:** tag / release corresponding to the paper submission (authors will mint a DOI via Zenodo or an institutional archive for the frozen version)  
- **Configuration:** `config.yaml` (declarative; validated at startup)  
- **License of use:** to be stated explicitly in the repository `LICENSE` file and in the manuscript Code Availability statement (include any commercial or redistribution restrictions)

Any restrictions on code availability must be described in the manuscript. Subsequent revisions should remain under version control; authors are encouraged to deposit DOI-minted releases for each paper-associated freeze.

---

## Citation

If you use MeliVision in academic work, please cite the associated *Scientific Reports* article (once published) and this repository version. A BibTeX entry will be added upon acceptance. Affiliation for correspondence:

**FLAME³**, Universidad de Concepción, Chile  
(Fundamental of Light, Applied optics and Microscale engineering, Experiments and Explorations)

---

## References

1. **Qin, X. et al.** (2020). U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection. *Pattern Recognition*, 106.
2. **Oquab, M. et al.** (2023). DINOv2: Learning Robust Visual Features without Supervision. *arXiv:2304.07193*.
3. **Caron, M. et al.** (2021). Emerging Properties in Self-Supervised Vision Transformers. *ICCV 2021*.
4. **Wang, X. et al.** (2019). Multi-Similarity Loss with General Pair Weighting for DML. *CVPR 2019*.
5. **Chefer, H. et al.** (2021). Transformer Interpretability Beyond Attention Visualization. *CVPR 2021*.
6. **Dosovitskiy, A. et al.** (2021). An Image Is Worth 16×16 Words. *ICLR 2021*.
7. **Touvron, H. et al.** (2022). DeiT III: Revenge of the ViT. *ECCV 2022*.
8. **Selvaraju, R. et al.** (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV 2017*.
9. **Louveaux, J. et al.** (1978). Methods of Melissopalynology. *Bee World*, 59(4).
10. **Sevillano, V. et al.** (2020). Precise Pollen Classification with Deep Learning. *PLOS ONE*.
11. **Musgrave, K. et al.** (2020). A Metric Learning Reality Check. *ECCV 2020*.
12. **Roth, K. et al.** (2022). Non-Isotropy Regularization for Proxy-based DML. *CVPR 2022*.
13. **Kim, S. et al.** (2023). HIER: Metric Learning Beyond Class-Level Similarities. *CVPR 2023*.
14. **Kornblith, S., Shlens, J., & Le, Q. V.** (2019). Do Better ImageNet Models Transfer Better? *CVPR 2019*.
15. **Macenko, M. et al.** (2009). A Method for Normalizing Histology Slides for Quantitative Analysis. *ISBI 2009*.
16. **Reinhard, E., Ashikhmin, M., Gooch, B., & Shirley, P.** (2001). Color Transfer between Images. *IEEE CG&A*, 21(5).
17. **Loshchilov, I., & Hutter, F.** (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. *ICLR 2017*.
18. **Shrivastava, A. et al.** (2016). Training Region-based Object Detectors with Online Hard Example Mining. *CVPR 2016* (OHEM legacy — `hard_mining`).
19. **Schroff, F. et al.** (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. *CVPR 2015* (triplet mining).

**Bibliographic audit (Ashikhmin, Macenko, venue corrections):** [`docs/20260207_172200_REVISION_REFERENCIAS_SOTA.md`](docs/20260207_172200_REVISION_REFERENCIAS_SOTA.md)

---

## Academic Documentation

| Document | Content |
|----------|---------|
| [`docs/2026-06-13_REVISION_BIBLIOGRAFICA_Y_PIPELINE_E2E.md`](docs/2026-06-13_REVISION_BIBLIOGRAFICA_Y_PIPELINE_E2E.md) | Literature review (physics, engineering, palynology) + E2E register |
| [`docs/2026-06-12_17-44-00_REGISTRO_DML_PIPELINE_REFERENCIAS.md`](docs/2026-06-12_17-44-00_REGISTRO_DML_PIPELINE_REFERENCIAS.md) | DML register, P0/P1 corrections |
| [`docs/2026-06-12_PLAN_CUMPLIMIENTO_ACADEMICO_SOTA.md`](docs/2026-06-12_PLAN_CUMPLIMIENTO_ACADEMICO_SOTA.md) | IEEE/SOTA compliance plan |
| [`docs/2026-06-11_PLAN_IMPLEMENTACION_POOLING_SLICE_ALIGNED.md`](docs/2026-06-11_PLAN_IMPLEMENTACION_POOLING_SLICE_ALIGNED.md) | Configurable pooling + slicing |

**Ablations F-01…F-07:** `python scripts/run_ablation_matrix.py --matrix pooling --dry-run`

---

**Version:** 7.0 (two-stage inference: dense ViT coarse-fine localises, ViT classifies)  
**Last updated:** 2026-11-26  
**Laboratory:** FLAME³, Universidad de Concepción  
**DML register and references:** [`docs/2026-06-12_17-44-00_REGISTRO_DML_PIPELINE_REFERENCIAS.md`](docs/2026-06-12_17-44-00_REGISTRO_DML_PIPELINE_REFERENCIAS.md)  
**E2E literature review and pipeline:** [`docs/2026-06-13_REVISION_BIBLIOGRAFICA_Y_PIPELINE_E2E.md`](docs/2026-06-13_REVISION_BIBLIOGRAFICA_Y_PIPELINE_E2E.md)  
**Academic compliance plan:** [`docs/2026-06-12_PLAN_CUMPLIMIENTO_ACADEMICO_SOTA.md`](docs/2026-06-12_PLAN_CUMPLIMIENTO_ACADEMICO_SOTA.md)
