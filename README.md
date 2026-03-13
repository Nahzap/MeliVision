# MeliVision

Sistema de **microscopía automatizada** para clasificación de granos de polen mediante **Deep Metric Learning** con **DINOv2 Vision Transformer** y detección de objetos salientes con **U²-Net**.

Aprende una función de embedding $f_\theta: \mathbb{R}^{H \times W \times 3} \rightarrow \mathbb{S}^{127}$ que mapea crops microscópicos a una hiperesfera unitaria de 128 dimensiones, donde la distancia euclidiana refleja la disimilitud taxonómica entre especies poliníferas.

---

## Tabla de Contenidos

- [Contexto Científico](#contexto-científico)
- [Pipeline Completo del Sistema](#pipeline-completo-del-sistema)
- [Etapa 1 — Adquisición Óptica](#etapa-1--adquisición-óptica)
- [Etapa 2 — Detección de Granos (U²-Net)](#etapa-2--detección-de-granos-u²-net)
- [Etapa 3 — Formato .seg y Dataset On-The-Fly](#etapa-3--formato-seg-y-dataset-on-the-fly)
- [Etapa 4 — Preprocesamiento y Augmentaciones](#etapa-4--preprocesamiento-y-augmentaciones)
- [Etapa 5 — Modelo de Embedding (AnalogyNet + DINOv2)](#etapa-5--modelo-de-embedding-analogynet--dinov2)
- [Etapa 6 — Entrenamiento (Multi-Similarity Loss)](#etapa-6--entrenamiento-multi-similarity-loss)
- [Etapa 7 — Evaluación y Métricas](#etapa-7--evaluación-y-métricas)
- [Etapa 8 — Análisis Post-Entrenamiento](#etapa-8--análisis-post-entrenamiento)
- [Etapa 9 — Inferencia End-to-End](#etapa-9--inferencia-end-to-end)
- [Etapa 10 — Explicabilidad (Atención DINOv2)](#etapa-10--explicabilidad-atención-dinov2)
- [Resultados Experimentales](#resultados-experimentales)
- [Arquitectura de Software](#arquitectura-de-software)
- [Referencias](#referencias)

---

## Contexto Científico

### Melisopalinología

La **melisopalinología** — estudio del contenido polínico de la miel — es fundamental para la certificación de origen geográfico y botánico. El análisis manual requiere microscopía óptica directa (>30 min/muestra) con alta dependencia del operador (Louveaux et al., 1978). Este sistema automatiza el proceso completo: detección, clasificación y conteo de granos de polen.

### ¿Por qué Deep Metric Learning?

A diferencia de clasificadores softmax con $C$ neuronas de salida fijas:

| Propiedad | Clasificador Softmax | Deep Metric Learning |
|-----------|---------------------|---------------------|
| Nuevas especies | Re-entrenamiento completo | Solo agregar embeddings de referencia |
| Escalabilidad | $O(C)$ neuronas finales | Base vectorial extensible |
| Few-shot | Requiere muchos datos/clase | Funciona con pocas referencias |
| Interpretabilidad | Logits abstractos | Distancias en espacio métrico |
| Generalización | Solo clases vistas | Zero-shot a clases no vistas |

### Dataset

El sistema soporta datasets con múltiples clases de polen. La partición estratificada recomendada es **70% train** · **15% val** · **15% test**.

---

## Pipeline Completo del Sistema

```
╔══════════════════════════════════════════════════════════════════╗
║                   PIPELINE COMPLETO (10 ETAPAS)                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  ┌───────────────────┐                                            ║
║  │ IMAGEN MICROSCOPIO │  Óptico, campo claro, sensor CCD/CMOS    ║
║  │ (H × W × 3)       │  Iluminación Köhler, obj. 60x       ║
║  └─────────┬─────────┘                                            ║
║            │                                                      ║
║            ▼                                                      ║
║  ┌──────────────────────────────────────────┐                     ║
║  │ ETAPA 2: U²-Net Salient Object Detection │                     ║
║  │  Imagen → U²-NETP (320×320) → Saliencia  │                     ║
║  │  → Umbral adaptativo → Morfología         │                     ║
║  │  → Contornos → Filtrado (área, circ.)     │                     ║
║  │  → N detecciones {bbox, contorno, prob}   │                     ║
║  └─────────┬────────────────────────────────┘                     ║
║            │                                                      ║
║            ▼                                                      ║
║  ┌──────────────────────────────────────────┐                     ║
║  │ ETAPA 3: Anotación .seg (texto plano)    │                     ║
║  │  CERO imágenes adicionales en disco       │                     ║
║  │  Formato: idx|cls|bbox|saliency|contorno  │                     ║
║  └─────────┬────────────────────────────────┘                     ║
║            │                                                      ║
║            ▼                                                      ║
║  ┌──────────────────────────────────────────┐                     ║
║  │ ETAPA 4-5: Crop + Mask + Augment + Embed │                     ║
║  │  .seg + img → Crop cuadrado + padding     │                     ║
║  │  → Máscara contorno → Gray bg (128)       │                     ║
║  │  → Albumentations (geom+color+micro)      │                     ║
║  │  → Normalización ImageNet                 │                     ║
║  │                                           │                     ║
║  │  ┌─────────────────────────────────────┐  │                     ║
║  │  │ DINOv2 ViT-S/14 (21M params)       │  │                     ║
║  │  │  14×14 patches → 324 tokens         │  │                     ║
║  │  │  12 transformer blocks, dim=384     │  │                     ║
║  │  │  Mean pooling → ℝ^384              │  │                     ║
║  │  ├─────────────────────────────────────┤  │                     ║
║  │  │ Projection Head (MLP)               │  │                     ║
║  │  │  384→256→256→128 + BN + ReLU        │  │                     ║
║  │  ├─────────────────────────────────────┤  │                     ║
║  │  │ L2 Norm → ê ∈ S^127               │  │                     ║
║  │  └─────────────────────────────────────┘  │                     ║
║  └─────────┬────────────────────────────────┘                     ║
║            │                                                      ║
║            ▼                                                      ║
║  ┌──────────────────────────────────────────┐                     ║
║  │ ETAPA 9: Clasificación por centroides    │                     ║
║  │  ŷ = argmin_c ‖ê − μ_c‖₂                │                     ║
║  └─────────┬────────────────────────────────┘                     ║
║            │                                                      ║
║            ▼                                                      ║
║  RESULTADO: {especie → conteo} + imagen anotada + heatmaps       ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Etapa 1 — Adquisición Óptica

### Sistema Óptico

| Parámetro | Especificación |
|-----------|---------------|
| **Microscopio** | Óptico compuesto, campo claro |
| **Objetivos** | 60x (inmersión en aceite) |
| **Iluminación** | Köhler (campo uniforme) |
| **Sensor** | CCD/CMOS digital, ≥ 1920×1080 |
| **Preparación** | Acetólisis de Erdtman o montaje directo en glicerina |

### Estructuras Diagnósticas del Grano de Polen

| Estructura | Descripción | Relevancia taxonómica |
|-----------|-------------|----------------------|
| **Exina** | Capa externa esporopolinínica | Ornamentación específica de especie |
| **Aperturas** | Colpos (surcos) y poros | Número y disposición definen tipos polínicos |
| **Forma** | Esferoidal, prolato, oblato | Relación eje polar/ecuatorial |
| **Ornamentación** | Reticulada, equinada, psilada, etc. | Diagnóstico a nivel de especie |
| **Tamaño** | 10–200 μm | Diagnóstico a nivel de familia/género |

### Desafíos Ópticos

| Desafío | Efecto | Augmentación compensatoria |
|---------|--------|---------------------------|
| Profundidad de campo limitada | Desenfoque parcial | Blur (kernel≤3) |
| Variación de iluminación | Gradientes de brillo | ColorJitter, CLAHE |
| Artefactos de preparación | Burbujas, residuos | CoarseDropout (fill=128) |
| Orientación aleatoria | Vistas polar / ecuatorial | Rotate ±180°, Flip H/V |
| Solapamiento de granos | Oclusiones parciales | Filtrado por circularidad |
| Ruido electrónico | Shot noise + read noise | GaussNoise |

---

## Etapa 2 — Detección de Granos (U²-Net)

### Arquitectura U²-Net

U²-Net (Qin et al., 2020) es una red de detección de objetos salientes (SOD) que genera mapas de saliencia pixel-level sin necesidad de anotaciones de bounding box.

```
Imagen RGB (320×320)
    │
    ▼
Encoder: 6 etapas RSU (Residual U-blocks)
    │     RSU = U-Net anidado dentro de cada bloque
    │     Captura contexto multi-escala a nivel de bloque
    │     320×320 → 160×160 → ... → 20×20
    │
    ▼
Decoder: 5 etapas RSU (skip connections)
    │     6 side outputs fusionados
    │
    ▼
Saliency P(x,y) ∈ [0, 1]   (probabilidad de objeto saliente)

Variantes:
  U²-NET  : 44M params, 176 MB — máxima precisión
  U²-NETP : 1.1M params, 4.7 MB — tiempo real (default)
```

### Post-Procesamiento

```
Saliency P(x,y)
  → Umbral adaptativo: T = μ(P) + k·σ(P),  k=0.30,  T ∈ [0.15, 0.50]
  → Binarización: B(x,y) = 1 si P(x,y) > T
  → Morfología: Closing + Opening (kernel elíptico 3×3)
  → findContours(EXTERNAL)
  → Filtrado:
      Área ∈ [1000, 120000] px²
      Circularidad: C = 4πA/P² ≥ 0.3
  → N DetectedGrain: {bbox, contorno, saliency, centroide}
```

---

## Etapa 3 — Formato .seg y Dataset On-The-Fly

### Formato de Anotación .seg

Cada imagen tiene un archivo `.seg` de texto plano asociado (mismo nombre, extensión diferente):

```
# Generado por: PollenGrainDetector (U2-NETP)
# Imagen: Acaena_sp_001.png | Dimensiones: 1920x1080
# Params: threshold: 0.3 | adaptive_k: 0.3 | min_area: 1000
#
# grain_index | class_index | bbox_x,y,w,h | saliency | contour
0 | 0 | 234,156,87,92 | 0.891 | 234,180,238,168,245,160,...
1 | 0 | 512,340,65,71 | 0.847 | 512,360,516,348,524,342,...
```

**Principio de diseño: CERO imágenes adicionales en disco.** Los crops se generan on-the-fly desde imagen original + coordenadas del `.seg`.

### SegmentedGrainDataset

```
Inicialización:
  Escanea root/{clase}/*.seg → registra cada grano como sample
  LRU cache de 64 imágenes originales

__getitem__(idx):
  1. Leer imagen original (cache LRU)
  2. Extraer crop cuadrado con padding 20%
  3. Aplicar máscara del contorno → Gray background (128)
  4. Resize a 252×252
  5. Aplicar transforms (Albumentations o torchvision)
  6. Return (tensor, class_idx) o (tensor, class_idx, mask)
```

### Gray Background: Justificación Matemática

Después de normalización ImageNet ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$):

$$\text{gray}_{norm} = \frac{128/255 - \mu}{\sigma} \approx (0.07, 0.02, -0.03) \approx \mathbf{0}$$

Los patches de fondo gris producen activaciones **~cero** en DINOv2 → contribuyen mínimamente al mean pooling sin necesidad de masked pooling explícito.

---

## Etapa 4 — Preprocesamiento y Augmentaciones

### Pipeline Albumentations (Training)

| # | Grupo | Transformación | Justificación óptica |
|---|-------|---------------|---------------------|
| 1 | **Escala** | RandomResizedCrop(scale=0.8–1.0, p=0.5) | Variación de magnificación |
| 2 | **Geom.** | HorizontalFlip (p=0.5) + VerticalFlip (p=0.5) | Sin orientación preferente |
| 3 | **Geom.** | Rotate(±180°, p=0.8, fill=128) | Rotación arbitraria del portaobjetos |
| 4 | **Geom.** | Affine(translate ±8%, scale 0.9–1.1, fill=128) | Posición variable en campo |
| 5 | **Color** | ColorJitter(B=0.3, C=0.6, S=0.2, H=0.1, p=0.8) | Variabilidad de tinción |
| 6 | **Micro** | RandomGamma(γ∈[80,120], p=0.5) | No-linealidad del sensor CCD |
| 7 | **Micro** | GaussNoise(σ∈[10,50]/255, p=0.5) | Ruido electrónico |
| 8 | **Micro** | Blur(kernel≤3, p=0.3) | Profundidad de campo limitada |
| 9 | **Micro** | OpticalDistortion(0.1, p=0.3) | Aberraciones de Seidel |
| 10 | **Micro** | CLAHE(clip=2, grid 8×8, p=0.5) | Iluminación Köhler imperfecta |
| 11 | **Domain** | RGBShift(±20, p=0.3) | Balance de blancos |
| 12 | **Domain** | CoarseDropout(1–3 huecos, fill=128, p=0.15) | Oclusiones (burbujas) |
| 13 | **Norm** | Normalize(μ_ImageNet, σ_ImageNet) + ToTensorV2 | Compatibilidad DINOv2 |

**Todos los bordes/huecos rellenados con gris (128)** — consistencia con gray background.

**Validación:** Solo `Resize(252) → Normalize → ToTensor` (sin augmentación).

---

## Etapa 5 — Modelo de Embedding (AnalogyNet + DINOv2)

### DINOv2 ViT-S/14 Backbone

**DINOv2** (Oquab et al., 2023): foundation model entrenado con self-supervised learning sobre 142M de imágenes curadas automáticamente (LVD-142M).

```
Imagen (3 × 252 × 252)
    │
    ▼
┌────────────────────────────────────────┐
│ Patch Embedding                        │
│  Conv2d(3, 384, kernel=14, stride=14)  │
│  252/14 = 18 patches por lado          │
│  18×18 = 324 patch tokens + 1 CLS      │
│  → (B, 325, 384)                       │
├────────────────────────────────────────┤
│ Positional Encoding (learnable)        │
├────────────────────────────────────────┤
│ 12 Transformer Encoder Blocks          │
│                                        │
│  Cada bloque:                          │
│  ┌──────────────────────────────────┐  │
│  │ LayerNorm                        │  │
│  │ Multi-Head Self-Attention        │  │
│  │   6 heads, dim=384, head_dim=64  │  │
│  │   Attn = softmax(QK^T/√64)·V    │  │
│  │ + Residual                       │  │
│  ├──────────────────────────────────┤  │
│  │ LayerNorm                        │  │
│  │ MLP (SwiGLU: 384→1536→384)      │  │
│  │ + Residual                       │  │
│  └──────────────────────────────────┘  │
│                                        │
│  Atención GLOBAL: cada patch atiende   │
│  a todos los demás en cada capa        │
├────────────────────────────────────────┤
│ Mean Pooling (excluir CLS)             │
│  (B, 324, 384) → mean → (B, 384)      │
└────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────┐
│ Projection Head (MLP)                  │
│  Linear(384, 256) + BN + ReLU + Drop   │
│  Linear(256, 256) + BN + ReLU + Drop   │
│  Linear(256, 128)                      │
├────────────────────────────────────────┤
│ L2 Normalization                       │
│  ê = e / ‖e‖₂  →  ê ∈ S^127          │
└────────────────────────────────────────┘
    │
    ▼
Embedding en hiperesfera unitaria (128-dim)
```

### ¿Por qué DINOv2 y no CNN?

| Propiedad | CNN (ResNet-50) | DINOv2 ViT-S/14 |
|-----------|----------------|------------------|
| Receptive field | Local → global (por capas) | **Global desde la capa 1** |
| Preentrenamiento | Supervisado (1.3M) | **SSL (142M imágenes)** |
| Segmentación emergente | No | **Sí** (attention heads) |
| Parámetros | 23.5M | **21M** (más eficiente) |
| Transfer learning | Bueno | **Superior** (features universales) |

### Propiedad de la L2-Normalización

En $\mathbb{S}^{d-1}$ ($\|a\|=\|b\|=1$):

$$\|a - b\|_2^2 = 2(1 - \cos(a, b))$$

Distancia euclidiana y similitud coseno son monótonamente equivalentes.

### Backbones Soportados

| Backbone | Features | Params | Fuente |
|----------|----------|--------|--------|
| ResNet-18 | 512 | 11.7M | torchvision |
| ResNet-50 | 2048 | 23.5M | torchvision |
| ConvNeXt V2 Tiny | 768 | 28.6M | timm (FCMAE+IN22k) |
| **DINOv2 ViT-S/14** | **384** | **21M** | **torch.hub (default)** |
| DINOv2 ViT-B/14 | 768 | 86M | torch.hub |
| DINOv2 ViT-S/14 + Reg | 384 | 21M | torch.hub (registers) |
| DINOv2 ViT-B/14 + Reg | 768 | 86M | torch.hub (registers) |

**Espacios de embedding:** Euclidiano ($\mathbb{S}^{d-1}$) o Hiperbólico (bola de Poincaré, Ermolov et al., CVPR 2022).

---

## Etapa 6 — Entrenamiento (Multi-Similarity Loss)

### Función de Pérdida

**Multi-Similarity Loss** (Wang et al., CVPR 2019) explota simultáneamente relaciones positivas y negativas de cada muestra respecto a todo el batch:

$$\mathcal{L}_{MS} = \frac{1}{m} \sum_{i=1}^{m} \left\{ \frac{1}{\alpha} \log \left[ 1 + \sum_{k \in \mathcal{P}_i} e^{-\alpha(S_{ik} - \lambda)} \right] + \frac{1}{\beta} \log \left[ 1 + \sum_{k \in \mathcal{N}_i} e^{\beta(S_{ik} - \lambda)} \right] \right\}$$

donde $S_{ik} = \langle f_\theta(x_i), f_\theta(x_k) \rangle$, $\mathcal{P}_i$ = positivos, $\mathcal{N}_i$ = negativos, $\alpha=2, \beta=50, \lambda=0.5$.

| Aspecto | Triplet Loss | Multi-Similarity Loss |
|---------|-------------|----------------------|
| Relaciones/batch | $O(n^3)$ tripletas | Todas: $O(n^2)$ |
| Mining | Externo (semi-hard) | Integrado |
| Gradientes | Sparse | Dense |
| Convergencia | Lenta | Rápida |

**Losses adicionales soportadas:** Triplet Margin, Proxy Anchor (CVPR 2020), ArcFace (CVPR 2019), SG-Softmax (AAAI 2023).

### Muestreo: M-Per-Class Balanced Sampler

```
P=6 clases/batch × K=12 muestras/clase = 72 imágenes/batch

Relaciones por batch:
  Pares positivos: K(K-1)/2 × P = 66 × 6 = 396
  Pares negativos: K² × P(P-1)/2 = 144 × 15 = 2160
  → MS Loss explota TODOS estos pares → gradientes densos
```

### Configuración de Optimización

| Componente | Técnica | Parámetros |
|-----------|---------|------------|
| **Optimizer** | AdamW | lr=10⁻⁴, weight_decay=5×10⁻⁴ |
| **Scheduler** | Cosine Annealing | T_max=30, η_min=10⁻⁶ |
| **Warmup** | Linear ramp | 3 epochs desde 10⁻⁵ |
| **Mixed Precision** | AMP (float16/32) | ~50% menos VRAM, ~30% más rápido |
| **Gradient Clipping** | Max norm | 5.0 |
| **Early Stopping** | R@1-based | Patience=10, restore best |

### Regularización SOTA (2022–2024)

| Técnica | Paper | Efecto |
|---------|-------|--------|
| **NIR** | Roth et al., CVPR 2022 | Anti-colapso isotrópico per-class |
| **HIER** | Kim et al., CVPR 2023 | Árbol jerárquico de proxies |
| **DADA** | Ren et al., AAAI 2024 | MMD entre embeddings originales/augmentados |
| **Self-Distillation** | Zeng et al., 2024 | EMA Teacher + KL sobre similitudes |

Todas opcionales, activables desde `config.yaml` sección `sota:`.

### Artefactos de Entrenamiento

| Artefacto | Descripción |
|-----------|-------------|
| `class_centroids.pt` | Centroides de clase (media de embeddings del train set) |
| `reference_embeddings.pt` | Todos los embeddings del train set + labels |
| `training_metrics.csv` | Métricas epoch-a-epoch |
| `best_model.pth` | Checkpoint con configuración de entrenamiento incorporada |

**Principio:** la evaluación nunca recalcula productos de la fase de entrenamiento.

---

## Etapa 7 — Evaluación y Métricas

| Métrica | Definición |
|---------|-----------|
| **Recall@K** | Fracción de queries con ≥1 vecino correcto en top-K (Wang et al., CVPR 2019) |
| **NMI** | Calidad de clustering (K-Means sobre embeddings) |
| **Distance Ratio** | $\rho = \bar{d}_{inter} / \bar{d}_{intra}$ — separación relativa (objetivo: $\rho > 3$) |
| **kNN Confusion Matrix** | Leave-one-out sobre embeddings de validación |
| **Centroid Classification** | $\hat{y} = \arg\min_c \|f_\theta(x) - \hat{\mu}_c\|_2$ |

---

## Etapa 8 — Análisis Post-Entrenamiento

Pipeline automatizado de **9 etapas** sobre el mejor modelo:

| # | Etapa | Output |
|---|-------|--------|
| 1 | Training curves (loss, recall, lr) | `loss_curves.png`, `recall_curves.png` |
| 2 | t-SNE 2D + 3D de embeddings | `tsne_2d.png`, `tsne_3d.png` |
| 3 | Confusion matrix (kNN) | `confusion_matrix.png` |
| 4 | Distribución distancias intra/inter | `distance_distribution.png` |
| 5 | Inference preview | `inference_preview.png` |
| 6 | **Attention heatmap gallery** | `gradcam_gallery/*.png` |
| 7 | Galería inferencia por clase | `inference_gallery/*.png` |
| 8 | Full-image inference (test) | `fullimage_inference/*.png` |
| 9 | Reporte markdown automático | `training_report.md` |

---

## Etapa 9 — Inferencia End-to-End

`FullImageClassifier`: pipeline completo desde imagen de cámara hasta conteo por especie.

```
Imagen cámara (cualquier resolución)
  │
  ├── PollenGrainDetector (U²-Net) → N granos {bbox, contorno, saliency}
  │
  └── Para cada grano:
        Crop + padding (20%) → Máscara contorno → Gray bg (128)
        → Val transform (Resize 252, Normalize) → AnalogyNet → ê ∈ S^127
        → ŷ = argmin_c ‖ê − μ_c‖₂ + distancia + confianza
  │
  └── Resultado:
        • Imagen anotada: bboxes coloreados por clase + labels
        • Conteo: {especie → cantidad}
        • Heatmaps de atención opcionales por grano
```

---

## Etapa 10 — Explicabilidad (Atención DINOv2)

### CLS-to-Patch Self-Attention

Los mapas de auto-atención de DINOv2 naturalmente segmentan el objeto de interés (Caron et al., ICCV 2021). Método correcto para ViTs (GradCAM fue diseñado para CNNs):

```
Hook en blocks[-1].attn.qkv → Extraer Q, K
  → A = softmax(QK^T / √d_k)         (B, 6 heads, 325, 325)
  → A_cls = A[:, :, 0, 1:]            CLS → 324 patches
  → Promedio sobre heads → (18, 18)
  → Interpolación bicúbica → (252, 252)
  → Heatmap [0,1]: rojo=grano, azul=fondo
```

### Gradient-Weighted Attention (Task-Specific)

Para metric learning, se pondera cada attention head por su contribución a la clasificación (Chefer et al., CVPR 2021):

```
Forward con gradientes → cos_sim(embedding, centroide_predicho) → backward
  → Importancia por head: h_j = mean(|∂sim/∂QKV_j|)
  → Atención ponderada: A_weighted = Σ_j (h_j · A_cls_j)
  → Muestra qué patches contribuyen a la CLASIFICACIÓN CORRECTA
```

### GradCAM vs Atención Nativa

| Propiedad | GradCAM (CNNs) | Atención DINOv2 |
|-----------|---------------|-----------------|
| Diseñado para | CNNs (Selvaraju, 2017) | **Transformers (Caron, 2021)** |
| Fiabilidad en ViTs | Baja (difuso/errores) | **Alta (propiedad emergente)** |
| Dependencia | pytorch_grad_cam | **Solo PyTorch hooks** |
| Task-specific | Sí | **Sí (gradient weighting)** |

El sistema auto-detecta el tipo de backbone y selecciona el método correcto automáticamente.

---

### Resultados Experimentales

El sistema ha sido validado con múltiples configuraciones de backbone (ResNet, ConvNeXt, DINOv2) y funciones de pérdida (Triplet, Multi-Similarity, ArcFace). Los mejores resultados se obtienen con:

- **Backbone:** DINOv2 ViT-S/14 (21M parámetros)
- **Embedding:** 128 dimensiones (hiperesfera S^127)
- **Loss:** Multi-Similarity Loss
- **Augmentaciones:** Gray background (128) + máscara de contorno

Las métricas de evaluación incluyen Recall@K, NMI, mAP@R y F1-Score Macro siguiendo protocolos estándar de Deep Metric Learning (CUB-200, Stanford Online Products).

---

## Arquitectura de Software

```
MeliVision/
├── app.py                              # Punto de entrada (GUI / CLI)
├── config.yaml                         # Configuración declarativa completa
│
├── src/                                # Núcleo científico
│   ├── models/
│   │   ├── analogy_net.py              # AnalogyNet: backbone + projection + L2
│   │   ├── hyperbolic.py               # Poincaré Ball (embedding hiperbólico)
│   │   └── u2net/                      # U²-Net para saliencia
│   │       └── u2net.py                # Arquitectura U²-Net / U²-NETP
│   │
│   ├── grain_detection/                # Pipeline de detección
│   │   ├── salient_detector.py         # Wrapper U²-Net (SOD)
│   │   ├── grain_detector.py           # PollenGrainDetector (end-to-end)
│   │   ├── seg_format.py              # Parser/Writer formato .seg
│   │   ├── annotator.py               # SegmentationAnnotator (batch)
│   │   ├── detected_grain.py          # Dataclass DetectedGrain
│   │   ├── contour_registry.py        # Registro de contornos
│   │   └── full_image_classifier.py   # FullImageClassifier (inferencia E2E)
│   │
│   ├── data/
│   │   ├── grain_dataset.py            # SegmentedGrainDataset (on-the-fly)
│   │   ├── dataset.py                  # MielDataset (legacy ImageFolder)
│   │   └── transforms_enhanced.py      # Augmentaciones SOTA (Albumentations)
│   │
│   ├── training/
│   │   ├── trainer_enhanced.py         # Trainer: AMP, mining, EMA, logging
│   │   └── ema_teacher.py             # Self-Distillation EMA (2024)
│   │
│   ├── losses/
│   │   ├── nir_regularizer.py          # NIR (CVPR 2022)
│   │   ├── hier_regularizer.py         # HIER (CVPR 2023)
│   │   ├── sg_softmax.py              # SG-Softmax (AAAI 2023)
│   │   └── dada_wrapper.py            # DADA (AAAI 2024)
│   │
│   ├── inference/
│   │   ├── knn_classifier.py           # Clasificador k-NN
│   │   └── faiss_index.py             # Índice FAISS
│   │
│   ├── utils/
│   │   ├── inference_utils.py          # Attention maps, GradCAM, embeddings
│   │   └── metrics_utils.py            # R@K, NMI, confusion matrix
│   │
│   └── visualization/                  # Herramientas de visualización
│
├── scripts/
│   ├── train.py                        # Entrenamiento completo
│   └── post_training.py               # Análisis post-entrenamiento (9 etapas)
│
├── lib/                                # Interfaz gráfica (PyQt5)
│   ├── gui_app.py                      # Ventana principal (5 tabs)
│   ├── config_editor.py               # Tab: Editor visual de config.yaml
│   ├── dataset_manager_gui.py         # Tab: Gestor de datos (split)
│   ├── contour_analysis_tab.py        # Tab: Análisis de contornos (U²-Net)
│   ├── inference_tab.py               # Tab: Inferencia (individual/batch)
│   └── styles.py                      # Sistema de estilos
│
├── models/weights/                     # Pesos U²-Net (auto-download)
├── data/processed/{train,val,test}/    # Datos particionados (symlinks)
├── runs/                               # Resultados por experimento
├── docs/                               # Documentación técnica
└── tests/                              # Tests unitarios (pytest)
```

### Stack Tecnológico

| Capa | Tecnología | Función |
|------|-----------|---------|
| **Deep Learning** | PyTorch ≥ 2.0 | Framework base |
| **Metric Learning** | pytorch-metric-learning ≥ 2.0 | Losses, miners, samplers |
| **Vision Transformer** | DINOv2 (torch.hub) | Backbone ViT-S/14 |
| **Object Detection** | U²-Net (custom) | Detección de granos (SOD) |
| **Augmentación** | Albumentations 2.0 | Pipeline de augmentación |
| **Búsqueda vectorial** | FAISS | Índices para k-NN |
| **Visualización** | matplotlib, seaborn | Gráficos y heatmaps |
| **Reducción dim.** | scikit-learn (t-SNE) | Visualización embeddings |
| **GUI** | PyQt5 | Interfaz gráfica |
| **Procesamiento** | OpenCV, Pillow | I/O y procesamiento imagen |
| **Testing** | pytest | Tests unitarios |

### Interfaz Gráfica (GUI)

5 tabs: **Configuración** (editor visual YAML) → **Gestor de Datos** (split train/val/test) → **Análisis de Contornos** (explorador U²-Net, click→crop) → **Entrenamiento** (asíncrono, progress bar) → **Inferencia** (individual/batch desde checkpoint).

---

## Referencias

1. **Qin, X. et al.** (2020). U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection. *Pattern Recognition*, 106.
2. **Oquab, M. et al.** (2023). DINOv2: Learning Robust Visual Features without Supervision. *arXiv:2304.07193*.
3. **Caron, M. et al.** (2021). Emerging Properties in Self-Supervised Vision Transformers. *ICCV 2021*.
4. **Wang, X. et al.** (2019). Multi-Similarity Loss with General Pair Weighting for DML. *CVPR 2019*.
5. **Chefer, H. et al.** (2021). Transformer Interpretability Beyond Attention Visualization. *CVPR 2021*.
6. **Dosovitskiy, A. et al.** (2021). An Image is Worth 16×16 Words. *ICLR 2021*.
7. **Touvron, H. et al.** (2022). DeiT III: Revenge of the ViT. *ECCV 2022*.
8. **Selvaraju, R. et al.** (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV 2017*.
9. **Louveaux, J. et al.** (1978). Methods of Melissopalynology. *Bee World*, 59(4).
10. **Sevillano, V. et al.** (2020). Precise Pollen Classification with Deep Learning. *PLOS ONE*.
11. **Musgrave, K. et al.** (2020). A Metric Learning Reality Check. *ECCV 2020*.
12. **Roth, K. et al.** (2022). Non-Isotropy Regularization for Proxy-based DML. *CVPR 2022*.
13. **Kim, S. et al.** (2023). HIER: Metric Learning Beyond Class-Level Similarities. *CVPR 2023*.

---

**Versión:** 4.0 (DINOv2 + U²-Net + Attention Maps)  
**Última actualización:** 2026-02-10  
**Metodología completa:** [`docs/20260210_2135_METODOLOGIA_COMPLETA_SISTEMA.md`](docs/20260210_2135_METODOLOGIA_COMPLETA_SISTEMA.md)  
**Mejoras atención + augmentations:** [`docs/20260210_1945_ANALISIS_MEJORAS_ATENCION_Y_AUGMENTATIONS.md`](docs/20260210_1945_ANALISIS_MEJORAS_ATENCION_Y_AUGMENTATIONS.md)  
**Documentación técnica:** [`docs/20260207_155700_DESCRIPCION_TECNICA_SISTEMA.md`](docs/20260207_155700_DESCRIPCION_TECNICA_SISTEMA.md)  
**Plan de mejoras SOTA:** [`docs/20260207_195500_PLAN_MEJORAS_SOTA.md`](docs/20260207_195500_PLAN_MEJORAS_SOTA.md)  
**Implementación SOTA:** [`docs/20260207_201600_IMPLEMENTACION_MEJORAS_SOTA.md`](docs/20260207_201600_IMPLEMENTACION_MEJORAS_SOTA.md)  
**Estudio SOTA detallado:** [`docs/20260207_202600_ESTUDIO_SOTA_DETALLADO.md`](docs/20260207_202600_ESTUDIO_SOTA_DETALLADO.md)
