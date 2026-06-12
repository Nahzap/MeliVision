# MeliVision

Sistema de **microscopía automatizada** para clasificación de granos de polen mediante **Deep Metric Learning (DML)** con **DINOv2 Vision Transformer**.

- **Localización:** U²-Net (SOD) + filtros morfológicos de polen — preprocesamiento `.seg` e inferencia full-image.
- **DML:** AnalogyNet (ViT entrenado) aprende embeddings 128D en $\mathbb{S}^{127}$ y clasifica por slice.
- **Inferencia cámara:** `PollenGrainDetector` localiza → `FullImageClassifier` + best model clasifica.

---

## Tabla de Contenidos

- [Contexto Científico](#contexto-científico)
- [Pipeline Completo del Sistema](#pipeline-completo-del-sistema)
- [Etapa 1 — Adquisición Óptica](#etapa-1--adquisición-óptica)
- [Etapa 2 — Detección de Granos (U²-Net)](#etapa-2--detección-de-granos-u²-net)
- [Etapa 3 — Formato .seg y Dataset On-The-Fly](#etapa-3--formato-seg-y-dataset-on-the-fly)
- [Etapa 4 — Preprocesamiento y Augmentaciones](#etapa-4--preprocesamiento-y-augmentaciones)
- [Etapa 5 — Modelo de Embedding (AnalogyNet + DINOv2)](#etapa-5--modelo-de-embedding-analogynet--dinov2)
- [Etapa 6 — Entrenamiento (Sliced MS Loss)](#etapa-6--entrenamiento-sliced-ms-loss)
- [Configuración declarativa y plan de alineación](#configuración-declarativa-y-plan-de-alineación)
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
║  ETAPA 2: U²-Net (SOLO preprocesamiento → .seg)                  ║
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
║  │  │  Capas 3+11 → masked multi-spec    │  │                     ║
║  │  │  (mean+max+std) → ℝ^1152 × 2       │  │                     ║
║  │  ├─────────────────────────────────────┤  │                     ║
║  │  │ Projection LF/HF → 64+64 + L2-Cat  │  │                     ║
║  │  ├─────────────────────────────────────┤  │                     ║
║  │  │ Embedding ê ∈ S^127 (128D)         │  │                     ║
║  │  └─────────────────────────────────────┘  │                     ║
║  └─────────┬────────────────────────────────┘                     ║
║            │                                                      ║
║            ▼                                                      ║
║  ┌──────────────────────────────────────────┐                     ║
║  │ ETAPA 6-8: Sliced MS + evaluación MLRC + post-training           ║
║  │ ETAPA 9: Inferencia — U²-Net localiza → ViT clasifica            ║
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

> **Rol:** U²-Net localiza granos (SOD) en preprocesamiento (`.seg`) **y** en inferencia full-image.  
> La **clasificación de especie** la hace el best model (AnalogyNet + slice-aware), no U²-Net.

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

### DINOv2 ViT-S/14 Backbone + pooling enmascarado

**DINOv2** (Oquab et al., 2023): foundation model SSL sobre LVD-142M. El proyecto **no usa el token CLS** para metric learning; agrega **patch tokens** de capas intermedias con estadísticas multi-espectrales y máscara del contorno.

```
Imagen (3 × H × W)  +  máscara alineada con augmentations
    │
    ▼
┌────────────────────────────────────────┐
│ Patch Embedding (14×14) + 12 bloques   │
│  Self-attention GLOBAL en cada capa    │
│  (fondo gris ≈ 0 tras norm ImageNet)   │
├────────────────────────────────────────┤
│ get_intermediate_layers(n=[3, 11])     │
│  capa 3  → textura / baja semántica    │
│  capa 11 → semántica / pre-final      │
├────────────────────────────────────────┤
│ Masked multi-spectral pooling          │
│  por capa: mean + max + std            │
│  solo patches con máscara > umbral       │
│  DropToken (train, p configurable)      │
│  → feat_lf, feat_hf ∈ ℝ^(3×384)=ℝ^1152 │
├────────────────────────────────────────┤
│ projection_lf / projection_hf (MLP)    │
│  1152 → 256 → 256 → 64 cada rama       │
├────────────────────────────────────────┤
│ L2-Cat-L2 → ê ∈ S^127 (128D)          │
└────────────────────────────────────────┘
    │
    ▼
Sliced MS Loss particiona 128D en S slices (default 4×32D)
```

**Nota:** la imagen **no** se multiplica por la máscara antes del ViT (artefactos posicionales). El filtrado es **post-atención**, en el pooling. Ver auditoría en `docs/20260210_1750_AUDITORIA_REGRESION_MASKED_POOLING.md`.

### Relación pooling ↔ slices espectrales

| Etapa | Qué hace | Efecto en capacidad discriminativa |
|-------|----------|-----------------------------------|
| **Pooling** | Comprime N patches → vector rico (mean/max/std, 2 capas) | Preserva morfología intra-grano (textura, heterogeneidad) |
| **Proyección** | Comprime 1152D → 128D | Aprende qué features entran al espacio métrico |
| **S slices** | Particiona 128D; MS+LDM independiente por slice | Eleva rango espectral efectivo ≈ S×(C−1) vs. colapso C−1 |

Los slices N **no sustituyen** un buen pooling: amplifican la geometría métrica **si** las dimensiones de cada slice corresponden a fuentes de señal coherentes. El [plan de implementación](docs/2026-06-11_PLAN_IMPLEMENTACION_POOLING_SLICE_ALIGNED.md) define el layout `slice_native` para alinear capas, pooling y partición S.

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
| **DINOv2 ViT-S/14** | **1152** (post-pool) / 384 (token) | **21M** | **torch.hub (default)** |
| DINOv2 ViT-B/14 | 768 | 86M | torch.hub |
| DINOv2 ViT-S/14 + Reg | 384 | 21M | torch.hub (registers) |
| DINOv2 ViT-B/14 + Reg | 768 | 86M | torch.hub (registers) |

**Espacios de embedding:** Euclidiano ($\mathbb{S}^{d-1}$) o Hiperbólico (bola de Poincaré, Ermolov et al., CVPR 2022).

---

## Etapa 6 — Entrenamiento (Sliced MS Loss)

El entrenamiento prioriza pares morfológicamente difíciles mediante **soft mining integrado** en cada slice (véase [Hard Example Mining → Soft Mining integrado](#hard-example-mining--soft-mining-integrado-en-sliced-ms)); los mineros explícitos (OHEM / triplet) quedan como legacy para `arcface`/`triplet`.

### Función de Pérdida (default: `sliced_ms`)

El pipeline actual usa **Sliced Multi-Similarity Loss con Learnable Dynamic Margins** (LDM-MS por subespacio). Particiona el embedding 128D en **S=4 subespacios disjuntos de 32D** y aplica MS Loss (Wang et al., CVPR 2019) de forma independiente en cada slice, con márgenes aprendibles por clase.

Motivación: mitigar **Neural Collapse** (Papyan et al., PNAS 2020) y elevar el rango espectral efectivo del espacio de embedding (Ko & Gu, CVPR 2020; slicing inspirado en Sanakoyeu et al., CVPR 2019).

$$\mathcal{L}_{\text{sliced}} = \frac{1}{S} \sum_{s=1}^{S} \mathcal{L}_{MS}^{(s)}(\text{normalize}(\mathbf{z}^{(s)}), y)$$

donde $\mathbf{z}^{(s)}$ es el slice $s$ del embedding y cada slice se L2-normaliza antes de calcular similitud coseno.

| Aspecto | k-NN global 128D | Sliced MS + inferencia slice-aware |
|---------|------------------|----------------------------------|
| Geometría train/infer | Puede divergir | Coherente por slice |
| Colapso dimensional | Tendencia a simplex (C−1) | S simplexes independientes |
| Clasificación inferencia | Centroides / k-NN 128D | Media coseno por slice vs representantes |

**Pérdidas alternativas:** `sliced_proxy` (SG-Softmax por slice), `hybrid_proxy_ms`, Triplet, ArcFace, Proxy Anchor.

Documentación: `docs/2026-06-09_INFORME_MODIFICACIONES_PIPELINE_MULTI_SLICE.md` (slices + inferencia) y `docs/2026-06-11_PLAN_IMPLEMENTACION_POOLING_SLICE_ALIGNED.md` (pooling configurable + alineación slice-native).

### Multi-Similarity Loss (base por slice)

**Multi-Similarity Loss** (Wang et al., CVPR 2019) explota simultáneamente relaciones positivas y negativas de cada muestra respecto a todo el batch:

$$\mathcal{L}_{MS} = \frac{1}{m} \sum_{i=1}^{m} \left\{ \frac{1}{\alpha} \log \left[ 1 + \sum_{k \in \mathcal{P}_i} e^{-\alpha(S_{ik} - \lambda)} \right] + \frac{1}{\beta} \log \left[ 1 + \sum_{k \in \mathcal{N}_i} e^{\beta(S_{ik} - \lambda)} \right] \right\}$$

donde $S_{ik} = \langle f_\theta(x_i), f_\theta(x_k) \rangle$, $\mathcal{P}_i$ = positivos, $\mathcal{N}_i$ = negativos, $\alpha=2, \beta=50, \lambda=0.5$.

| Aspecto | Triplet Loss | Multi-Similarity Loss |
|---------|-------------|----------------------|
| Relaciones/batch | $O(n^3)$ tripletas | Todas: $O(n^2)$ |
| Mining | Externo (semi-hard) | Integrado (soft mining) |
| Gradientes | Sparse | Dense |
| Convergencia | Lenta | Rápida |

### Hard Example Mining → Soft Mining integrado en Sliced MS

Una característica planificada desde el inicio del proyecto fue **enfocar el entrenamiento en las diferencias morfológicas más difíciles de separar** — pares positivo/negativo en el borde del margen, clases confusables en melisopalinología. Esa idea se implementó en dos fases:

| Fase | Mecanismo | Loss compatible | Estado |
|------|-----------|-----------------|--------|
| **v1 (mar 2026)** | `HardNegativeMiner` (OHEM) + triplet miners opcionales | `arcface`, `triplet` | Legacy — código en `src/training/hard_mining.py` |
| **v2 (actual)** | **Soft mining** vía LogSumExp de MS Loss + LDM por slice | `sliced_ms` (default) | **Activo** — sustituye al minero externo |

#### v1 — Mineros explícitos (legacy)

Dos bloques en `config.yaml`, ambos **desactivados** con `sliced_ms`:

```yaml
training:
  mining:          # Triplet Mining — semihard / hard / distance_weighted (pytorch_metric_learning)
    enabled: false
  hard_mining:     # OHEM — top-k% muestras con mayor pérdida (Shrivastava et al., CVPR 2016)
    enabled: false
```

- **`training.hard_mining`:** calcula pérdida por muestra, selecciona el top-k% más difícil y promedia solo sobre esos ejemplos. Funcionó en runs ArcFace (mar 2026, ratio 0.3).
- **`training.mining`:** mineros de triplets (`TripletMarginMiner`). Útil con Triplet Loss; con ArcFace se omite automáticamente.

#### v2 — Mining integrado en cada slice (pipeline actual)

Con **`sliced_ms`**, el minero externo **no participa** (OHEM auto-desactivado; triplet mining incompatible con LDM). La detección de pares difíciles ocurre **dentro** de cada subespacio 32D:

1. **Soft mining MS:** el término LogSumExp con $\beta=50$ pondera exponencialmente los **negativos cercanos** (hard negatives implícitos); $\alpha$ modula positivos lejanos. Todos los pares del batch contribuyen — gradientes densos, sin recortar el grafo.
2. **LDM por clase y slice:** `pos_margin` / `neg_margin` aprendibles adaptan la dificultad **por especie** en cada subespacio.
3. **S=4 slices:** la dificultad se distribuye en 4 geometrías independientes → rango espectral ≈ $S \times (C-1)$ vs. colapso a un simplex.

> **Nota de diseño:** usar mineros duros (`indices_tuple`) con MS/LDM **distorsiona** el soft mining y favorece colapso dimensional (`src/losses/ldm_loss.py`). Por eso el mining explícito quedó fuera de ruta al adoptar `sliced_ms`.

**En Methods del paper:** describir como *implicit hard-pair weighting via Multi-Similarity LogSumExp across S orthogonal subspaces with learnable dynamic margins* — no como OHEM ni triplet mining externo.

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
| **Early Stopping** | `validation_metric` (default mAP@R) | Patience=20, restore best |

### Regularización SOTA (2022–2024)

| Técnica | Paper | Estado en trainer |
|---------|-------|-------------------|
| **NIR** | Roth et al., CVPR 2022 | ✅ Opcional |
| **HIER** | Kim et al., CVPR 2023 | ✅ Opcional |
| **Anti-collapse** | Wang & Isola, ICML 2020 | ✅ Opcional (`sota.anti_collapse`) |
| **Self-Distillation** | Zeng et al., 2024 | ✅ Opcional |
| Uniformity / VICReg / disentanglement (GUI) | — | ❌ No implementado — validator rechaza si enabled |

### Artefactos de Entrenamiento

| Artefacto | Descripción |
|-----------|-------------|
| `slice_representatives.pt` | **Primario inferencia** — representantes por slice `(S, C, D_s)` |
| `class_centroids.pt` | Centroides globales (referencia / legacy) |
| `reference_embeddings.pt` | Embeddings del train set + labels |
| `training_metrics.csv` | Métricas epoch-a-epoch |
| `best_model.pth` | Checkpoint con configuración de entrenamiento incorporada |

**Principio:** la evaluación nunca recalcula productos de la fase de entrenamiento.

---

## Configuración declarativa y plan de alineación

### Principio: nada hardcodeado

Toda decisión arquitectónica debe vivir en `config.yaml` y validarse al arranque (`src/utils/config_validator.py`). Valores que **hoy están fijos en código** y pasarán a config:

| Parámetro | Código actual | Destino config |
|-----------|---------------|----------------|
| Capas ViT | `[3, 11]` | `model.pooling.layers` |
| Estadísticas | mean+max+std | `model.pooling.statistics` |
| DropToken | `p=0.3` | `model.pooling.drop_token` |
| Umbral máscara patch | `0.01` | `model.pooling.masked.threshold` |
| Layout proyección | LF/HF dual | `model.projection.layout: slice_native` |
| S slices | `num_slices: 4` | ya en `training.loss.params` |

### Parámetros centrales (hoy)

```yaml
model:
  embedding_dim: 128      # divisible por num_slices
training:
  loss:
    type: sliced_ms
    params:
      num_slices: 4       # S — controla rango espectral
grain_detection:
  localization_detector: u2net
  discovery_mode: false
  masked_pooling: true
```

### Roadmap de implementación (2026-06-11)

| Fase | Entregable | Prioridad |
|------|------------|-----------|
| **0** | Baseline reproducible + inventario hardcodes | Alta |
| **1** | Schema `model.pooling` + validator extendido | Alta |
| **2** | `PoolingModule` + `EmbeddingExtractor` unificado | **Crítica** |
| **3** | Proyección `slice_native` (dims ↔ slices) | Alta |
| **4** | XAI/Grad-CAM mismo forward que train | Media |
| **5** | Matriz de ablaciones (GeM, capas, S, masking) | Media |
| **6** | Migración checkpoints + docs | Baja |

Plan completo con fórmulas, criterios de aceptación y referencias 2022–2026:
**[`docs/2026-06-11_PLAN_IMPLEMENTACION_POOLING_SLICE_ALIGNED.md`](docs/2026-06-11_PLAN_IMPLEMENTACION_POOLING_SLICE_ALIGNED.md)**

Implementación (2026-06-11): **`docs/2026-06-11_IMPLEMENTACION_POOLING_SLICE_CHANGELOG.md`** — smoke test: `python scripts/smoke_test_pooling_slice.py --config test_config.yaml`

### Brechas resueltas (2026-06-12)

- Checkpoint e informe alineados con `validation_metric` (mAP@R).
- `val_loss` de validación usa `prenorm` en sliced_ms.
- Post-training separa métricas slice-aware vs 1-NN LOO.
- Inferencia full-image sin atención ViT como detector (`PollenGrainDetector` / U²-Net).
- Validator rechaza regularizadores no implementados.

Registro completo: [`docs/2026-06-12_17-44-00_REGISTRO_DML_PIPELINE_REFERENCIAS.md`](docs/2026-06-12_17-44-00_REGISTRO_DML_PIPELINE_REFERENCIAS.md)

---

## Etapa 7 — Evaluación y Métricas

| Métrica | Definición | Rol |
|---------|-----------|-----|
| **Checkpoint** | `training.validation_metric` (default **mAP@R**) | Selección del best model (Musgrave et al., ECCV 2020) |
| **R@1, mAP@R, NMI, F1-macro** | 1-NN LOO MLRC en embeddings | Métricas retrieval estándar |
| **slice_accuracy** | Media coseno por slice vs representantes | Clasificador de despliegue (Sanakoyeu et al., CVPR 2019) |
| **effective_rank** | Componentes para 95% varianza en espectro val | Anti-colapso dimensional |
| **val_loss** | P×K auxiliar (alineado con prenorm en sliced_ms) | Curvas; no criterio de checkpoint |

**No mezclar** slice_accuracy con R@1 en tablas del paper — son protocolos distintos.

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

Pipeline en **dos etapas** (localizar → identificar), estándar en microscopía automatizada:

```
Imagen cámara (cualquier resolución)
  │
  ├── PollenGrainDetector (U²-Net SOD + morfología polen) → N granos
  │
  └── Para cada grano:
        Crop + padding → Máscara contorno → fondo imagenet_neutral
        → Val transform → AnalogyNet(mask si masked_pooling) → ê ∈ S^127
        → SliceAwareClassifier vs slice_representatives.pt
  │
  └── Resultado: imagen anotada + conteo por especie + heatmaps opcionales (Grad-CAM del ViT)
```

| Etapa | Módulo | Tarea | ¿Entrena? |
|-------|--------|-------|-----------|
| Localización | `PollenGrainDetector` | Dónde está el grano | No (SOD fijo) |
| Identificación | AnalogyNet + `SliceAwareClassifier` | Qué especie es | Sí (best checkpoint) |

Config (`grain_detection`):

```yaml
localization_detector: u2net   # u2net (default) | vit (legacy: atención ViT)
discovery_mode: false          # true = multi-umbral agresivo (más FP)
adaptive_k: 0.3                # umbral adaptativo saliency
min_area: 1000                 # área mínima grano (px²)
min_circularity: 0.3           # filtro morfológico polen
```

`ModelGrainDetector` (atención ViT) queda como backend legacy (`localization_detector: vit`).

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
Forward con gradientes → cos_sim(embedding, target_slice_aware) → backward
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

El sistema ha sido validado con múltiples configuraciones de backbone (ResNet, ConvNeXt, DINOv2) y funciones de pérdida (Triplet, Multi-Similarity, ArcFace, **Sliced MS**). Los mejores resultados para el estudio actual se obtienen con:

- **Backbone:** DINOv2 ViT-S/14
- **Loss:** `sliced_ms` (4×32D) + LDM
- **Inferencia:** `SliceAwareClassifier` + `FullImageClassifier`

- **Backbone:** DINOv2 ViT-S/14 (21M parámetros)
- **Embedding:** 128 dimensiones (hiperesfera S^127)
- **Loss:** `sliced_ms` (Sliced Multi-Similarity + LDM)
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
│   │   ├── analogy_net.py              # AnalogyNet: backbone + pooling + projection
│   │   ├── pooling.py                  # (planificado) PoolingModule configurable
│   │   ├── embedding_extractor.py      # (planificado) forward canónico único
│   ├── inference/
│   │   └── slice_classifier.py         # SliceAwareClassifier + representantes
│   │   ├── hyperbolic.py               # Poincaré Ball (embedding hiperbólico)
│   │   └── u2net/                      # U²-Net para saliencia
│   │       └── u2net.py                # Arquitectura U²-Net / U²-NETP
│   │
│   ├── grain_detection/                # Pipeline de detección
│   │   ├── salient_detector.py         # Wrapper U²-Net (SOD)
│   │   ├── grain_detector.py           # PollenGrainDetector (localización)
│   │   ├── model_grain_detector.py     # Backend legacy (atención ViT)
│   │   ├── seg_format.py               # Parser/Writer formato .seg
│   │   ├── annotator.py                # SegmentationAnnotator (batch)
│   │   └── full_image_classifier.py    # Pipeline E2E cámara
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
14. **Kornblith, S., Shlens, J., & Le, Q. V.** (2019). Do Better ImageNet Models Transfer Better? *CVPR 2019*.
15. **Macenko, M. et al.** (2009). A Method for Normalizing Histology Slides for Quantitative Analysis. *ISBI 2009*.
16. **Reinhard, E., Ashikhmin, M., Gooch, B., & Shirley, P.** (2001). Color Transfer between Images. *IEEE CG&A*, 21(5).
17. **Loshchilov, I., & Hutter, F.** (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. *ICLR 2017*.
18. **Shrivastava, A. et al.** (2016). Training Region-based Object Detectors with Online Hard Example Mining. *CVPR 2016* (OHEM legacy — `hard_mining`).
19. **Schroff, F. et al.** (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. *CVPR 2015* (triplet mining).

**Auditoría bibliográfica (correcciones Ashikhmin, Macenko, venues):** [`docs/20260207_172200_REVISION_REFERENCIAS_SOTA.md`](docs/20260207_172200_REVISION_REFERENCIAS_SOTA.md)

**Plan cumplimiento académico SOTA:** [`docs/2026-06-12_PLAN_CUMPLIMIENTO_ACADEMICO_SOTA.md`](docs/2026-06-12_PLAN_CUMPLIMIENTO_ACADEMICO_SOTA.md)

**Ablaciones F-01…F-07:** `python scripts/run_ablation_matrix.py --matrix pooling --dry-run`

---

**Versión:** 6.2 (inferencia two-stage: U²-Net localiza, ViT clasifica)  
**Última actualización:** 2026-06-11  
**Registro DML y referencias:** [`docs/2026-06-12_17-44-00_REGISTRO_DML_PIPELINE_REFERENCIAS.md`](docs/2026-06-12_17-44-00_REGISTRO_DML_PIPELINE_REFERENCIAS.md)  
**Plan cumplimiento académico:** [`docs/2026-06-12_PLAN_CUMPLIMIENTO_ACADEMICO_SOTA.md`](docs/2026-06-12_PLAN_CUMPLIMIENTO_ACADEMICO_SOTA.md)
