# RSHazeNet: Deep Learning Model Architecture Detailed Breakdown

RSHazeNet is a state-of-the-art vision-language-guided deep learning model specifically designed for remote sensing image dehazing. It is built as a hierarchical **U-Net** style architecture that combines multi-scale spatial feature extraction with cross-modal text-image conditioning.

Here is a comprehensive breakdown of all the layers, modules, and physical formulations used in the RSHazeNet architecture.

---

## 1. Overall Architecture Structure
The model follows an encoder-bottleneck-decoder structure, scaling the spatial dimensions and channel capacities across three main depth levels (by default, configured as 2, 3, and 4 blocks per level).

- **Encoder**: Extracts features while progressively reducing spatial resolution and increasing channel depth.
- **Bottleneck**: The deepest representation of the image, where semantic text guidance (from a Vision-Language Model caption) is injected.
- **Decoder**: Gradually upsamples the features back to the original resolution, fusing them with encoder features via skip connections.
- **Output Layer**: A final formulation that predicts parameters for a physical atmospheric scattering model rather than predicting raw pixel colors directly.

---

## 2. Core Modules & Layers

### A. Overlap Patch Embedding (`OverlapPatchEmbed`)
Instead of projecting raw pixels directly, the model uses an overlapping 3x3 Convolution (with `reflect` padding) to map the 3-channel (RGB) input image into an initial high-dimensional feature space (default: 32 channels). This preserves local spatial structures better than non-overlapping linear projections (like standard ViT patch embeddings).

### B. Basic Block (`BasicBlock`)
This is the primary workhorse module for intra-stage feature extraction, designed to capture features at multiple receptive fields simultaneously.
- **Multi-Scale Convolution**: The input channel dimension is split evenly into 4 groups. Each group is processed by a different parallel convolution:
  - 1x1 Convolution (local details)
  - 3x3 Dilated Convolution (dilation=3)
  - 5x5 Dilated Convolution (dilation=3)
  - 7x7 Dilated Convolution (dilation=3)
- **Concatenation & LayerScale**: The 4 groups are concatenated back together. A learnable scalar parameter (`LayerScale`) multiplies the features to improve training stability for deep networks.
- **MLP (Multi-Layer Perceptron)**: A sequence of two 1x1 Convolutions separated by a ReLU activation refines the concatenated features before they are added back to the original input via a residual connection.

### C. Cross-stage Multi-scale Interaction Module (`CMFI`)
Located in the encoder, this module fuses information between consecutive scales (e.g., Level 1 and Level 2) before they pass deeper into the network. 
- It uses a specialized **spatial cross-attention mechanism**. 
- The shallower (higher-resolution) features act as **Queries**, and the deeper (lower-resolution) features act as **Keys**.
- The attention map is calculated via matrix multiplication, softmax normalized, and applied symmetrically to the **Values** of both stages. This allows fine details from shallow layers to enrich deep layers, while deep semantic contexts guide the shallow layers.

### D. Downsampling & Upsampling
- **`Downsample`**: A 3x3 Convolution followed by a `PixelUnshuffle` operation. This reduces the spatial resolution by half (H/2, W/2) while multiplying the channel count by 2.
- **`Upsample`**: A 3x3 Convolution followed by a `PixelShuffle` operation. This increases the spatial resolution by 2x (H*2, W*2) while halving the channel count, allowing smooth feature map expansion.

### E. Intra-stage Transposed Fusion Module (`IFTE`)
Located in the decoder, this module replaces standard concatenation for merging encoder skip-connections with the upsampled decoder features.
- It uses **Channel-wise (Transposed) Attention**.
- The encoder and decoder features are concatenated, passed through an Adaptive Average Pool, and then split into Queries and Keys via 1x1 Convolutions.
- By computing attention across the *channel* dimension (rather than spatial), the model dynamically selects and weights the most important feature maps from both the encoder and decoder to synthesize the final high-resolution feature.

---

## 3. Cross-Modal Text Conditioning

RSHazeNet's most unique feature is how it uses textual descriptions of haziness to guide the image restoration.

### A. Caption Text Encoder (`CaptionTextEncoder`)
Before the image enters the model, an offline Vision-Language Model (like Qwen2-VL) generates a text caption describing the fog density (e.g., *"Heavily hazy. The entire scene is covered by a thick white fog..."*).
- This text is passed through a **frozen CLIP Text Encoder** (`openai/clip-vit-base-patch32`).
- The text is converted into sequence tokens of shape `(Batch, Tokens, 512)`. 

### B. Cross Attention Conditioning (`CrossAttentionCond`)
Located exactly at the deepest bottleneck of the U-Net.
- **Query**: The flattened latent visual features from the image.
- **Keys / Values**: The 512-dimensional CLIP text tokens.
- **Mechanism**: Standard Multi-Head Attention (MHA) calculates how each visual pixel should be modulated based on the semantic text tokens. This means if the text says "heavy haze", the attention map universally shifts the visual features to trigger aggressive dehazing filters. It is followed by an FFN (Feed-Forward Network) and added residually to the visual bottleneck.

---

## 4. Physics-Based Output Formulation

Most Deep Learning models directly predict the output RGB image (e.g., using a final 3x3 conv). RSHazeNet instead takes inspiration from the **Atmospheric Scattering Model**, which models hazy images as:
> *Input = Clear_Image * Transmission + Atmospheric_Light * (1 - Transmission)*

To reverse this, the final layer (`output_level_1`, a 3x3 Conv) outputs **4 channels**:
1. **`K`** (1 channel): Represents the inverse transmission map (scale factor).
2. **`B`** (3 channels): Represents the additive atmospheric bias.

The final clear image is calculated deterministically inside the network using the formula:
```python
# K * Input_Image - Bias + Input_Image
Output = K * Input - B + Input
```
This forces the neural network to learn the physical properties of the fog rather than just painting over pixels, resulting in much higher color fidelity and reduced artifacts.


# Adapt RSHazeNet Notebook from Colab to Kaggle

The notebook is currently written for Google Colab. We need to adapt it to run on Kaggle with both SateHaze1k and RRSHID datasets across all fog levels.

## Resolved Questions

| Question | Answer |
|---|---|
| SateHaze1k Kaggle slug | `xuxingxing233/satehaze1k` → `/kaggle/input/satehaze1k/` |
| Validation split (SateHaze1k) | Use `test` split as validation (no separate `valid` split) |
| Fog levels | Train on **all** levels: thin, moderate, thick (both datasets) |
| RRSHID source | Download from GitHub release inside notebook |

## Proposed Changes

### 1. [MODIFY] [RSHazeNet.ipynb](file:///media/mxData/Documents/reps/RSIDehazing/RSHazeNet/RSHazeNet.ipynb)

Complete rewrite of notebook cells for Kaggle:

**Cell 1** – Install dependencies (keep as-is, works on Kaggle)

**Cell 2** – Setup (replaces Google Drive mount)
- Create output dirs under `/kaggle/working/RSHazeNet_Results/`
- `git clone` repo into `/kaggle/working/RSHazeNet/` and `%cd` there

**Cell 3** – SateHaze1k paths (replaces Dropbox download)
- Reference Kaggle input: `/kaggle/input/satehaze1k/`
- List directory structure to confirm layout

**Cell 3b** – Download RRSHID (NEW)
- `wget` from GitHub release: `https://github.com/lwCVer/RRSHID/releases/download/dataset/RRSHID.zip`
- Unzip to `/kaggle/working/RRSHID/`
- Auto-detect GT folder name (`clear`, `gt`, or `GT`)

**Cell 4** – VLM captioning (all splits, all datasets)
- Run `caption.py` on all fog levels for both datasets
- Use improved prompt (see section below)

**Cell 5** – Download pretrained weights (keep `gdown`)

**Cell 6** – FLOPs computation (no changes)

**Cells 7-8** – Pretrained inference + evaluation (for each dataset/level combo)

**Cells 9** – Training cells for each dataset/fog level:
- SateHaze1k: thin, moderate, thick (using test as val)
- RRSHID: thin_fog, moderate_fog, thick_fog (has val split)

**Cells 10-11** – Trained model inference + evaluation

**Cell 12** – Resume training cell

#### Path Mappings

**SateHaze1k** (from Kaggle input, read-only):
| Split | Hazy | GT |
|---|---|---|
| train | `/kaggle/input/satehaze1k/Haze1k_{level}/train/hazy/` | `/kaggle/input/satehaze1k/Haze1k_{level}/train/GT/` |
| test/val | `/kaggle/input/satehaze1k/Haze1k_{level}/test/hazy/` | `/kaggle/input/satehaze1k/Haze1k_{level}/test/GT/` |

**RRSHID** (downloaded to working dir):
| Split | Hazy | GT |
|---|---|---|
| train | `/kaggle/working/RRSHID/{level}/train/hazy/` | `/kaggle/working/RRSHID/{level}/train/{gt_name}/` |
| val | `/kaggle/working/RRSHID/{level}/val/hazy/` | `/kaggle/working/RRSHID/{level}/val/{gt_name}/` |
| test | `/kaggle/working/RRSHID/{level}/test/hazy/` | `/kaggle/working/RRSHID/{level}/test/{gt_name}/` |

**Output**: `/kaggle/working/RSHazeNet_Results/`

---

### 2. [MODIFY] [caption.py](file:///media/mxData/Documents/reps/RSIDehazing/RSHazeNet/caption.py)

Improve the VLM prompt for better remote sensing haze classification. The current prompt produces level 2 (moderately hazy) for thick-haze images because VLMs calibrate against natural photos where "heavily hazy" means near-zero visibility. In RS imagery, even thick haze still shows some ground features.

**Current prompt** (too generic):
```
"You are an expert analyzing a remote sensing satellite or aerial image..."
"Use one of these levels: clear, lightly hazy, moderately hazy, heavily hazy."
```

**Improved prompt** (RS-calibrated with explicit criteria):
```
"You are a remote sensing imagery analyst specializing in atmospheric conditions in satellite and aerial photographs.
Classify the haze/fog density in this image using these exact criteria:

- HEAVILY HAZY (level 3): Colors are strongly washed out or whitened. Ground features (roads, buildings, fields) are barely distinguishable or blurred. The overall scene looks milky, foggy, or covered by a thick white/gray layer. Contrast is very low.
- MODERATELY HAZY (level 2): Features are visible but noticeably degraded. Colors are muted. There is an obvious atmospheric veil reducing contrast and sharpness. Fine details like building edges or road markings are hard to see.
- LIGHTLY HAZY (level 1): Most features are clearly visible with only a slight atmospheric effect. Colors are slightly desaturated. Contrast is mildly reduced but details are still sharp.
- CLEAR (level 0): Sharp, high-contrast image with vivid colors and no atmospheric degradation.

Reply with EXACTLY ONE sentence: '<level>. <brief description of what you observe>'.
For example: 'Heavily hazy. The entire scene is covered by a thick white fog layer with barely visible ground features and extremely low contrast.'"
```

Key improvements:
- Explicit visual criteria per level (not just labels)
- RS-specific vocabulary (ground features, atmospheric veil, whitened)
- Example output to anchor the VLM response format
- Criteria calibrated to satellite imagery (where even "thick" haze shows some ground)

> [!NOTE]
> **Data augmentation**: Yes, the model already performs 8-way augmentation during training in [datasets.py](file:///media/mxData/Documents/reps/RSIDehazing/RSHazeNet/datasets.py#L78-L98): random patch cropping + random flip (H/V) + random rotation (90°/180°/270°) + flip+rotation combos.

## Verification Plan

### Manual Verification
- Validate notebook JSON syntax
- Verify all Kaggle paths are correct
- Ensure `git clone` + `%cd` approach provides working module imports
- Cross-check dataset folder structures against `data.md`
