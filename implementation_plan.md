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
