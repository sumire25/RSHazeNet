###  Encoder-minimal and Decoder-minimal Framework for Remote Sensing Image Dehazing

**Abstract**: Haze obscures remote sensing images, hindering valuable information extraction. To this end, we propose RSHazeNet, an encoder-minimal and decoder-minimal framework for efficient remote sensing image dehazing. Specifically, regarding the process of merging features within the same level, we develop an innovative module called intra-level transposed fusion module (ITFM). This module employs adaptive transposed self-attention to capture comprehensive context-aware information, facilitating the robust context-aware feature fusion. Meanwhile, we present a cross-level multi-view interaction module (CMIM) to enable effective interactions between features from various levels, mitigating the loss of information due to the repeated sampling operations. In addition, we propose a multi-view progressive extraction block (MPEB) that partitions the features into four distinct components and employs convolution with varying kernel sizes, groups, and dilation factors to facilitate view-progressive feature learning. Extensive experiments demonstrate the superiority of our proposed RSHazeNet. We release the source code and all pre-trained models at https://github.com/chdwyb/RSHazeNet.

### News 🚀🚀🚀

- **Oct 17, 2023**: Our new work ([Encoder-free Multi-axis Physics-aware Fusion Network for Remote Sensing Image Dehazing](https://ieeexplore.ieee.org/abstract/document/10287960)) is accepted by IEEE Transactions on Geoscience and Remote Sensing.


### Requirements

```python
python 3.8.6
torch 1.9.0
torchvision 0.11.0
pillow 9.2.0
scikit-image 0.19.3
timm 0.6.7
tqdm 4.64.0
opencv-python 4.5.2.54
```

### VLM-guided dehazing (caption conditioning)

This fork adds an optional **VLM haziness-caption conditioning** path:
a vision-language model captions each hazy image offline (describing the
haze level), those captions are encoded by a frozen CLIP text encoder,
and the tokens condition the network bottleneck via cross-attention.
This lets the model adapt to the haze severity of each image.

Conditioning is **off by default**, so the network behaves exactly like
the original RSHazeNet. Pass `--caption` to enable it.

**Extra dependencies** (install on Colab):
```python
pip install transformers accelerate
```

**Step 1 — caption the dataset (run once per split):**
```python
# Loads Qwen2-VL-2B-Instruct in fp16 (<15 GB VRAM), caches captions.json
python caption.py --input_dir ./Haze1k-thick/train/hazy
python caption.py --input_dir ./Haze1k-thick/val/hazy
python caption.py --input_dir ./Haze1k-thick/test/hazy
```
Idempotent — interrupt and re-run; already-captioned images are skipped.
`captions.json` is written next to the input dir and read automatically
by train/test when `--caption` is set. A remote API can be used instead
of a local VLM: `--api openai --api_key sk-...` (or `--api gemini`).

### Train

If you intend to conduct training on our proposed RSHazeNet using your own datasets, it is imperative to initially ascertain the training and testing paths specified in `options.py`. Specifically, the paths should be provided in the manner illustrated below.

```python
# training
self.Input_Path_Train = './Haze1k-thick/train/hazy/'
self.Target_Path_Train = './Haze1k-thick/train/gt/'
# validation
self.Input_Path_Val = './Haze1k-thick/val/hazy/'
self.Target_Path_Val = './Haze1k-thick/val/gt/'
```

Subsequently, you may attempt the execution of the following command in order to initiate the training process.

```python
python train.py
```

**VLM-conditioned training** (after captioning):
```python
python train.py --caption --haze_weight 0.5
```
Optional `--haze_weight` up-weights the loss for heavier-haze samples.

**Resuming from a checkpoint** (model + optimizer + scheduler + epoch):
```python
python train.py --resume ./model_best.pth
```
The cosine schedule continues from the saved epoch and metrics are
appended to `training_metrics.csv`. Older weights-only files can still
be loaded with `--pretrained` (new conditioning params stay at their
init via `strict=False`).

### Pre-trained models

To facilitate expeditious testing on the datasets utilized in our study, we additionally furnish all the pre-trained models for the purpose of any conceivable verification.

| Dataset           | Link                                                         | Dataset              | Link                                                         |
| ----------------- | ------------------------------------------------------------ | -------------------- | ------------------------------------------------------------ |
| StateHaze1K-thick | [[Baidu Cloud](https://pan.baidu.com/s/1KGz0Lyo6E3mDJBSdDzFfgg), code: rsid]   [[Google Drive](https://drive.google.com/file/d/1Leyg1sw4x48wEo5zsPKBGtVaCF5NWdY_/view?usp=sharing)] | StateHaze1K-moderate | [[Baidu Cloud](https://pan.baidu.com/s/1cyzLZFK0-pX-uyUC3yTAXQ), code: rsid]   [[Google Drive](https://drive.google.com/file/d/1Jxz0ZpMUAFYP-4nYS4__HyxR506_kF7d/view?usp=sharing)] |
| StateHaze1K-thin  | [[Baidu Cloud](https://pan.baidu.com/s/16rhnMKq47mqlgZ5hYE_bLw), code: rsid]   [[Google Drive](https://drive.google.com/file/d/15FeoHGhfRSk22zWzWfH96mHIjjnoHb-z/view?usp=sharing)] | RS-Haze              | [[Baidu Cloud](https://pan.baidu.com/s/11CQE01WXtxGX9bdigkiMJg), code: rsid]   [[Google Drive](https://drive.google.com/file/d/1-UD3eJeAULBB4mzf3SvNMdzAvuSLFK9G/view?usp=sharing)] |
| LHID              | [[Baidu Cloud](https://pan.baidu.com/s/1rF3eYJ6f7s5mVmIO9NriQg), code: rsid]   [[Google Drive](https://drive.google.com/file/d/1L0yDz4aP5NfNetqSmHC1CV7GiL45b8nK/view?usp=sharing)] | DHID                 | [[Baidu Cloud](https://pan.baidu.com/s/1n9uh8daqVDmFkkr3skQw9A), code: rsid]   [[Google Drive](https://drive.google.com/file/d/14RSfxdepbhaLBmPfryzCc1erGn1cjrOn/view?usp=sharing)] |
| RICE              | [[Baidu Cloud](https://pan.baidu.com/s/1IdaugM5MrxH8QMFeT6sy4g), code: rsid]   [[Google Drive](https://drive.google.com/file/d/1FsABBJRbUA0mJbzsjMlMofI7exe5T82D/view?usp=sharing)] | RSID                 | [[Baidu Cloud](https://pan.baidu.com/s/12cZt2e4p85u2n59pYZ-rvw), code: rsid]   [[Google Drive](https://drive.google.com/file/d/1HozsEo2H49SpRMb0ws1VBULqUevSpboA/view?usp=sharing)] |
| Dense-Haze        | [[Baidu Cloud](https://pan.baidu.com/s/1KM6QneCvYZ_Bh0nCP6Eh5Q), code: rsid]   [[Google Drive](https://drive.google.com/file/d/1h83Gcy9Z4ET1m5Jw-b15HR4h9x4hWk_h/view?usp=sharing)] | NH-Haze              | [[Baidu Cloud](https://pan.baidu.com/s/15pzjMIT2IjzjDlUTIMcZ1g), code: rsid]   [[Google Drive](https://drive.google.com/file/d/13vqNv1SzIt1bxEsSYjl_Y5zAYXKfLbLn/view?usp=sharing)] |

### Test

To assess the effectiveness of our proposed RSHazeNet on your personalized datasets, it is imperative to initially identify the testing paths delineated in the `options.py` file. For instance, the paths should be specified in the following format.

```python
# testing
self.Input_Path_Test = './Haze1k-hick/test/hazy/'
self.Target_Path_Test = './Haze1k-hick/test/gt/'
self.Result_Path_Test = './Haze1k-hick/test/result/'
```

Subsequently, it may be necessary to modify the path of the pre-trained model as illustrated below.

```python
#  pre-trained model
self.MODEL_PRE_PATH = './rs_haze.pth'
```

Now you can proceed with the testing phase and assess the performance of our proposed RSHazeNet.

```pyth
python test.py
```

For **VLM-conditioned testing** (after captioning the test split):
```python
python test.py --caption
```

### Dataset

If you intend to conduct experiments on our collected real-world remote sensing hazy dataset, named RRSD300, please download it from [[Baidu Cloud](https://pan.baidu.com/s/1lM9vEvDwgDrCoyPJAW490A), code: rsid] or [[Google Drive](https://drive.google.com/file/d/198dmAL5Vrw1qm_f5t4nW8l1Jmw-HNLuy/view?usp=sharing)].

### Contact  us

This repository is currently under preparation. If you have any inquiries or questions regarding our work, please feel free to contact us at wyb@chd.edu.cn.
