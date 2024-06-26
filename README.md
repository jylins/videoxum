# VideoXum: Cross-modal Visual and Textural Summarization of Videos

**This repository is the official PyTorch implementation of VideoXum.**

[[Project Page](https://videoxum.github.io/)]  [[Paper](https://arxiv.org/pdf/2303.12060.pdf)] [[Dataset](https://huggingface.co/datasets/jylins/videoxum)] [[Model Zoo](https://huggingface.co/jylins/vtsum_blip)]


## What is cross-modal video summarization?
Cross-modal video summarization is a novel task in the field of video summarization, extending the scope from single-modal to cross-modal video summarization. This new task focuses on creating video summaries that containing both visual and textual elements with semantic coherence.

![videoxum](resources/v2x-sum.png)

## Contents
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Training](#Training)
- [Evaluation](#evaluation)
- [Model Zoo](#model-zoo)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## 1 VideoXum Dataset
The VideoXum dataset represents the novel task: cross-modal video summarization. Built upon the foundation of ActivityNet Captions, VideoXum is a large-scale dataset, including over 14,000 long-duration and open-domain videos. Each video is paired with 10 corresponding video summaries, amounting to a total of 140,000 video-text summary pairs.

Download from Huggingface repository of VideoXum ([link](https://huggingface.co/datasets/jylins/videoxum)).

### 1.1 Dataset Structure

#### 1.1.1 Dataset Splits
|             |train  |validation| test  | Overall |
|-------------|------:|---------:|------:|--------:|
| # of videos | 8,000 |  2,001   | 4,000 | 14,001  |

#### 1.1.2 Dataset Resources
- `train_videoxum.json`: annotations of training set
- `val_videoxum.json`: annotations of validation set
- `test_videoxum.json`: annotations of test set

#### 1.1.3 Dataset Fields
- `video_id`: `str` a unique identifier for the video.
- `duration`: `float` total duration of the video in seconds.
- `sampled_frames`: `int` the number of frames sampled from source video at 1 fps with a uniform sampling schema.
- `timestamps`: `List_float` a list of timestamp pairs, with each pair representing the start and end times of a segment within the video.
- `tsum`: `List_str` each textual video summary provides a summarization of the corresponding video segment as defined by the timestamps.
- `vsum`: `List_float` each visual video summary corresponds to key frames within each video segment as defined by the timestamps. The dimensions (3 x 10) suggest that each video segment was reannotated by 10 different workers.
- `vsum_onehot`: `List_bool` one-hot matrix transformed from 'vsum'. The dimensions (10 x 83) denotes the one-hot labels spanning the entire length of a video, as annotated by 10 workers.

#### 1.1.4 Annotation Sample
For each video, We hire workers to annotate ten shortened video summaries.
``` bash
{
    'video_id': 'v_QOlSCBRmfWY',
    'duration': 82.73,
    'sampled_frames': 83
    'timestamps': [[0.83, 19.86], [17.37, 60.81], [56.26, 79.42]],
    'tsum': ['A young woman is seen standing in a room and leads into her dancing.',
             'The girl dances around the room while the camera captures her movements.',
             'She continues dancing around the room and ends by laying on the floor.'],
    'vsum': [[[ 7.01, 12.37], ...],
             [[41.05, 45.04], ...],
             [[65.74, 69.28], ...]] (3 x 10 dim)
    'vsum_onehot': [[[0,0,0,...,1,1,...], ...],
                    [[0,0,0,...,1,1,...], ...],
                    [[0,0,0,...,1,1,...], ...],] (10 x 83 dim)
}
```

### 1.2 Prepare Dataset

#### 1.2.1 Download source videos from ActivityNet Captions datatset
Please download the source videos from ActivityNet Captions datatset following the instruction from official website ([link](https://cs.stanford.edu/people/ranjaykrishna/densevid/)). Additionally, you can follow the open-source tool to download the source videos from Huggingface ([link](https://huggingface.co/datasets/Leyo/ActivityNet_Captions)).

#### 1.2.2 Download VideoXum dataset from Huggingface
Please download VideoXum dataset from Huggingface ([link](https://huggingface.co/datasets/jylins/videoxum)), including annotations for each video. we provide train/val/test splits.

#### 1.2.3 File Structure of VideoXum dataset
The file structure of VideoXum looks like:
```
dataset
└── ActivityNet
    ├── anno
    │   ├── test_videoxum.json
    │   ├── train_videoxum.json
    │   └── val_videoxum.json
    └── feat
        ├── blip
        │   ├── v_00Dk03Jr70M.npz
        │   └── ...
        └── vt_clipscore
            ├── v_00Dk03Jr70M.npz
            └── ...
```


## 2 Requirements
- Python 3.8
- PyTorch == 1.10.1
- torchvision = 0.11.2
- CUDA == 11.1
- timm == 0.4.12
- transformers == 4.15.0
- fairscale == 0.4.4
- ruamel.yaml==0.17.21
- CLIP == 1.0
- Other dependencies: pycocoevalcap, opencv-python, scipy, pandas, ftfy, regex, tqdm

## 3 Installation

- Clone this repository:

  ```bash
  git clone https://github.com/jylins/videoxum.git
  ```

- Create a conda virtual environment and activate it:

  ```bash
  conda create -n videoxum python=3.8 -y
  conda activate videoxum
  ```

- Install `PyTorch==1.10.1` and `torchvision==0.11.2` with `CUDA==11.1`:

  ```bash
  pip install torch==1.10.1 torchvision==0.11.2 --index-url https://download.pytorch.org/whl/cu111
  ```

- Install `transformers==4.15.0`, `fairscale==0.4.4` and `timm==0.4.12`:

  ```bash
  pip install transformers==4.15.0
  pip install fairscale==0.4.4
  pip install timm==0.4.12
  ```
  
- Install `ruamel.yaml==0.17.21`:

  ```bash
  pip install ruamel.yaml==0.17.21
  ```

- Install `clip==1.0`:

  ```bash
  pip install git+https://github.com/openai/CLIP.git
  ```

- Install `pycocoevalcap` :

  ```bash
  cd pycocoevalcap
  pip install -e .
  ```

- Install other requirements:

  ```bash
  pip install -U scikit-learn
  pip install opencv-python scipy pandas ftfy regex tqdm
  ```



## 4 Model Zoo

| Version              | Checkpoint                                                                           | F1 score | Kendall | Spearman | BLEU@4 | METEOR | ROUGE-L | CIDEr | VT-CLIPScore |
|----------------------|--------------------------------------------------------------------------------------|----------|---------|----------|--------|--------|---------|-------|-----------|
| VTSUM-BLIP + TT      | [vtsum_tt](https://huggingface.co/jylins/vtsum_blip/resolve/main/vtsum_tt.pth)       | 22.4     | 0.176   | 0.233    | 5.7    | 12.0   | 24.9    | 22.4  | 29.0      |
| VTSUM-BLIP + TT + CA | [vtsum_tt_ca](https://huggingface.co/jylins/vtsum_blip/resolve/main/vtsum_tt_ca.pth) | 23.5     | 0.196   | 0.258    | 5.8    | 12.2   | 25.1    | 23.1  | 29.5      |

Note that the results are slightly different (~0.1%) from what we reported in the paper.
The file structure of Model zoo looks like:
```
outputs
├── blip
│   └── model_base_capfilt_large.pth
├── vt_clipscore
│   └── vt_clip.pth
├── vtsum_tt
│   └── vtsum_tt.pth
└── vtsum_tt_ca
    └── vtsum_tt_ca.pth
```

## 5 Training
### VTSUM-BLIP + Temporal Transformer (TT)
```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=4 train_v2vt_sum.py \
  --config configs/vtsum_blip_tt.yaml \
  --output_dir outputs/vtsum_tt \
  --model vtsum_blip_tt_ca \
  --max_epoch 56 \
  --lambda_tsum 1.0 \
  --lambda_vsum 10.0 \
  --batch_size 16 \
  --ckpt_freq 56
```

### VTSUM-BLIP + Temporal Transformer (TT) + Context Attention (CA)

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=4 train_v2vt_sum.py \
  --config configs/vtsum_blip_tt_ca.yaml \
  --output_dir outputs/vtsum_tt_ca \
  --model vtsum_blip_tt_ca \
  --max_epoch 56 \
  --lambda_tsum 1.0 \
  --lambda_vsum 15.0 \
  --init_lr 2e-5 \
  --kernel_size 5 \
  --batch_size 16 \
  --ckpt_freq 56
```
## 6 Evaluation

### VTSUM-BLIP + Temporal Transformer (TT)
```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=4 eval_v2vt_sum.py \
  --config configs/vtsum_blip_tt.yaml \
  --output_dir outputs/vtsum_tt \
  --pretrained_model outputs/vtsum_tt/vtsum_tt.pth \
  --model vtsum_blip_tt
```

### VTSUM-BLIP + Temporal Transformer (TT) + Context Attention (CA)
```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=4 eval_v2vt_sum.py \
  --config configs/vtsum_blip_tt_ca.yaml \
  --output_dir outputs/vtsum_tt_ca \
  --pretrained_model outputs/vtsum_tt_ca/vtsum_tt_ca.pth \
  --model vtsum_blip_tt_ca \
  --kernel_size 5
```

## 7 Citation
The paper has been accepted by IEEE Transactions on Multimedia.
```bash
@article{lin2023videoxum,
  author    = {Lin, Jingyang and Hua, Hang and Chen, Ming and Li, Yikang and Hsiao, Jenhao and Ho, Chiuman and Luo, Jiebo},
  title     = {VideoXum: Cross-modal Visual and Textural Summarization of Videos},
  journal   = {IEEE Transactions on Multimedia},
  year      = {2023},
}
```
## 8 Acknowledgements
This project is built upon the [BLIP](https://github.com/salesforce/BLIP) codebase.