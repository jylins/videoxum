# VideoXum: Cross-modal Visual and Textural Summarization of Videos

This repository is the official PyTorch implementation of VideoXum.

## 1 VideoXum Dataset
The VideoXum dataset represents a novel task in the field of video summarization, extending the scope from single-modal to cross-modal video summarization. This new task focuses on creating video summaries that containing both visual and textual elements with semantic coherence. Built upon the foundation of ActivityNet Captions, VideoXum is a large-scale dataset, including over 14,000 long-duration and open-domain videos. Each video is paired with 10 corresponding video summaries, amounting to a total of 140,000 video-text summary pairs.

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



## 2 TODO
- [ ] training code of videoxum
- [ ] evaluation code of videoxum


## 3 Citation

The paper has been accepted by IEEE Transactions on Multimedia.
```bash
@article{lin2023videoxum,
  author    = {Lin, Jingyang and Hua, Hang and Chen, Ming and Li, Yikang and Hsiao, Jenhao and Ho, Chiuman and Luo, Jiebo},
  title     = {VideoXum: Cross-modal Visual and Textural Summarization of Videos},
  journal   = {IEEE Transactions on Multimedia},
  year      = {2023},
}
```