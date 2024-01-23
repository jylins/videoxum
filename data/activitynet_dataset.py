import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset

from data.utils import pre_caption, pre_video


class ActivityNet(Dataset):
    def __init__(self,
                 video_embeddings_root, ann_file,
                 max_frames=512, max_words=512, prompt='',
                 clip_root=None):
        '''
        video_embeddings_root (string): Root directory of video embedding
        ann_file (string): the annotation file
        max_frames (int): the maximum number of frames
        max_words (int): the maximum number of words
        prompt (string): the prompt
        '''
        annotation = json.load(open(ann_file, 'r'))
        self.video_embeddings_root = video_embeddings_root
        self.max_frames = max_frames
        self.max_words = max_words
        self.prompt = prompt
        self.clip_root = clip_root

        # sorted by durations
        durations = []
        for anno in annotation:
            durations.append(anno['duration'])
        durations = np.array(durations)
        dur_sort_inds = np.argsort(durations)

        self.annotation = []
        for i in dur_sort_inds:
            data = dict(
                video_id=annotation[i]['video_id'],
                tsum_labels=''.join(annotation[i]['tsum']),
                vsum_labels=np.array(annotation[i]['vsum_onehot']))
            self.annotation.append(data)

        self.video_ids = {}
        n = 0
        for ann in self.annotation:
            video_id = ann['video_id']
            if video_id not in self.video_ids.keys():
                self.video_ids[video_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        video_embeddings_path = os.path.join(self.video_embeddings_root, ann['video_id']+'.npz')
        video_embeddings = np.load(video_embeddings_path)['features']
        video_embeddings = torch.from_numpy(video_embeddings).float()

        vsum_labels = torch.from_numpy(ann['vsum_labels']).float()
        video_embeddings, video_mask, vsum_labels = pre_video(
            video_embeddings, vsum_labels, self.max_frames)

        tsum_labels = self.prompt + pre_caption(ann['tsum_labels'], self.max_words)

        assert video_embeddings.shape[0] == vsum_labels.shape[1]
        if self.clip_root is not None:
            clip_path = os.path.join(self.clip_root, ann['video_id'] + '.npz')
            clip_features = np.load(clip_path)
            video_clip_features = torch.from_numpy(clip_features['vision']).float()
            text_clip_features = torch.from_numpy(clip_features['text']).float()
            video_clip_features, _, _ = pre_video(
                video_clip_features, vsum_labels, self.max_frames)
            if video_clip_features.shape[0] != vsum_labels.shape[1]:
                max_len = min(video_clip_features.shape[0], vsum_labels.shape[1])
                video_embeddings = video_embeddings[:max_len, :]
                video_mask = video_mask[:max_len]
                video_clip_features = video_clip_features[:max_len, :]
                vsum_labels = vsum_labels[:, :max_len]
            return video_embeddings, video_mask, vsum_labels, tsum_labels, \
                    video_clip_features, text_clip_features, self.video_ids[ann['video_id']]
        return video_embeddings, video_mask, vsum_labels, tsum_labels, self.video_ids[ann['video_id']]