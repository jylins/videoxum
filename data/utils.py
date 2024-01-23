import re
import json
import os

import torch
import torch.distributed as dist

import utils

def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption


def pre_video(video_embeddings, vsum_labels, max_frames=512):
    video_mask = torch.ones(video_embeddings.size(0), dtype=torch.long)

    if video_embeddings.size(0) > max_frames:
        video_embeddings = video_embeddings[:max_frames]
        video_mask = video_mask[:max_frames]
        vsum_labels = vsum_labels[:, :max_frames]

    return video_embeddings, video_mask, vsum_labels


def pre_question(question,max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    ) 
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question


def save_result(result, result_dir, filename, remove_duplicate='', is_gt=False):
    result_file = os.path.join(result_dir, f'{filename}_rank{utils.get_rank()}.json')
    final_result_file = os.path.join(result_dir, f'{filename}.json')

    json.dump(result, open(result_file, 'w'))

    dist.barrier()

    if utils.is_main_process():
        # combine results from all processes
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, f'{filename}_rank{rank}.json')
            res = json.load(open(result_file, 'r'))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new

        if is_gt:
            images = []
            for res in result:
                images.append({"id": res["id"]})
            result = dict(annotations=result, images=images)

        json.dump(result, open(final_result_file, 'w'))
        print(f'result file saved to {final_result_file}')

    return final_result_file



from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url

def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
            'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'}
    filenames = {'val':'coco_karpathy_val_gt.json','test':'coco_karpathy_test_gt.json'}    
    
    download_url(urls[split],coco_gt_root)
    annotation_file = os.path.join(coco_gt_root,filenames[split])
    
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
    return coco_eval


def collate_fn_padd_tsum(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    batch_size = len(batch)
    device = batch[0][0].device
    # get sequence lengths
    lengths = torch.tensor([t[0].shape[0] for t in batch]).to(device)
    max_length = lengths.max().item()

    # video embeddings
    batch_0 = torch.stack([
        torch.cat([t[0], torch.zeros(max_length - t[0].size(0), t[0].size(1)).to(device)], dim=0) for t in batch], dim=0)
    # video masks
    batch_1 = torch.stack([
        torch.cat([t[1], torch.zeros(max_length - t[1].size(0)).to(device)], dim=0) for t in batch], dim=0)
    # video tsum labels
    batch_2 = [batch[i][2] for i in range(batch_size)]
    # video ids
    batch_3 = torch.tensor([t[3] for t in batch])

    return batch_0, batch_1, batch_2, batch_3


def collate_fn_padd_vsum(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    batch_size = len(batch)
    device = batch[0][0].device
    # get sequence lengths
    lengths = torch.tensor([t[0].shape[0] for t in batch]).to(device)
    max_length = lengths.max().item()

    # video_embeddings
    batch_0 = torch.stack([
        torch.cat([t[0], torch.zeros(max_length - t[0].size(0), t[0].size(1)).to(device)], dim=0) for t in batch], dim=0)
    # video_mask
    batch_1 = torch.stack([
        torch.cat([t[1], torch.zeros(max_length - t[1].size(0)).to(device)], dim=0) for t in batch], dim=0)
    # vsum_labels
    batch_2 = torch.stack([
        torch.cat([t[2], torch.zeros([t[2].size(0), max_length - t[2].size(1)]).to(device)], dim=1) for t in batch], dim=0)
    batch_3 = torch.tensor([t[3] for t in batch])

    return batch_0, batch_1, batch_2, batch_3


def collate_fn_padd_vtsum(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    batch_size = len(batch)
    device = batch[0][0].device
    # get sequence lengths
    lengths = torch.tensor([t[0].shape[0] for t in batch]).to(device)
    max_length = lengths.max().item()

    # video embeddings
    batch_0 = torch.stack([
        torch.cat([t[0], torch.zeros(max_length - t[0].size(0), t[0].size(1)).to(device)], dim=0) for t in batch], dim=0)
    # video mask
    batch_1 = torch.stack([
        torch.cat([t[1], torch.zeros(max_length - t[1].size(0)).to(device)], dim=0) for t in batch], dim=0)
    # vsum labels
    batch_2 = torch.stack([
        torch.cat([t[2], torch.zeros([t[2].size(0), max_length - t[2].size(1)]).to(device)], dim=1) for t in batch], dim=0)
    # tsum labels
    batch_3 = [batch[i][3] for i in range(batch_size)]
    # video ids
    batch_4 = torch.tensor([t[4] for t in batch])

    return batch_0, batch_1, batch_2, batch_3, batch_4


def collate_fn_padd_vtsum_eval(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    batch_size = len(batch)
    device = batch[0][0].device
    # get sequence lengths
    lengths = torch.tensor([t[0].shape[0] for t in batch]).to(device)
    max_length = lengths.max().item()

    # video_embeddings
    batch_0 = torch.stack([
        torch.cat([t[0], torch.zeros(max_length-t[0].size(0), t[0].size(1)).to(device)], dim=0) for t in batch], dim=0)
    # video mask
    batch_1 = torch.stack([
        torch.cat([t[1], torch.zeros(max_length-t[1].size(0)).to(device)], dim=0) for t in batch], dim=0)
    # vsum labels
    batch_2 = torch.stack([
        torch.cat([t[2], torch.zeros([t[2].size(0), max_length-t[2].size(1)]).to(device)], dim=1) for t in batch],
        dim=0)
    # tsum labels
    batch_3 = [batch[i][3] for i in range(batch_size)]
    # video clip features
    batch_4 = torch.stack([
        torch.cat([t[4], torch.zeros(max_length-t[4].size(0), t[4].size(1)).to(device)], dim=0) for t in batch],
        dim=0)
    # text clip features
    batch_5 = torch.cat([t[5] for t in batch], dim=0)
    # video ids
    batch_6 = torch.tensor([t[6] for t in batch])

    return batch_0, batch_1, batch_2, batch_3, batch_4, batch_5, batch_6