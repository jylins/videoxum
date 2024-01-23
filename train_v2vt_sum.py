import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.vtsum_blip import v2vt_sum
from models.video_clip import build_video_clip_model
import utils
from utils import cosine_lr_schedule, update_config, compute_f1, compute_kendall, compute_spearman, concat_all_gather
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, collate_fn_padd_vtsum, collate_fn_padd_vtsum_eval
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def train(model, data_loader, optimizer, epoch, device, args):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train V2VTSum Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq

    for i, (video_embeddings, video_mask, vsum_labels, tsum_labels, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        video_embeddings = video_embeddings.to(device)
        video_mask = video_mask.to(device)
        vsum_labels = vsum_labels.to(device)

        loss_tsum, loss_vsum = model(video_embeddings, video_mask, vsum_labels, tsum_labels)
        loss = args.lambda_vsum * loss_vsum + args.lambda_tsum * loss_tsum

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_tsum=loss_tsum.item())
        metric_logger.update(loss_vsum=loss_vsum.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, clip_model, data_loader, device, config, eval_type):
    # evaluate
    model.eval()
    clip_model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Visual and Textual Summarization:'
    print_freq = 50

    # generate tsum
    result = []
    gt = []
    f1s_max = []
    f1s_mean = []
    kendalls_mean = []
    spearmans_mean = []
    clipscores = []
    for i, (video_embeddings, video_mask, vsum_labels, tsum_labels, video_clip_features, text_clip_features,
            video_ids) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        video_embeddings = video_embeddings.to(device)
        video_mask = video_mask.to(device)
        vsum_labels = vsum_labels.to(device)
        video_clip_features = video_clip_features.to(device)
        video_mask = video_mask.to(device)

        tsum_preds, saliency_scores = model.generate(
            video_embeddings, video_mask, sample=False, num_beams=config['num_beams'],
            max_length=config['max_text_length'], min_length=config['min_text_length'])

        # compute vsum evaluation metrics
        num_repeat = vsum_labels.shape[1]
        num_frame = vsum_labels.shape[2]
        saliency_scores = saliency_scores.repeat_interleave(num_repeat, dim=0)
        video_mask = video_mask.repeat_interleave(num_repeat, dim=0)
        vsum_labels = vsum_labels.reshape(-1, num_frame)

        f1_min, f1_max, f1_mean = compute_f1(saliency_scores, vsum_labels, video_mask)
        kendall_min, kendall_max, kendall_mean = compute_kendall(
            saliency_scores[::num_repeat],
            vsum_labels.reshape(-1, 10, num_frame).sum(dim=1) / 10.,
            video_mask[::num_repeat], 1)
        spearman_min, spearman_max, spearman_mean = compute_spearman(
            saliency_scores[::num_repeat],
            vsum_labels.reshape(-1, 10, num_frame).sum(dim=1) / 10.,
            video_mask[::num_repeat], 1)
        f1s_max.append(f1_max)
        f1s_mean.append(f1_mean)
        kendalls_mean.append(kendall_mean)
        spearmans_mean.append(spearman_mean)

        # compute tsum evaluation metrics
        for tsum_pred, video_id in zip(tsum_preds, video_ids):
            result.append({"image_id": video_id.item(), "caption": tsum_pred})

        for tsum_label, video_id in zip(tsum_labels, video_ids):
            gt.append({"image_id": video_id.item(), "id": video_id.item(), "caption": tsum_label[11:]})

        # compute clip evaluation metrics
        text_clip_features = clip_model.encoding_text(tsum_preds, device)
        text_clip_features = text_clip_features / text_clip_features.norm(dim=-1, keepdim=True)

        saliency_scores = saliency_scores[::num_repeat].squeeze(dim=-1) + video_mask[::num_repeat] * 1e+4
        _, sorted_inds = torch.sort(saliency_scores, dim=1, descending=True)
        top15_num = (video_mask[::num_repeat].sum(dim=1) * 0.15).int().clamp(min=1)
        video_clip_features_avg = torch.cat([
            video_clip_features[i, sorted_inds[i, :top15_num[i].item()]].mean(dim=0, keepdim=True) \
            for i in range(top15_num.size(0))], dim=0)
        video_clip_features_avg = video_clip_features_avg / video_clip_features_avg.norm(dim=-1, keepdim=True)

        logits = video_clip_features_avg @ text_clip_features.t()  # TxT
        clip_score = logits.diag()
        clipscores.append(clip_score)

    f1_max_score = torch.cat(f1s_max, dim=0).to(device)
    f1_mean_score = torch.cat(f1s_mean, dim=0).to(device)
    kendall_mean_score = torch.cat(kendalls_mean, dim=0).to(device)
    spearman_mean_score = torch.cat(spearmans_mean, dim=0).to(device)
    clipscores = torch.cat(clipscores, dim=0).to(device)

    f1_max_score = concat_all_gather(f1_max_score).mean().item()
    f1_mean_score = concat_all_gather(f1_mean_score).mean().item()
    kendall_mean_score = concat_all_gather(kendall_mean_score).mean().item()
    spearman_mean_score = concat_all_gather(spearman_mean_score).mean().item()
    clipscores = concat_all_gather(clipscores).mean().item()

    # save results
    result_file = save_result(result, config['result_dir'], eval_type, remove_duplicate='image_id')
    gt_file = save_result(gt, config['result_dir'], f'gt_{eval_type}', remove_duplicate='image_id', is_gt=True)

    # coco evaluation
    if utils.is_main_process():
        coco = COCO(gt_file)
        coco_result = coco.loadRes(result_file)

        # create coco_eval object by taking coco and coco_result
        coco_eval = COCOEvalCap(coco, coco_result)
        # evaluate results
        coco_eval.evaluate(['Bleu', 'METEOR', 'ROUGE_L', 'CIDEr'])

        # print output evaluation scores
        for metric, score in coco_eval.eval.items():
            print(f'{metric}: {score:.3f}')
        print('F1 score: MAX {:.4f}, MEAN {:.4f}'.format(f1_max_score, f1_mean_score))
        print('Kendall score: {:.4f}'.format(kendall_mean_score))
        print('Spearman score: MEAN {:.4f}'.format(spearman_mean_score))
        print('CLIP score: {:.4f}'.format(clipscores))

        log_stats = {**{f'val_{k}': v for k, v in coco_eval.eval.items()}}
        log_stats['F1_MAX'] = f1_max_score
        log_stats['F1_MEAN'] = f1_mean_score
        log_stats['Kendall'] = kendall_mean_score
        log_stats['Spearman'] = spearman_mean_score
        log_stats['VT_CLIPScore'] = clipscores
        with open(os.path.join(args.output_dir, f"evaluate_{eval_type}.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")
        with open(os.path.join(args.output_dir, f"evaluate_{eval_type}.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

    dist.barrier()


def main(args, config):
    utils.init_distributed_mode(args)
    print(args)
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating videoxum dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('videoxum', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(
            [train_dataset, val_dataset, test_dataset],
            [True, False, False], num_tasks, global_rank)
    else:
        samplers = [None] * 3
    
    train_loader, val_loader, test_loader = create_loader(
        [train_dataset, val_dataset, test_dataset], samplers,
        batch_size=[config['batch_size']]*3, num_workers=[4, 4, 4],
        is_trains=[True, False, False],
        collate_fns=[collate_fn_padd_vtsum, collate_fn_padd_vtsum_eval, collate_fn_padd_vtsum_eval])

    #### Model ####
    print("Creating model")
    if config['model'] == 'vtsum_blip_tt':
        model = v2vt_sum(config['model'], pretrained=config['pretrained'],
                         tt_depth=config['tt_depth'],
                         loss_type=config['loss_type'],
                         vit=config['vit'],
                         prompt=config['prompt'],
                         max_text_length=config['max_text_length'])
    elif config['model'] == 'vtsum_blip_tt_ca':
        model = v2vt_sum(config['model'], pretrained=config['pretrained'],
                         tt_depth=config['tt_depth'],
                         kernel_size=config['kernel_size'],
                         loss_type=config['loss_type'],
                         vit=config['vit'],
                         prompt=config['prompt'],
                         max_text_length=config['max_text_length'])

    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    clip_model = build_video_clip_model("ViT-B/16", model_path=config['pretrained_clip'], device=f"cuda:{args.gpu}")
    clip_model_without_ddp = clip_model
    if args.distributed:
        clip_model = torch.nn.parallel.DistributedDataParallel(clip_model, device_ids=[args.gpu])
        clip_model_without_ddp = clip_model.module

    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    print("Start training")
    start_time = time.time()
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            train(model, train_loader, optimizer, epoch, device, args)

        if utils.is_main_process():
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch}
            torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint.pth'))
        dist.barrier()

        if (epoch + 1) % config['max_epoch'] == 0 or args.evaluate:
            # val set
            evaluate(model_without_ddp, clip_model_without_ddp, val_loader, device, config, 'val')
            # test set
            evaluate(model_without_ddp, clip_model_without_ddp, test_loader, device, config, 'test')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='vtsum_blip')
    parser.add_argument('--config', default='configs/vtsum_blip.yaml')
    parser.add_argument('--output_dir', default='outputs/vtsum_blip')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--init_lr', type=float, default=1e-5)
    parser.add_argument('--max_epoch', type=int, default=28)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lambda_tsum', type=float, default=1.0)
    parser.add_argument('--lambda_vsum', type=float, default=1.0)
    parser.add_argument('--clip_root', type=str, default='dataset/ActivityNet/feat/vt_clipscore')

    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--ckpt_freq', type=int, default=28)

    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    config = update_config(config, args)

    config['result_dir'] = os.path.join(args.output_dir, 'result')
    config['ckpt_dir'] = os.path.join(args.output_dir, 'checkpoints')
    config['logger_pth'] = os.path.join(args.output_dir, 'train.log')
    config['kernel_size'] = args.kernel_size
    args.logger_pth = config['logger_pth']

    Path(config['result_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['ckpt_dir']).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
