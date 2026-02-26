# ------------------------------------------------------------------------
# TBDQ-Net: Tracking by Detection and Query
# Copyright (c) 2026 Shukun Jia (jsk0011@163.com). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from CO-MOT (https://github.com/BingfengYan/CO-MOT)
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from util.load_asso import load_model

import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset
from engine import train_one_epoch_mot, evaluate
from models import build_model
import torch.backends.cudnn as cudnn
import shutil

cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



def get_args_parser():
    parser = argparse.ArgumentParser('TBDQ-Net', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    parser.add_argument('--detector', default='', type=str)
    parser.add_argument('--det_nms', default=0.8, type=float)
    parser.add_argument('--qualified_threshold', default=0.2, type=float)

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float, help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--pe_temperatureH', default=20, type=int, help='')  # for DINO
    parser.add_argument('--pe_temperatureW', default=20, type=int, help='')  # for DINO

    # * Transformer
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int, help="Number of query slots")
    parser.add_argument('--extra_track_attn', action='store_true')
    parser.add_argument('--append_crowd', default=False, action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--match_type', default='', help='gmatch')
    parser.add_argument('--mix_match', action='store_true', )
    parser.add_argument('--set_cost_class', default=2, type=float, help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float, help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float, help="giou box coefficient in the matching cost")
    parser.add_argument('--match_unstable_error', default=True, action='store_true')  # for DINO

    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--gt_file_train', type=str)
    parser.add_argument('--gt_file_val', type=str)

    parser.add_argument('--output_dir', default='tmp/', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pretrained', default=None, help='resume from checkpoint')
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--yolox_weights', default='', help='resume from checkpoint')

    # end-to-end mot settings.
    parser.add_argument('--mot_path', default='/data/Dataset/mot', type=str)
    parser.add_argument('--det_db', default='', type=str)
    parser.add_argument('--data_txt_path_train', default='./datasets/data_path/detmot17.train', type=str,
                        help="path to dataset txt split")
    parser.add_argument('--data_txt_path_val', default='./datasets/data_path/detmot17.train', type=str,
                        help="path to dataset txt split")
    parser.add_argument('--training_set', default='sub', type=str)
    parser.add_argument('--mode', default='', type=str)

    parser.add_argument('--query_interaction_layer', default='QIM', type=str, help="GQIM, QIM, QIMv2")
    parser.add_argument('--sample_mode', type=str, default='fixed_interval')
    parser.add_argument('--sample_interval', type=int, default=1)
    parser.add_argument('--random_drop', type=float, default=0)
    parser.add_argument('--fp_ratio', type=float, default=0)

    parser.add_argument('--sampler_steps', type=int, nargs='*')
    parser.add_argument('--sampler_lengths', type=int, nargs='*')
    parser.add_argument('--exp_name', default='submit', type=str)

    parser.add_argument('--g_size', default=1, type=int)

    # test
    parser.add_argument('--score_threshold', default=0.3, type=float)
    parser.add_argument('--miss_tolerance', default=20, type=int)
    parser.add_argument('--not_valid', action='store_false', default=True)

    # Grounding DINO
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path_detector", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")

    return parser


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    def worker_init_fn(worker_id):
        # 全局种子是 torch.initial_seed()
        # 用全局种子为每个 worker 生成唯一种子
        seed = torch.initial_seed() % (2**32)  # 转换为 32 位整数
        print(f"Initializing worker {worker_id} with seed {seed}")
        np.random.seed(seed)  # 初始化 NumPy 的随机种子
        random.seed(seed)     # 初始化 Python 的随机种子
    

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model

    # Freeze backbone and detector parameters.
    for n, p in model_without_ddp.named_parameters():
        if 'backbone' in n:
            p.requires_grad = False
        if 'detector' in n:
            p.requires_grad = False
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    collate_fn = utils.mot_collate_fn
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=args.num_workers,
                                   pin_memory=True, worker_init_fn=worker_init_fn)
    data_loader_val = DataLoader(dataset_val, batch_size=1, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers,
                                 pin_memory=True, worker_init_fn=worker_init_fn)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad and p.is_leaf],
            "lr": args.lr,
        }
    ]
    print('trainable params:',
          len(param_dicts[0]["params"]) + len(param_dicts[1]["params"]) + len(param_dicts[2]["params"]))

    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=True)  # True
        model_without_ddp = model.module


    if args.pretrained is not None:
        model_without_ddp = load_model(model_without_ddp, args.pretrained)
    
    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            
            checkpoint['optimizer']['param_groups'][0]['lr'] *= 10
            checkpoint['optimizer']['param_groups'][1]['lr'] *= 10
            checkpoint['optimizer']['param_groups'][2]['lr'] *= 10
            
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # Override lr_drop in resumed lr_scheduler.
            from collections import Counter
            lr_scheduler.milestones = Counter(args.lr_drop)
            lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

    if 0:
        print('start evaluation')
        t0 = time.time()
        hota = evaluate(model, criterion, None, data_loader_val, device, args.output_dir, args=args)

        print(hota)
        print('finish evaluation, time:', time.time() - t0)

    print("Start training: %d" % args.start_epoch)
    start_time = time.time()
    hota_all = 0
    dataset_train.set_epoch(args.start_epoch)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch_mot(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()

        if args.not_valid:
            save_middle_ckpt = True
            if save_middle_ckpt and ((epoch + 1) > args.lr_drop[1] or (epoch + 1) % args.epochs == 0):
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, os.path.join(output_dir, 'checkpoint_%s.pth'%(str(epoch))))
                
            hota = evaluate(model, criterion, None, data_loader_val, device, args.output_dir, args=args)
            print(hota)
            
            if hota_all < hota:
                hota_all = hota
                checkpoint_path = output_dir / f'checkpoint_best.pth'
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
                ckpts = os.listdir(output_dir)
                ckpts = sorted([ckpt for ckpt in ckpts if 'checkpoint0' in ckpt])
                if len(ckpts) > 3:
                    remove_file = os.path.join(output_dir, ckpts[0])
                    os.remove(remove_file)
                print('save the best checkpoint on epoch', str(epoch), ':', checkpoint_path)
        dataset_train.step_epoch()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TBDQ-Net training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
