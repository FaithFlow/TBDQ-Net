'''
Author: Shukun Jia
Date: 2025-03-20
Description: Entrance of the project.
'''

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
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets', ], type=str,
                        nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int, nargs='+')
    parser.add_argument('--save_period', default=50, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    parser.add_argument('--meta_arch', default='deformable_detr', type=str)
    parser.add_argument('--detector', default='', type=str)
    parser.add_argument('--det_nms', default=0.8, type=float)
    parser.add_argument('--qualified_threshold', default=0.2, type=float)

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--accurate_ratio', default=False, action='store_true')
    parser.add_argument('--dn_labelbook_size', default=91, type=int)  # for DINO
    parser.add_argument('--dec_pred_class_embed_share', default=True, action='store_true')  # for DINO
    parser.add_argument('--dec_pred_bbox_embed_share', default=True, action='store_true')  # for DINO
    parser.add_argument('--fix_refpoints_hw', default=-1, type=int)  # for DINO
    parser.add_argument('--two_stage_class_embed_share', default=False, action='store_true')  # for DINO
    parser.add_argument('--two_stage_bbox_embed_share', default=False, action='store_true')  # for DINO
    parser.add_argument('--use_dn', default=True, action='store_true')  # for DINO
    parser.add_argument('--dn_number', default=100, type=int)  # for DINO
    parser.add_argument('--dn_box_noise_scale', default=0.4, type=float)  # for DINO
    parser.add_argument('--dn_label_noise_ratio', default=0.5, type=float)  # for DINO
    parser.add_argument('--num_select', default=300, type=int)  # for DINO
    parser.add_argument('--nms_iou_threshold', default=-1, type=float)  # for DINO

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # parser.add_argument('--num_anchors', default=1, type=int)

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--enable_fpn', action='store_true')
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float, help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--pe_temperatureH', default=20, type=int, help='')  # for DINO
    parser.add_argument('--pe_temperatureW', default=20, type=int, help='')  # for DINO
    parser.add_argument('--return_interm_indices', default=[0, 1, 2, 3], type=int, nargs='+')  # for DINO
    parser.add_argument('--backbone_freeze_keywords', default=None, type=str, nargs='+')  # for DINO

    # * Transformer
    parser.add_argument('--trans_mode', default='DeformableTransformer',
                        help='DeformableTransformer, DeformableTransformerCross')
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int, help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--decoder_cross_self', default=False, action='store_true')
    parser.add_argument('--sigmoid_attn', default=False, action='store_true')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--cj', action='store_true')
    parser.add_argument('--extra_track_attn', action='store_true')
    parser.add_argument('--loss_normalizer', action='store_true')
    parser.add_argument('--max_size', default=1333, type=int)
    parser.add_argument('--val_width', default=800, type=int)
    parser.add_argument('--filter_ignore', action='store_true')
    parser.add_argument('--append_crowd', default=False, action='store_true')
    parser.add_argument('--decoder_layer_noise', default=False, action='store_true')  # for DINO
    parser.add_argument('--dln_xy_noise', default=0.2, type=float, help="")  # for DINO
    parser.add_argument('--dln_hw_noise', default=0.2, type=float, help="")  # for DINO
    parser.add_argument('--use_detached_boxes_dec_out', default=False, action='store_true')  # for DINO
    parser.add_argument('--unic_layers', default=0, type=int)  # for DINO
    parser.add_argument('--pre_norm', default=False, action='store_true')  # for DINO
    parser.add_argument('--query_dim', default=4, type=int)  # for DINO
    parser.add_argument('--transformer_activation', default='relu', type=str)  # for DINO
    parser.add_argument('--num_patterns', default=0, type=int)  # for DINO
    parser.add_argument('--use_deformable_box_attn', default=False, action='store_true')  # for DINO
    parser.add_argument('--box_attn_type', default='roi_align', type=str)  # for DINO
    parser.add_argument('--add_channel_attention', default=False, action='store_true')  # for DINO
    parser.add_argument('--add_pos_value', default=False, action='store_true')  # for DINO
    parser.add_argument('--random_refpoints_xy', default=False, action='store_true')  # for DINO
    parser.add_argument('--two_stage_type', default='standard', type=str)  # for DINO
    parser.add_argument('--two_stage_pat_embed', default=0, type=int)  # for DINO
    parser.add_argument('--two_stage_add_query_num', default=0, type=int)  # for DINO
    parser.add_argument('--two_stage_learn_wh', default=False, action='store_true')  # for DINO
    parser.add_argument('--two_stage_keep_all_tokens', default=False, action='store_true')  # for DINO
    parser.add_argument('--dec_layer_number', default=None, type=int, nargs='+')  # for DINO
    parser.add_argument('--decoder_sa_type', default='sa', type=str,
                        help='["sa", "ca_label", "ca_content"]')  # for DINO
    parser.add_argument('--decoder_module_seq', default=['sa', 'ca', 'ffn'], type=str, nargs='+')  # for DINO
    parser.add_argument('--embed_init_tgt', default=True, action='store_true')  # for DINO
    parser.add_argument('--no_interm_box_loss', default=False, action='store_true')  # for DINO
    parser.add_argument('--interm_loss_coef', default=1.0, type=float)  # for DINO

    # * Segmentation
    parser.add_argument('--masks', action='store_true', help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--match_type', default='')
    parser.add_argument('--set_cost_class', default=2, type=float, help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float, help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float, help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    
    parser.add_argument('--output_dir', default='tmp/', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pretrained', default=None, help='resume from checkpoint')
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--yolox_weights', default='', help='resume from checkpoint')

    # end-to-end mot settings.
    parser.add_argument('--mot_path', default='/data/Dataset/mot', type=str)
    parser.add_argument('--data_txt_path_train', default='./datasets/data_path/detmot17.train', type=str,
                        help="path to dataset txt split")
    parser.add_argument('--data_txt_path_val', default='./datasets/data_path/detmot17.train', type=str,
                        help="path to dataset txt split")
    parser.add_argument('--training_set', default='sub', type=str)
    parser.add_argument('--mode', default='', type=str)

    parser.add_argument('--query_interaction_layer', default='QIM', type=str)
    parser.add_argument('--sample_mode', type=str, default='fixed_interval')
    parser.add_argument('--sample_interval', type=int, default=1)
    parser.add_argument('--random_drop', type=float, default=0)
    parser.add_argument('--fp_ratio', type=float, default=0)
    parser.add_argument('--merger_dropout', type=float, default=0.1)
    parser.add_argument('--update_query_pos', action='store_true')

    parser.add_argument('--sampler_steps', type=int, nargs='*')
    parser.add_argument('--sampler_lengths', type=int, nargs='*')
    parser.add_argument('--exp_name', default='submit', type=str)
    parser.add_argument('--memory_bank_score_thresh', type=float, default=0.)
    parser.add_argument('--memory_bank_len', type=int, default=4)
    parser.add_argument('--memory_bank_type', type=str, default=None)
    parser.add_argument('--memory_bank_with_self_attn', action='store_true', default=False)

    parser.add_argument('--use_checkpoint', action='store_true', default=False)
    parser.add_argument('--query_denoise', type=float, default=0.)

    parser.add_argument('--g_size', default=1, type=int)

    # test
    parser.add_argument('--score_threshold', default=0.3, type=float)
    #parser.add_argument('--update_score_threshold', default=0.5, type=float)
    parser.add_argument('--miss_tolerance', default=20, type=int)
    parser.add_argument('--not_valid', action='store_false', default=True)

    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
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
        seed = torch.initial_seed() % (2**32)
        print(f"Initializing worker {worker_id} with seed {seed}")
        np.random.seed(seed)  # 初始化 NumPy 的随机种子
        random.seed(seed)     # 初始化 Python 的随机种子
    

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model


    for n, p in model_without_ddp.named_parameters():

        if 'backbone' in n:
            p.requires_grad = False
        if 'detector' in n:
            #print(n)
            p.requires_grad = False
        #print(n, type(p), p.requires_grad)
    # '''
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
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad and p.is_leaf],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
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
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=True)  # True
        model_without_ddp = model.module

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    if args.pretrained is not None:
        model_without_ddp = load_model(model_without_ddp, args.pretrained)
    
    output_dir = Path(args.output_dir)
    
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
                #checkpoint_path = output_dir / f'checkpoint{epoch:04}.pth'
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
    parser = argparse.ArgumentParser('TBDQ training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
