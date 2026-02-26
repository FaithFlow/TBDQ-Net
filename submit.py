# ------------------------------------------------------------------------
# TBDQ-Net: Tracking by Detection and Query
# Copyright (c) 2026 Shukun Jia. All Rights Reserved.
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
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from util.eval_tool import load_model
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset
from engine import train_one_epoch_mot, evaluate
from models import build_model
import torch.backends.cudnn as cudnn
import torch.distributed as dist

cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def get_args_parser():
    parser = argparse.ArgumentParser('TBDQ-Net', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    parser.add_argument('--test_mode', default='val', type=str)
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
    parser.add_argument('--pe_temperatureH', default=20, type=int, help='')
    parser.add_argument('--pe_temperatureW', default=20, type=int, help='')

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

    if args.test_mode == 'val':
        dataset_val = build_dataset(image_set='val', args=args)
    elif args.test_mode == 'test':
        dataset_val = build_dataset(image_set='test', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    collate_fn = utils.mot_collate_fn
    data_loader_val = DataLoader(dataset_val, batch_size=1, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

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
                       not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        }
    ]
    print('trainable params:', len(param_dicts[0]["params"]) + len(param_dicts[1]["params"]) + len(param_dicts[2]["params"]))

    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module


    if args.pretrained is not None:
        model_without_ddp = load_model(model_without_ddp, args.pretrained)

    output_dir = Path(args.output_dir)

    print('start evaluation')
    t0 = time.time()
    hota = inference_demo(model, criterion, None, data_loader_val, device, args.output_dir, args=args)
    print(hota)
    print('finish evaluation, time:', time.time() - t0)


from engine import ListImgDataset, filter_dt_by_score, filter_dt_by_area, attr_dict
from collections import defaultdict
from copy import deepcopy
import cv2
import json
import os
from models.structures import Instances

COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237), (138, 43, 226),
             (238, 130, 238),
             (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0), (255, 239, 213),
             (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222), (65, 105, 225),
             (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
             (144, 238, 144),
             (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
             (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
             (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255), (176, 224, 230),
             (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
             (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
             (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]

def plot_one_box(x, img, color=None, label=None, score=None, line_thickness=None):
    # Plots one bounding box on image img

    # tl = line_thickness or round(
    #     0.002 * max(img.shape[0:2])) + 1  # line thickness
    tl = 2
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        if score is not None:
            cv2.putText(img, score, (c1[0], c1[1] + 30), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def visualize_img_with_bbox(img_path, img, dt_instances: Instances, ref_pts=None, gt_boxes=None):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if dt_instances.has('scores'):
        img_show = draw_bboxes(img,
                               np.concatenate([dt_instances.boxes, dt_instances.scores.reshape(-1, 1)], axis=-1),
                               dt_instances.obj_idxes)
    else:
        img_show = draw_bboxes(img, dt_instances.boxes, dt_instances.obj_idxes)
    if ref_pts is not None:
        img_show = draw_points(img_show, ref_pts)
    if gt_boxes is not None:
        img_show = draw_bboxes(img_show, gt_boxes, identities=np.ones((len(gt_boxes),)) * -1)
    cv2.imwrite(img_path, img_show)
    return img_show

def draw_points(img: np.ndarray, points: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    assert len(points.shape) == 2 and points.shape[1] == 2, 'invalid points shape: {}'.format(points.shape)
    for i, (x, y) in enumerate(points):
        if i >= 300:
            color = (0, 255, 0)
        cv2.circle(img, (int(x), int(y)), 2, color=color, thickness=2)
    return img


def draw_bboxes(ori_img, bbox, identities=None, offset=(0, 0), cvt_color=False):
    if cvt_color:
        ori_img = cv2.cvtColor(np.asarray(ori_img), cv2.COLOR_RGB2BGR)
    img = ori_img
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box[:4]]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        if len(box) > 4:
            score = '{:.2f}'.format(box[4])
        else:
            score = None
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = COLORS_10[id % len(COLORS_10)]
        label = '{:d}'.format(id)
        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        img = plot_one_box([x1, y1, x2, y2], img, color, label, score=score)
    return img

@torch.no_grad()
def inference_demo(model, criterion, postprocessors, data_loader, device, output_dir, args=None):
    model.eval()
    criterion.eval()
    # metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # header = 'Test:'
    # print_freq = 10
    predict_path = os.path.join(output_dir, 'tracker')
    prob_threshold = args.score_threshold #args.box_threshold
    area_threshold = 100
    
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for data_dict in data_loader:
        seq_num = os.path.basename(data_dict['video_name'][0])
        img_list = os.listdir(os.path.join(data_dict['video_name'][0], 'img1'))
        img_list = [os.path.join(data_dict['video_name'][0], 'img1', i) for i in img_list if 'jpg' in i]

        img_list = sorted(img_list)

        track_instances = None
        det_db = []
        loader = DataLoader(ListImgDataset('', img_list, det_db), 1, num_workers=2)
        lines = defaultdict(list)
        total_dts = defaultdict(int)
        total_occlusion_dts = defaultdict(int)
        record_name = ''
        for i, data in enumerate(loader):
            cur_img, ori_img, proposals, f_path = [d[0] for d in data]
            cur_img, proposals = cur_img.to(device), proposals.to(device)

            seq_name = f_path.split('/')[-3]
            if record_name != seq_name:
                record_name = seq_name
                track_instances = None

            if track_instances is not None:
                track_instances.remove('boxes')
            seq_h, seq_w, _ = ori_img.shape

            # 内部包含backboe+encode+decode+跟踪匹配关系+跟踪目标过滤（从query中过滤）
            try:
                res = model.module.inference_single_image(cur_img, (seq_h, seq_w), track_instances, proposals)
            except:
                res = model.inference_single_image(cur_img, (seq_h, seq_w), track_instances, proposals)
            track_instances = res['track_instances']

            dt_instances_all = deepcopy(track_instances).get_bn(0)

            dt_instances_all = dt_instances_all.to(torch.device('cpu'))

            # Filter detections by score and area.
            dt_instances_all = filter_dt_by_score(dt_instances_all, prob_threshold)
            dt_instances_all = filter_dt_by_area(dt_instances_all, area_threshold)

            if 0 and (i % 10 == 0 or i < 2000):
                vis_img_path = os.path.join('.', predict_path, seq_name, '{:06d}.jpg'.format(i+1))
                os.makedirs(os.path.join('.', predict_path, seq_name), exist_ok=True)
                vis_img = visualize_img_with_bbox(vis_img_path, ori_img.cpu().numpy(), dt_instances_all)
                #print(vis_img_path)

                cv2.imwrite(vis_img_path, vis_img)

            active_indx = []
            full_indx = torch.arange(len(dt_instances_all), device=dt_instances_all.scores.device)
            for id in torch.unique(dt_instances_all.obj_idxes):
                indx = torch.where(dt_instances_all.obj_idxes == id)[0]
                active_indx.append(full_indx[indx][dt_instances_all.scores[indx].argmax()])
            if len(active_indx):
                active_indx = torch.stack(active_indx)
                dt_instances_all = dt_instances_all[active_indx]

            for g_id in range(args.g_size):
                dt_instances = dt_instances_all

                total_dts[g_id] += len(dt_instances)

                bbox_xyxy = dt_instances.boxes.tolist()
                identities = dt_instances.obj_idxes.tolist()
                scores = dt_instances.scores.tolist()
                save_format = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n'
                for xyxy, track_id, s in zip(bbox_xyxy, identities, scores):
                    if track_id < 0 or track_id is None:
                        continue
                    x1, y1, x2, y2 = xyxy
                    w, h = x2 - x1, y2 - y1
                    if args.dataset_file == 'e2e_mot':
                        frame_ith = int(os.path.splitext(os.path.basename(f_path))[0])
                        lines[g_id].append(save_format.format(frame=frame_ith, id=track_id, x1=x1, y1=y1, w=w, h=h))
                    else:
                        lines[g_id].append(save_format.format(frame=i + 1, id=track_id, x1=x1, y1=y1, w=w, h=h, s=s))

        for g_id in range(args.g_size):
            os.makedirs(os.path.join(predict_path + '%d' % g_id), exist_ok=True)
            with open(os.path.join(predict_path + '%d' % g_id, f'{seq_num}.txt'), 'w') as f:
                f.writelines(lines[g_id])
            print("{}: totally {} dts {} occlusion dts".format(seq_num, total_dts[g_id], total_occlusion_dts[g_id]))

    if dist.is_initialized():
        dist.barrier()
    
    if args.dataset_file == 'e2e_sports':
        import sys
        sys.path.append("/root/code/TrackEval/scripts")
        import run_mot_challenge
        for g_id in range(args.g_size):
            res_eval = run_mot_challenge.main(SPLIT_TO_EVAL="val",
                        METRICS=['HOTA', 'CLEAR', 'Identity'],
                        GT_FOLDER="/root/autodl-tmp/SportsMOT/images/val/",
                        SEQMAP_FILE="/root/autodl-tmp/SportsMOT/images/val_seqmap.txt",
                        SKIP_SPLIT_FOL=True,
                        TRACKERS_TO_EVAL=[''],
                        TRACKER_SUB_FOLDER='',
                        USE_PARALLEL=True,
                        NUM_PARALLEL_CORES=8,
                        PLOT_CURVES=False,
                        TRACKERS_FOLDER="%s"%(predict_path+'%d'%g_id),
                        BENCHMARK='MOT17'
                        )
        
        return float(np.mean(res_eval[0]['MotChallenge2DBox']['']['COMBINED_SEQ']['pedestrian']['HOTA']['HOTA']))    
    else:
        import sys
        sys.path.append("/root/code/TrackEval/scripts")
        import run_mot_challenge
        for g_id in range(args.g_size):
            res_eval = run_mot_challenge.main(SPLIT_TO_EVAL="val",
                                              METRICS=['HOTA', 'CLEAR', 'Identity'],
                                              GT_FOLDER="/root/autodl-tmp/DanceTrack/val",
                                              SEQMAP_FILE="/root/autodl-tmp/DanceTrack/val_seqmap.txt",
                                              SKIP_SPLIT_FOL=True,
                                              TRACKERS_TO_EVAL=[''],
                                              TRACKER_SUB_FOLDER='',
                                              USE_PARALLEL=True,
                                              NUM_PARALLEL_CORES=8,
                                              PLOT_CURVES=False,
                                              TRACKERS_FOLDER="%s" % (predict_path + '%d' % g_id)
                                              )
        return float(np.mean(res_eval[0]['MotChallenge2DBox']['']['COMBINED_SEQ']['pedestrian']['HOTA']['HOTA']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('TBDQ-Net inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
