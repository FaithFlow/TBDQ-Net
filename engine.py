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


"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import torch.distributed as dist
import torch
import util.misc as utils
import numpy as np
from datasets.data_prefetcher import data_dict_to_cuda

def train_one_epoch_mot(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    try:
        model.module.detector.eval()
    except:
        print('SET DETECTOR TO EVAL MODE FAILING !!!')

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1000
    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        data_dict = data_dict_to_cuda(data_dict, device)
        data_dict['epoch'] = epoch
        outputs = model(data_dict)

        loss_dict = criterion(outputs, data_dict)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        loss_two = 0
        loss_ori = 0
        for k, v in loss_dict_reduced_scaled.items():
            if '_two_' in k: loss_two += v
            else: loss_ori += v
        loss_dict_reduced_scaled['loss_ori'] = loss_ori
        loss_dict_reduced_scaled['loss_two'] = loss_two

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        if torch.isnan(grad_total_norm).any():
            print(data_dict['gt_instances'])
            optimizer.zero_grad()

        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
  
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


import cv2
import json
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import torchvision.transforms.functional as F
class ListImgDataset(Dataset):
    def __init__(self, mot_path, img_list, det_db) -> None:
        super().__init__()
        self.mot_path = mot_path
        self.img_list = img_list
        self.det_db = det_db

        '''
        common settings
        '''
        self.img_height = 800
        self.img_width = 1536      
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_img_from_file(self, f_path):
        cur_img = cv2.imread(os.path.join(self.mot_path, f_path))
        assert cur_img is not None, f_path
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        proposals = []
        im_h, im_w = cur_img.shape[:2]
        if len(self.det_db):
            for line in self.det_db[f_path[:-4].replace('dancetrack/', 'DanceTrack/') + '.txt']:
                l, t, w, h, s = list(map(float, line.split(',')))
                proposals.append([(l + w / 2) / im_w,
                                    (t + h / 2) / im_h,
                                    w / im_w, h / im_h, s])
        return cur_img, torch.as_tensor(proposals).reshape(-1, 5), f_path

    def init_img(self, img, proposals):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        ##target_h = int(self.seq_h * scale)
        ##target_w = int(self.seq_w * scale)
        target_h = int(self.seq_h * scale / 32) * 32
        target_w = int(self.seq_w * scale / 32) * 32
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img, proposals

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img, proposals, f_path = self.load_img_from_file(self.img_list[index])
        img, ori_img, proposals = self.init_img(img, proposals)
        return img, ori_img, proposals, f_path


def filter_dt_by_score(dt_instances, prob_threshold):
    keep = dt_instances.scores > prob_threshold
    keep &= dt_instances.obj_idxes >= 0
    return dt_instances[keep]

def filter_dt_by_dual_score(dt_instances, prob_threshold):
    keep = (dt_instances.scores > prob_threshold) * (dt_instances.invididual > prob_threshold)
    keep &= dt_instances.obj_idxes >= 0
    return dt_instances[keep]

def filter_dt_by_area(dt_instances, area_threshold):
    wh = dt_instances.boxes[..., 2:4] - dt_instances.boxes[..., 0:2]
    areas = wh[..., 0] * wh[..., 1]
    keep = areas > area_threshold
    return dt_instances[keep]

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, output_dir, args=None):
    model.eval()
    criterion.eval()
    predict_path = os.path.join(output_dir, 'tracker')
    prob_threshold = args.score_threshold
    area_threshold = 100

    for data_dict in data_loader:
        # data_dict = data_dict_to_cuda(data_dict, device)
        # outputs = model.inference_single_image (data_dict)
        
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

        # Track sequence boundaries.
        record_name = ''

        for i, data in enumerate(loader):   # tqdm(loader)):
            cur_img, ori_img, proposals, f_path = [d[0] for d in data]
            cur_img, proposals = cur_img.to(device), proposals.to(device)
            # print(i)
            # Reset tracker at sequence boundary.
            seq_name = f_path.split('/')[-3]
            if record_name != seq_name:
                record_name = seq_name
                track_instances = None

            if track_instances is not None:
                track_instances.remove('boxes')
            seq_h, seq_w, _ = ori_img.shape

            # Run model inference: detection + tracking.
            try:
                res = model.module.inference_single_image(cur_img, (seq_h, seq_w), track_instances, proposals)
            except:
                res = model.inference_single_image(cur_img, (seq_h, seq_w), track_instances, proposals)
            track_instances = res['track_instances']

            dt_instances_all = deepcopy(track_instances).get_bn(0)

            # Filter detections by score and area.
            dt_instances_all = filter_dt_by_score(dt_instances_all, prob_threshold)
            dt_instances_all = filter_dt_by_area(dt_instances_all, area_threshold)

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

                save_format = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n'
                for xyxy, track_id in zip(bbox_xyxy, identities):
                    if track_id < 0 or track_id is None:
                        continue
                    x1, y1, x2, y2 = xyxy
                    w, h = x2 - x1, y2 - y1
                    if args.dataset_file == 'e2e_mot':
                        frame_ith = int(os.path.splitext(os.path.basename(f_path))[0])
                        lines[g_id].append(save_format.format(frame=frame_ith, id=track_id, x1=x1, y1=y1, w=w, h=h))
                    else:
                        lines[g_id].append(save_format.format(frame=i + 1, id=track_id, x1=x1, y1=y1, w=w, h=h))
                    
        for g_id in range(args.g_size):
            os.makedirs(os.path.join(predict_path+'%d'%g_id), exist_ok=True)
            with open(os.path.join(predict_path+'%d'%g_id, f'{seq_num}.txt'), 'w') as f:
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
                        TRACKERS_FOLDER="%s"%(predict_path+'%d'%g_id)
                        )
        return float(np.mean(res_eval[0]['MotChallenge2DBox']['']['COMBINED_SEQ']['pedestrian']['HOTA']['HOTA']))

