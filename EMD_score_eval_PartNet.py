import argparse
import datetime
import importlib
import models
import os
import time
import numpy as np
import open3d
import torch
import typing

import sys
sys.path.append('/home/wuruihai/PyTorchEMD')
from emd import earth_mover_distance

device = 3

gt_path = '/home/wuruihai/CompleteShapePC_npz_fps/test'
pred_path = '/home/wuruihai/Detail-Preserved-Point-Cloud-Completion-via-SFA/log/partnet_rfa/eval/TEST_npzs/00'

EMD_loss = {}
n_files = {}

''' 读取 CD '''
# score_dict = {}
# score_dir = '/home/wuruihai/Detail-Preserved-Point-Cloud-Completion-via-SFA/data/rfa/VAL_scores/03001627/scores_cd.txt'
# score_dir2 = '/home/wuruihai/Detail-Preserved-Point-Cloud-Completion-via-SFA/data/rfa/VAL_scores/03001627/scores_cd_emd_2.txt'
#
# file = open(score_dir, 'r')
# while True:
#     line = file.readline()
#     if line:
#         line = line.split('\t')
#         score_dict[line[0]] = [float(line[1])]   # only CD
#     else:
#         break



EMD_loss = torch.tensor(0.0).to(device)
n_files = 0

for root, dirs, files in os.walk(gt_path):
    for file in files:
        n_files += 1
        # if n_files > 2:
        #     break

        model_id = os.path.splitext(file)[0]
        print(n_files, model_id)

        gt = np.load(os.path.join(gt_path, file))['pts']
        gt = torch.from_numpy(gt).to(device)

        pred = np.load(os.path.join(pred_path, model_id + "'.npz"))['pts']
        pred = torch.from_numpy(pred).to(device)

        emd = earth_mover_distance(pred.reshape(1, -1, 3), gt.reshape(1, -1, 3), transpose=False) / pred.shape[0]
        # print(emd)
        EMD_loss += emd.mean()

''' 存 CD + EMD '''
# fw = open(score_dir2, 'w')
# for model_id in score_dict.keys():
#     print(model_id, score_dict[model_id])
#     fw.write('%s\t%s\t%s\n' % (model_id, score_dict[model_id][0], score_dict[model_id][1]))  # model_id \t CD \t EMD
# print(len(score_dict))


all_EMD = 0.0
print('EMD: ', EMD_loss.item() / n_files)










