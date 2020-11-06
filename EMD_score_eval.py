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

gt_path = '/raid/wuruihai/GRNet_FILES/xkh/ShapeNetCompletion/test/complete'
pred_path = '/home/wuruihai/Detail-Preserved-Point-Cloud-Completion-via-SFA/data/rfa/npzs'
test_taxonomy_list = ['02691156', '02933112', '02958343', '03001627', '03636649', '04256520', '04379243', '04530566']
test_taxonomy_list = ['03001627']
# test_taxonomy_list = ['04379243', '04530566']

EMD_loss = {}
n_files = {}

for taxonomy in test_taxonomy_list:
    cur_path = os.path.join(gt_path, taxonomy)
    print(cur_path)
    EMD_loss[taxonomy] = torch.tensor(0.0).to(device)
    n_files[taxonomy] = 0

    for root, dirs, files in os.walk(cur_path):
        for file in files:
            n_files[taxonomy] += 1
            if n_files[taxonomy] > 2:
                break
            model_id = os.path.splitext(file)[0]
            print('model_id: ', model_id)
            gt = open3d.io.read_point_cloud(os.path.join(cur_path, file))
            gt = np.array(gt.points, dtype=np.float32)
            gt = torch.from_numpy(gt, ).to(device)

            pred = np.load(os.path.join(pred_path, taxonomy, model_id + "'.npz"))['pts']
            pred = torch.from_numpy(pred).to(device)

            emd = earth_mover_distance(pred.reshape(1, -1, 3), gt.reshape(1, -1, 3), transpose=False) / pred.shape[0]
            EMD_loss[taxonomy] += emd.mean()

all_EMD = 0.0
for taxonomy in test_taxonomy_list:
    # if taxonomy in EMD_loss.keys():
    cur_taxonomy_emd = EMD_loss[taxonomy].item() / n_files[taxonomy]
    all_EMD += cur_taxonomy_emd
    print('%s: %f' % (taxonomy, EMD_loss[taxonomy].item() / n_files[taxonomy]))

print(all_EMD / len(test_taxonomy_list))







