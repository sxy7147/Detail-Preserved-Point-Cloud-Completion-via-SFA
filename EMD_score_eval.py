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
from emd_kaichun import earth_mover_distance

device = 3

gt_path = '/raid/wuruihai/GRNet_FILES/xkh/ShapeNetCompletion/test/complete'
pred_path = '/home/wuruihai/Detail-Preserved-Point-Cloud-Completion-via-SFA/data/shapenet_rfa/TEST_npzs'
test_taxonomy_list = ['02691156', '02933112', '02958343', '03001627', '03636649', '04256520', '04379243', '04530566']
# test_taxonomy_list = ['03001627']
# test_taxonomy_list = ['04379243', '04530566']

EMD_loss = {}
n_files = {}

''' 读取 CD '''
score_dict = {}
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

for taxonomy in test_taxonomy_list:
    cur_path = os.path.join(gt_path, taxonomy)
    print(cur_path)
    EMD_loss[taxonomy] = torch.tensor(0.0).to(device)
    n_files[taxonomy] = 0

    for root, dirs, files in os.walk(cur_path):
        for file in files:
            n_files[taxonomy] += 1
            # if n_files[taxonomy] > 2:
            #     break

            model_id = os.path.splitext(file)[0]
            if model_id[-1] == "'":
                model_id = model_id[0: -1]
            print('model_id: ', model_id)

            # test
            # p1 = torch.from_numpy(np.array([[[0.7, -0.1, 0.1], [0.5, 0.2, 0.3]]], dtype=np.float32)).cuda()
            # p1 = p1.repeat(1024, 1, 1)
            # pred = p1.reshape(1, -1, 3)
            # print('pred: ', pred)
            #
            # p2 = torch.from_numpy(np.array([[[0.3, 0.8, 0.2], [0.2, -0.2, 0.3]]], dtype=np.float32)).cuda()
            # p2 = p2.repeat(1024, 1, 1)
            # gt = p2.reshape(1, -1, 3)
            # print('pred.shape ', pred.shape[1])


            gt = open3d.io.read_point_cloud(os.path.join(cur_path, file))
            gt = np.array(gt.points, dtype=np.float32) + 0.5
            gt = torch.from_numpy(gt, ).to(device)

            pred = np.load(os.path.join(pred_path, taxonomy, model_id + "'.npz"))['pts']
            pred = pred.reshape(1, -1, 3) + 0.5
            pred = torch.from_numpy(pred).to(device)

            cur_emd = earth_mover_distance(pred.reshape(1, -1, 3), gt.reshape(1, -1, 3), transpose=False) / pred.shape[1]
            # print(emd)
            EMD_loss[taxonomy] += cur_emd.mean()
            score_dict[model_id]=(cur_emd.mean().item())
            print(model_id, score_dict[model_id])



''' 存 CD + EMD '''
# fw = open(score_dir2, 'w')
# for model_id in score_dict.keys():
#     print(model_id, score_dict[model_id])
#     fw.write('%s\t%s\t%s\n' % (model_id, score_dict[model_id][0], score_dict[model_id][1]))  # model_id \t CD \t EMD
# print(len(score_dict))


all_EMD = 0.0
for taxonomy in test_taxonomy_list:
    # if taxonomy in EMD_loss.keys():
    cur_taxonomy_emd = EMD_loss[taxonomy].item() / n_files[taxonomy]
    all_EMD += cur_taxonomy_emd
    print('%s: %f' % (taxonomy, EMD_loss[taxonomy].item() / n_files[taxonomy]))

print(all_EMD / len(test_taxonomy_list))









