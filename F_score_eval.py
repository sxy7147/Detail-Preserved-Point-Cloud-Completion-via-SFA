import argparse
import datetime
import importlib
import models
import os
import tensorflow as tf
import time
from data_util import lmdb_dataflow, get_queued_data
from termcolor import colored
from tf_util import add_train_summary
from visu_util import plot_pcd_three_views
import numpy as np
import open3d
import typing

def calculate_fscore(gt: open3d.geometry.PointCloud, pr: open3d.geometry.PointCloud, th: float = 0.01) -> typing.Tuple[
    float, float, float]:
    '''Calculates the F-score between two point clouds with the corresponding threshold value.'''
    d1 = pr.compute_point_cloud_distance(gt)
    d2 = gt.compute_point_cloud_distance(pr)

    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0

    return recall, precision, fscore

class TrainProvider:
    def __init__(self, args, is_training):
        df_test, self.num_test = lmdb_dataflow(args.lmdb_test, args.batch_size,
                                                 args.num_input_points, args.num_gt_points, is_training=False)
        batch_test = get_queued_data(df_test.get_data(), [tf.string, tf.float32, tf.float32],
                                      [[args.batch_size],
                                       [args.batch_size, args.num_input_points, 3],
                                       [args.batch_size, args.num_gt_points, 3]])
        self.batch_data = batch_test


''' 读出前面已存的CD, EMD '''
# score_dict = {}
# score_dir = '/home/wuruihai/Detail-Preserved-Point-Cloud-Completion-via-SFA/data/rfa/VAL_scores/03001627/scores2.txt'
# score_dir2 = '/home/wuruihai/Detail-Preserved-Point-Cloud-Completion-via-SFA/data/rfa/VAL_scores/03001627/scores3.txt'
#
# file = open(score_dir, 'r')
# while True:
#     line = file.readline()
#     if line:
#         line = line.split('\t')
#         score_dict[line[0]] = [float(line[1]), float(line[2])]   # CD, EMD
#     else:
#         break


def train(args):
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')

    #Note that theta is a parameter used for progressive training
    theta = tf.train.piecewise_constant(global_step, [10000, 20000, 50000],
                                        [0.01, 0.1, 0.5, 1.0], 'theta_op')

    provider = TrainProvider(args, is_training_pl)
    ids, inputs, gt = provider.batch_data
    num_eval_steps = provider.num_test // args.batch_size

    print('provider.num_valid', provider.num_test)
    print('num_eval_steps', num_eval_steps)

    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(inputs, gt, theta, False)
    add_train_summary('alpha', theta)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=10)
    saver.restore(sess, args.model_path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    start_time = time.time()
    while not coord.should_stop():
        print(colored('Testing...', 'grey', 'on_green'))
        total_time = 0
        total_loss_fine = 0
        total_recall = 0
        total_precision = 0
        cd_per_cat = {}
        recall_per_cat = {}
        precision_per_cat = {}
        sess.run(tf.local_variables_initializer())
        for j in range(num_eval_steps):
            start = time.time()

            ids_eval, inputs_eval, gt_eval, loss_fine, fine = sess.run([ids, inputs, gt, model.loss_fine, model.fine],
                               feed_dict={is_training_pl: False})

            pc_gt = open3d.geometry.PointCloud()
            pc_pr = open3d.geometry.PointCloud()
            # print(np.squeeze(gt_eval).shape)
            pc_gt.points = open3d.utility.Vector3dVector(np.squeeze(gt_eval))
            pc_pr.points = open3d.utility.Vector3dVector(np.squeeze(fine))

            recall, precision, f_score = calculate_fscore(pc_gt, pc_pr)
            # print('f_score:', f_score)

            synset_id = str(ids_eval[0]).split('_')[0].split('\'')[1]

            ''' 只算 chair '''
            # if synset_id != '03001627':
            #     continue

            if not cd_per_cat.get(synset_id):
                cd_per_cat[synset_id] = []
            if not recall_per_cat.get(synset_id):
                recall_per_cat[synset_id] = []
            if not precision_per_cat.get(synset_id):
                precision_per_cat[synset_id] = []
            cd_per_cat[synset_id].append(f_score)
            recall_per_cat[synset_id].append(recall)
            precision_per_cat[synset_id].append(precision)


            ''' output scores '''
            # model_id = str(ids_eval[0]).split('_')[1]
            # if model_id[-1] == "'":
            #     model_id = model_id[0: -1]
            # print(model_id)
            # total_loss_fine += f_score
            # total_recall += recall
            # total_precision += precision
            # total_time += time.time() - start
            #
            # score_dict[model_id].append(precision)
            # score_dict[model_id].append(recall)
            # score_dict[model_id].append(f_score)


            # if args.plot:
            #     for i in range(args.batch_size):
            #         model_id = str(ids_eval[i]).split('_')[1]
            #         os.makedirs(os.path.join(args.save_path, 'plots', synset_id), exist_ok=True)
            #         plot_path = os.path.join(args.save_path, 'plots', synset_id, '%s.png' % model_id)
            #         plot_pcd_three_views(plot_path, [inputs_eval[i], fine[i], gt_eval[i]],
            #                              ['input', 'output', 'ground truth'],
            #                              'CD %.4f' % (loss_fine),
            #                              [0.5, 0.5, 0.5])


        # print('Average F_score: %f' % (total_loss_fine / num_eval_steps))
        # print('Average recall: %f' % (total_recall / num_eval_steps))
        # print('Average precision: %f' % (total_precision / num_eval_steps))
        print('F_score per category')
        # dict_known = {'02691156': 'airplane','02933112': 'cabinet', '02958343': 'car', '03001627': 'chair', '03636649': 'lamp', '04256520': 'sofa',
        #         '04379243' : 'table','04530566': 'vessel'}
        dict_known = {'00': 'PartNet'}
        temp_loss=0
        for synset_id in dict_known.keys():
            temp_loss += np.mean(cd_per_cat[synset_id])
            print(dict_known[synset_id], ' %f\t%f\t%f' % (np.mean(cd_per_cat[synset_id]), np.mean(recall_per_cat[synset_id]), np.mean(precision_per_cat[synset_id])))
        break
    print('Total time', datetime.timedelta(seconds=time.time() - start_time))
    coord.request_stop()
    coord.join(threads)
    sess.close()

    ''' 存 CD + EMD + precision + recall + Fscore '''
    # fw = open(score_dir2, 'w')
    # for model_id in score_dict.keys():
    #     print(model_id, score_dict[model_id])
    #     fw.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (model_id, score_dict[model_id][0], score_dict[model_id][1], score_dict[model_id][2], score_dict[model_id][3], score_dict[model_id][4]))  # model_id CD END precision recall fscore
    # print(len(score_dict))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_test', default='/home/wuruihai/PartNet_fps_test.lmdb')
    parser.add_argument('--model_type', default='rfa')
    parser.add_argument('--model_path', default='/home/wuruihai/Detail-Preserved-Point-Cloud-Completion-via-SFA/log/partnet_rfa/model-158000')
    parser.add_argument('--save_path', default='/home/wuruihai/Detail-Preserved-Point-Cloud-Completion-via-SFA/log/partnet_rfa/eval')
    # parser.add_argument('--model_type', default='glfa')
    # parser.add_argument('--model_path', default='data/trained_models/glfa')
    # parser.add_argument('--save_path', default='data/glfa')
    parser.add_argument('--plot', default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_input_points', type=int, default=2048)
    parser.add_argument('--num_gt_points', type=int, default=16384)
    args = parser.parse_args()

    train(args)