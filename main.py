#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
import setproctitle
import dataset
from model import BGCN, BGCN_Info
from utils import check_overfitting, early_stop, logger
from train import train
from metric import Recall, NDCG, MRR
from config import CONFIG
from test import test
import loss
from itertools import product
import time
from tensorboardX import SummaryWriter


def main():
    #  set env
    setproctitle.setproctitle(f"train{CONFIG['name']}")
    os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['gpu_id']
    device = torch.device('cuda')

    #  fix seed
    seed = 123
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    #  load data
    # remove tune data
    bundle_train_data, bundle_test_data, item_data, assist_data = \
            dataset.get_dataset(CONFIG['path'], CONFIG['dataset_name'], task=CONFIG['task'])

    train_loader = DataLoader(bundle_train_data, 1024, True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(bundle_test_data, 1024, False,
                             num_workers=4, pin_memory=True)

    #  pretrain
    if 'pretrain' in CONFIG:
        pretrain = torch.load(CONFIG['pretrain'], map_location='cpu')
        print('load pretrain')

    #  graph
    ub_graph = bundle_train_data.ground_truth_u_b
    ui_graph = item_data.ground_truth_u_i
    bi_graph = assist_data.ground_truth_b_i

    #  metric
    metrics = [Recall(30), Recall(50), NDCG(30), NDCG(50)]
    TARGET = 'Recall@30'

    #  loss
    loss_func = loss.BPRLoss('mean')

    #  log
    log = logger.Logger(os.path.join(
        CONFIG['log'], CONFIG['dataset_name'], 
        f"{CONFIG['model']}_{CONFIG['task']}", ''), 'best', checkpoint_target=TARGET)

    theta = 0.6

    time_path = time.strftime("%y%m%d-%H%M%S", time.localtime(time.time()))

    for lr, decay, message_dropout, node_dropout \
            in product(CONFIG['lrs'], CONFIG['decays'], CONFIG['message_dropouts'], CONFIG['node_dropouts']):

        visual_path =  os.path.join(CONFIG['visual'], 
                                    CONFIG['dataset_name'],  
                                    f"{CONFIG['model']}_{CONFIG['task']}", 
                                    f"{time_path}@{CONFIG['note']}", 
                                    f"lr{lr}_decay{decay}_medr{message_dropout}_nodr{node_dropout}")

        # model
        if CONFIG['model'] == 'BGCN':
            graph = [ub_graph, ui_graph, bi_graph]
            info = BGCN_Info(64, decay, message_dropout, node_dropout, 2)
            model = BGCN(info, assist_data, graph, device, pretrain=None).to(device)

        assert model.__class__.__name__ == CONFIG['model']

        # op
        op = optim.Adam(model.parameters(), lr=lr)
        # env
        env = {'lr': lr,
               'op': str(op).split(' ')[0],   # Adam
               'dataset': CONFIG['dataset_name'],
               'model': CONFIG['model'], 
               'sample': CONFIG['sample'],
               }

        #  continue training
        if CONFIG['sample'] == 'hard' and 'conti_train' in CONFIG:
            model.load_state_dict(torch.load(CONFIG['conti_train']))
            print('load model and continue training')

        log.update_modelinfo(info, env, metrics)
            # train & test
        train_writer = SummaryWriter(log_dir=visual_path, comment='train')
        test_writer = SummaryWriter(log_dir=visual_path, comment='test') 
        for epoch in range(CONFIG['epochs']):
            # train
            trainloss = train(model, epoch+1, train_loader, op, device, CONFIG, loss_func)
            train_writer.add_scalars('loss/single', {"loss": trainloss}, epoch)

            # test
            output_metrics = test(model, test_loader, device, CONFIG, metrics)

            for metric in output_metrics:
                test_writer.add_scalars('metric/all', {metric.get_title(): metric.metric}, epoch)
                if metric==output_metrics[0]:
                    test_writer.add_scalars('metric/single', {metric.get_title(): metric.metric}, epoch)

            # log
            log.update_log(metrics, model) 
    log.close()


if __name__ == "__main__":
    main()