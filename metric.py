#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from config import CONFIG
import os

_is_hit_cache = {}

def get_is_hit(scores, ground_truth, topk):
    global _is_hit_cache
    cacheid = (id(scores), id(ground_truth))
    if topk in _is_hit_cache and _is_hit_cache[topk]['id'] == cacheid:
        return _is_hit_cache[topk]['is_hit']
    else:
        device = scores.device
        _, col_indice = torch.topk(scores, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(
            scores.shape[0], device=device, dtype=torch.long).view(-1, 1)
        is_hit = ground_truth[row_indice.view(-1),
                              col_indice.view(-1)].view(-1, topk)
        _is_hit_cache[topk] = {'id': cacheid, 'is_hit': is_hit}
        return is_hit

class _Metric:
    '''
    base class of metrics like Recall@k NDCG@k MRR@k
    '''

    def __init__(self):
        self.start()
        self.load_bi()
        print(len(self.bi.keys()))

    @property
    def metric(self):
        return self._metric

    def __call__(self, scores, ground_truth):
        '''
        - scores: model output
        - ground_truth: one-hot test dataset shape=(users, all_bundles/all_items).
        '''
        raise NotImplementedError

    def get_title(self):
        raise NotImplementedError

    def start(self):
        '''
        clear all
        '''
        global _is_hit_cache
        _is_hit_cache = {}
        self._cnt = 0
        self._metric = 0
        self._sum = 0
        self.bi = {}

    def stop(self):
        global _is_hit_cache
        _is_hit_cache = {}
        self._metric = self._sum/self._cnt
        
    def load_bi(self):
        path = CONFIG['path']
        name = CONFIG['dataset_name']
        with open(os.path.join(path, name, 'bundle_item.txt'), 'r') as f:
            for line in f.readlines():
                b, i = line.strip().split()
                if int(b) not in self.bi:
                    self.bi[int(b)] = set([int(i)])
                else:
                    self.bi[int(b)].add(int(i))

class Recall(_Metric):
    '''
    Recall in top-k samples
    '''

    def __init__(self, topk):
        super().__init__()
        self.topk = topk
        self.epison = 1e-8

    def get_title(self):
        return "Recall@{}".format(self.topk)

    def __call__(self, scores, ground_truth):
        is_hit = get_is_hit(scores, ground_truth, self.topk)
        is_hit = is_hit.sum(dim=1)
        num_pos = ground_truth.sum(dim=1)
        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += (is_hit/(num_pos+self.epison)).sum().item()


class Jaccard(_Metric):
    '''
    Jaccard in top-k samples
    '''
    
    def __init__(self, topk):
        super().__init__()
        self.epison = 1e-8
        self.topk = topk
        
    def get_title(self):
        return "Jaccard@{}".format(self.topk)
    
    def cal_overlap(self, list_bun1, list_bun2):
        ret = 0
        for i in list_bun1:
            tmp = 0
            if len(list_bun2) == 0:
                continue
            for j in list_bun2:
                overlap = self.bi[i].intersection(self.bi[j])
                tmp += len(overlap) / (len(self.bi[i]) + len(self.bi[j]) - len(overlap))
            tmp /= len(list_bun2)
            ret += tmp
        ret /= len(list_bun1)
        return ret

    def __call__(self, scores, ground_truth):
        # is_hit = get_is_hit(scores, ground_truth, self.topk)
        # is_hit = is_hit.sum(dim=1)
        # num_pos = ground_truth.sum(dim=1)
        # self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        # self._sum += (is_hit/(2 * self.topk-is_hit)).sum().item()
        row_id, col_id = torch.topk(scores, self.topk)
        is_hit = get_is_hit(scores, ground_truth, self.topk)
        num_pos = is_hit.sum(dim=1)
        row, col = np.where(is_hit.cpu().numpy() == 1)
        # gold_bun = col_id[row, col]
        gold_bun = []
        # for i in range(len(col_id)):
        #     tmp = [col_id[i][j] for j in range(len(col_id[i])) if is_hit[i][j] == 1]
        #     gold_bun.append(tmp)
        for i in range(len(ground_truth)):
            gold_bun.append(np.where(ground_truth[i].cpu().numpy() == 1)[0])
        print(gold_bun, col_id)
        for i in range(len(row_id)):
            self._sum += self.cal_overlap(col_id[i], gold_bun[i])
        self._cnt = scores.shape[0] - (num_pos == 0).sum().item()
        
class NDCG(_Metric):
    '''
    NDCG in top-k samples
    In this work, NDCG = log(2)/log(1+hit_positions)
    '''

    def DCG(self, hit, device=torch.device('cpu')):
        hit = hit/torch.log2(torch.arange(2, self.topk+2,
                                          device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(self, num_pos):
        hit = torch.zeros(self.topk, dtype=torch.float)
        hit[:num_pos] = 1
        return self.DCG(hit)

    def __init__(self, topk):
        super().__init__()
        self.topk = topk
        self.IDCGs = torch.empty(1 + self.topk, dtype=torch.float)
        self.IDCGs[0] = 1  # avoid 0/0
        for i in range(1, self.topk + 1):
            self.IDCGs[i] = self.IDCG(i)

    def get_title(self):
        return "NDCG@{}".format(self.topk)

    def __call__(self, scores, ground_truth):
        device = scores.device
        is_hit = get_is_hit(scores, ground_truth, self.topk)
        num_pos = ground_truth.sum(dim=1).clamp(0, self.topk).to(torch.long).to('cpu')
        dcg = self.DCG(is_hit, device)
        idcg = self.IDCGs[num_pos]
        ndcg = dcg/idcg.to(device)
        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += ndcg.sum().item()


class MRR(_Metric):
    '''
    Mean reciprocal rank in top-k samples
    '''

    def __init__(self, topk):
        super().__init__()
        self.topk = topk
        self.denominator = torch.arange(1, self.topk+1, dtype=torch.float)

    def get_title(self):
        return "MRR@{}".format(self.topk)

    def __call__(self, scores, ground_truth):
        device = scores.device
        is_hit = get_is_hit(scores, ground_truth, self.topk)
        is_hit /= self.denominator.to(device)
        first_hit_rr = is_hit.max(dim=1)[0]
        num_pos = ground_truth.sum(dim=1)
        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += first_hit_rr.sum().item()