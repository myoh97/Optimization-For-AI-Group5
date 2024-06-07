from __future__ import print_function, absolute_import
import time

import torch.nn as nn
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, \
    CrossEntropyLabelSmooth_weighted, SoftTripletLoss_weight
from .utils.meters import AverageMeter
from .utils.my_tools import *
from reid.metric_learning.distance import cosine_similarity, cosine_distance


class Trainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        self.T = 2
        
    def train(self, args, epoch, data_loader_train, optimizer, training_phase, loss_plate=[], train_iters=200):

        self.model.train()

        end = time.time()

        for i in range(train_iters):            
            train_inputs = data_loader_train.next()

            s_inputs, targets, cids, domains = self._parse_data(train_inputs)

            s_features, bn_features, s_cls_out= self.model(s_inputs, domains, training_phase)
            
            loss_ce, loss_tp = self._forward(s_features, s_cls_out, targets)
            loss = loss_ce + loss_tp
            loss_plate.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            t = time.time() - end
            # batch_time.update()
            end = time.time()
            
            if (i + 1) == train_iters:
                print('Epoch: [{}][{}/{}]\t'
                    'Time {:.3f} \t'
                    'Loss_ce {:.3f} \t'
                    'Loss_tp {:.3f} \t'
                    .format(epoch, i + 1, train_iters,
                            t,
                            loss_ce,
                            loss_tp))
    
        return loss_plate
    
    def _parse_data(self, inputs):
        imgs, _, pids, cids, domains = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets, cids, domains

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        loss_tr = self.criterion_triple(s_features, s_features, targets)
        return loss_ce, loss_tr
