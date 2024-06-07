from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys

from torch.backends import cudnn
import copy
import torch.nn as nn
import random
import json

# from reid import datasets
import reid.datasets as datasets
from reid.evaluators import Evaluator, extract_gallery
from reid.utils.data import IterLoader
from reid.utils.data.sampler import RandomMultipleGallerySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid.utils.my_tools import *
from reid.models.resnet import build_resnet_backbone
from reid.models.layers import DataParallel
from reid.trainer import Trainer

def get_data(name, data_dir, height, width, batch_size, workers, num_instances, eval_only=False):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = sorted(dataset.train)

    iters = int(len(train_set) / batch_size)
    num_classes = dataset.num_train_pids

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None

    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=False, drop_last=True), length=iters)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=False)

    init_loader = DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=test_transformer),
                             batch_size=128, num_workers=workers,shuffle=False, pin_memory=False, drop_last=False)
    if eval_only:
        return dataset, num_classes, train_loader, test_loader, init_loader
    
    # ======
    query_loader = DataLoader(
        Preprocessor(list(dataset.query), root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=False)
    
    gallery_loader = DataLoader(
        Preprocessor(list(dataset.gallery), root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=False)
    
    return dataset, num_classes, train_loader, test_loader, init_loader, query_loader, gallery_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    return test_loader

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)

def main_worker(args):
    cudnn.benchmark = True
    log_name = 'log.txt'
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    else:
        log_dir = osp.dirname(args.resume)
        sys.stdout = Logger(osp.join(log_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))
    print("=> Training {}".format(args.dataset))

    # Create data loaders
    dataset_db1, num_classes_db1, train_loader_db1, test_loader_db1, _, query_loader_db1, gallery_loader_db1= \
        get_data(args.dataset, args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    # Create model
    model = build_resnet_backbone(num_class=num_classes_db1, depth='50x', args=args)
    model.cuda()
    model = DataParallel(model)

    # Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['mAP']
        print("=> Start epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))

    # Evaluator
    start_epoch = 0
    evaluator = Evaluator(model)

    # Opitimizer initialize
    params = []
    for key, value in model.named_params(model):
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(params)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(params)
    elif args.optim == 'adadelta':
        optimizer = torch.optim.Adadelta(params)
    elif args.optim == 'adagrad':
        optimizer = torch.optim.Adagrad(params)
    elif args.optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(params)
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(params)
    elif args.optim == 'asgd':
        optimizer = torch.optim.ASGD(params)

    lr_scheduler = WarmupMultiStepLR(optimizer, [40, 70], gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

    # Start training
    # Train db1
    loss_plate = []
    trainer = Trainer(model, num_classes_db1, margin=args.margin)
    for epoch in range(start_epoch, args.epochs):

        train_loader_db1.new_epoch()
        loss_plate = trainer.train(args, epoch, train_loader_db1, optimizer, training_phase=1, train_iters=len(train_loader_db1), loss_plate=loss_plate)
        lr_scheduler.step()

        if ((epoch + 1) == args.epochs):
            _, mAP = evaluator.evaluate(test_loader_db1, dataset_db1.query, dataset_db1.gallery, cmc_flag=False)

            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': mAP,
            }, True, fpath=osp.join(args.logs_dir, f'{args.dataset}_checkpoint.pth.tar'))

            print('Finished epoch {:3d}  {} mAP: {:5.1%} '.format(epoch, args.dataset, mAP))

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import pickle
    
    plt.plot(loss_plate)
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.tight_layout()
    plt.savefig(osp.join(args.logs_dir, 'loss_curve.png'))
    with open(osp.join(args.logs_dir, 'loss.pkl'), 'wb') as f:
        pickle.dump(loss_plate, f)
    print('finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    
    # order
    parser.add_argument('--dataset', type=str, choices=['market1501', 'dukemtmc', 'cuhk_sysu'], default='market1501')
    
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    
    # optimizer
    parser.add_argument('--optim', type=str, default='adam', choices=['adadelta', 'adagrad', 'adam', 'sgd', 'rmsprop', 'adamw', 'asgd'])
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70],
                        help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=200)
    parser.add_argument('--margin', type=float, default=0.3, help='margin for the triplet loss with batch hard')
    
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/root/dataset/ReID')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--rr-gpu', action='store_true',
                        help="use GPU for accelerating clustering")
    
    main()