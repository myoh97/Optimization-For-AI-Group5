from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys

from torch.backends import cudnn
import copy
import torch.nn as nn
import random

from reid import datasets
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
import warnings
warnings.filterwarnings('ignore')
import json
def get_data_2(name, data_dir, height, width, batch_size, workers, add_num=0):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])


    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    query_loader = DataLoader(
        Preprocessor(list(dataset.query), root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=False)
    
    gallery_loader = DataLoader(
        Preprocessor(list(dataset.gallery), root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=False)
    
    return dataset, query_loader, gallery_loader

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
    db1, db2, db3, db4 = args.order.split(",")
    print("=> Training Order: {} => {} => {} => {}".format(db1, db2, db3, db4))

    # Create data loaders
    dataset_db1, num_classes_db1, train_loader_db1, test_loader_db1, _, query_loader_db1, gallery_loader_db1= \
        get_data(db1, args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_db2, num_classes_db2, train_loader_db2, test_loader_db2, init_loader_db2, query_loader_db2, gallery_loader_db2 = \
        get_data(db2, args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_db3, num_classes_db3, train_loader_db3, test_loader_db3, init_loader_db3, query_loader_db3, gallery_loader_db3 = \
        get_data(db3, args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_db4, num_classes_db4, train_loader_db4, test_loader_db4, init_loader_db4, query_loader_db4, gallery_loader_db4 = \
        get_data(db4, args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)


    # Create model
    model = build_resnet_backbone(num_class=num_classes_db1, depth='50x')
    model.cuda()
    model = DataParallel(model)

    model.eval()

    evaluator = Evaluator(model)
    try:
        checkpoint = load_checkpoint(osp.join(args.resume, 'db1_checkpoint.pth.tar'))
    except:
        checkpoint = load_checkpoint(osp.join('logs/ptkp/baseline', 'market_checkpoint.pth.tar'))
    copy_state_dict(checkpoint['state_dict'], model)
    # print("\n=> Market Performance (old, old)")

    gallery_features_db1, gallery_pid_market = extract_gallery(model, gallery_loader_db1)
    
    # duke
    try:
        checkpoint = load_checkpoint(osp.join(args.resume, 'db2_checkpoint.pth.tar'))
    except:
        checkpoint = load_checkpoint(osp.join(args.resume, 'duke_checkpoint.pth.tar'))
    copy_state_dict(checkpoint['state_dict'], model)
    # print("\n=> Duke Performance (old, old)")
    gallery_features_db2, gallery_pid_duke = extract_gallery(model, gallery_loader_db2)
            

    # cuhksysu
    try:
        checkpoint = load_checkpoint(osp.join(args.resume, 'db3_checkpoint.pth.tar'))
    except:
        checkpoint = load_checkpoint(osp.join(args.resume, 'cuhksysu_checkpoint.pth.tar'))
    copy_state_dict(checkpoint['state_dict'], model)
    # print("\n=> CUHK Performance (old, old)")
    gallery_features_db3, gallery_pid_cuhksysu = extract_gallery(model, gallery_loader_db3)

    # msmt
    try:
        checkpoint = load_checkpoint(osp.join(args.resume, 'db4_checkpoint.pth.tar'))
    except:
        checkpoint = load_checkpoint(osp.join(args.resume, 'msmt17_checkpoint.pth.tar'))
        
    copy_state_dict(checkpoint['state_dict'], model)
    
    mAP_db1_ll = evaluator.evaluate(test_loader_db1, dataset_db1.query, dataset_db1.gallery, cmc_flag=False)
    mAP_db2_ll = evaluator.evaluate(test_loader_db2, dataset_db2.query, dataset_db2.gallery, cmc_flag=False)
    mAP_db3_ll = evaluator.evaluate(test_loader_db3, dataset_db3.query, dataset_db3.gallery, cmc_flag=False)
    
    mAP_db4 = evaluator.evaluate(test_loader_db4, dataset_db4.query, dataset_db4.gallery,
                                        cmc_flag=True)
    # print("\n=> Market Performance (new, new)")
    # mAP_market = evaluator.evaluate(test_loader_market, dataset_market.query, dataset_market.gallery,
                                # cmc_flag=True)
    # print("\n=> Duke Performance (new, new)")
    # mAP_duke = evaluator.evaluate(test_loader_duke, dataset_duke.query, dataset_duke.gallery,
                                        # cmc_flag=True)
    # print("\n=> CUHK Performance (new, new)")
    # mAP_cuhk = evaluator.evaluate(test_loader_cuhksysu, dataset_cuhksysu.query, dataset_cuhksysu.gallery,
                                        # cmc_flag=True)
    
    # print("\n=> MSMT Performance (new, new)")
    # mAP_msmt = evaluator.evaluate(test_loader_msmt17, dataset_msmt17.query, dataset_msmt17.gallery,
                                        # cmc_flag=True)

    print('\n=>Testing compatibility')
    mAP_db1_comp = evaluator.evaluate_compatible(
        query_loader_db1, gallery_features_db1, dataset_db1.query, dataset_db1.gallery, cmc_flag=True)
    mAP_db2_comp = evaluator.evaluate_compatible(
        query_loader_db2, gallery_features_db2, dataset_db2.query, dataset_db2.gallery, cmc_flag=True)
    mAP_db3_comp = evaluator.evaluate_compatible(
        query_loader_db3, gallery_features_db3, dataset_db3.query, dataset_db3.gallery, cmc_flag=True)
    
    print("\n=> Saving results...")
    
    orders = [db1, db2, db3, db4]
    # rank_list = [rank1_db1_comp, rank1_db2_comp, rank1_db3_comp, rank1_db4]
    mAP_list = [mAP_db1_comp, mAP_db2_comp, mAP_db3_comp, mAP_db4]
    mAP_ll_list = [mAP_db1_ll, mAP_db2_ll, mAP_db3_ll, mAP_db4]
    res = {}
    res['order'] = args.order
    
    res['lifelong'] = {}
    res['lifelong']['rank-1'] = {}
    res['lifelong']['mAP'] = {}
    res['compatibility'] = {}
    res['compatibility']['rank-1'] = {}
    res['compatibility']['mAP'] = {}
    for idx, db in enumerate(orders):
        # res['compatibility']['rank-1'][db] = rank_list[idx]
        res['compatibility']['mAP'][db] = mAP_list[idx]
        res['lifelong']['mAP'][db] = mAP_ll_list[idx]
    
    with open(osp.join(args.resume, 'result.json'), 'w') as f:
        json.dump(res, f)
    print("=> Done: {}".format(osp.join(args.resume, 'result.json')))
    print('finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    dataset_list = ['dukemtmc', 'market1501', 'cuhk_sysu', 'msmt17']
    from itertools import permutations

    lst = [0, 1, 2, 3]
    all_permutations = permutations(lst)
    all_permutations = [list(item) for item in all_permutations]

    orders = [f"{dataset_list[item[0]]}_{dataset_list[item[1]]}_{dataset_list[item[2]]}_{dataset_list[item[3]]}" for item in all_permutations]
    
    # order
    parser.add_argument('--order', type=str, choices=orders, default='market1501,dukemtmc,cuhk_sysu,msmt17')
    
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-br', '--replay-batch-size', type=int, default=128)
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
    
    parser.add_argument('--w_prop', type=float, default=0.1)
    parser.add_argument('--temp_prop', type=float, default=1.0)
    parser.add_argument('--weighting', action='store_true')
    
    parser.add_argument('--wo_ptkp', action='store_true')
    parser.add_argument('--wo_prop', action='store_true')
    
    #
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/root/dataset/ReID')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--rr-gpu', action='store_true',
                        help="use GPU for accelerating clustering")
    
    main()