# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2020-12-05 19:02
# Description:  
#--------------------------------------------
import os
import time
import argparse
import random
import torch
import numpy as np
from torch import optim
from concurrent.futures import ProcessPoolExecutor
from src.model import MF, MLP, GMF, NeuMF
from src.utils import WorkerInitObj
from src.dataset import ml_1mTrainDataLoader, ml_1mTestData
from src.evaluate import evaluate

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization.")
    parser.add_argument('--epoch', default=32, type=int, help='The epoch of train')
    parser.add_argument('--input_dir', default=None, type=str, required=True, help="The train input data dir.")
    parser.add_argument('--output_dir', default=None, type=str, required=True, help="The output of checkpoint dir.")
    parser.add_argument('--train_batch_size', default=32, type=int, help="Total batch size for training.")
    parser.add_argument('--layer_hiddens_mlp', nargs='+', type=int, default=[32, 16, 8],  help='The layer hiddens for mlp.')
    parser.add_argument('--embedding_dim', default=8, type=int, help='The dimension of user and item embeddings')
    parser.add_argument('--use_cuda', default=False, action='store_true', help='Whether use gpu')
    parser.add_argument('--devices', type=str, default='0,1', help='The devices id of gpu')
    parser.add_argument('--user_nums', type=int, required=True, help='The number of users.')
    parser.add_argument('--item_nums', type=int, required=True, help='The number of items.')
    parser.add_argument('--mlp_dim_rate_neumf', type=float, default=0.5, help='The rate of split embdding for mlp.')
    parser.add_argument('--learning_rate', default=1.5e-3, type=float, help="The initial learning rate for optimizer")
    parser.add_argument('--topk', default=3, type=int, help='evaluate top k result')
    parser.add_argument('--eval_freq', default=100000, type=int, help='The freq of eval test set')
    parser.add_argument('--log_freq', default=200, type=int, help='The freq of print log')
    parser.add_argument('--num_negs', default=3, type=int, help='The num of negative for every user')
    parser.add_argument("--init_checkpoint", default=None, type=str, help="The initial checkpoint to start training from.")

    args = parser.parse_args()

    return args

def setup_training(args):

    args.multi_gpu = False
    if args.use_cuda:
        assert torch.cuda.is_available()

        args.device_ids = [int(id) for id in args.devices.strip().split(',')]
        if len(args.device_ids) > 1:
            assert torch.cuda.device_count() >= len(args.device_ids)
            os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
            device = torch.device('cuda:0')
            args.multi_gpu = True
        else:
            device = torch.device('cuda:{}'.format(args.devices))
            os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    else:
        device = torch.device('cpu')

    args.train_path = os.path.join(args.input_dir, 'ml-1m.train.rating')
    args.test_pos_path = os.path.join(args.input_dir, 'ml-1m.test.rating')
    args.test_neg_path = os.path.join(args.input_dir, 'ml-1m.test.negative')

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    return device, args

def prepare_model_and_optimizer(args, device):
    user_nums = args.user_nums
    item_nums = args.item_nums
    embedding_dim = args.embedding_dim
    layer_hiddens = args.layer_hiddens_mlp
    mlp_dim_rate = args.mlp_dim_rate_neumf
    lr = args.learning_rate

    #model
    # model = MF(user_nums, item_nums, embedding_dim)
    # model = MLP(user_nums, item_nums, embedding_dim, layer_hiddens)
    # model = GMF(user_nums, item_nums, embedding_dim)
    model = NeuMF(user_nums, item_nums, embedding_dim, layer_hiddens)

    args.model_name = model.__class__.__name__
    if args.init_checkpoint is not None and os.path.isfile(args.init_checkpoint):
        checkpoint_name = args.init_checkpoint.split('/')[-1].split('_')[0]
        if checkpoint_name == args.model_name:
            checkpoint = torch.load(args.init_checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            raise ValueError('expect {} model, but get {} model'.format(args.model_name, checkpoint_name))

    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    criterion = torch.nn.BCELoss()
    if args.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    model.to(device)
    criterion.to(device)

    return model, optimizer, criterion

def prepare_test_data(args):
    t_pos, t_neg = ml_1mTestData(args.test_pos_path, args.test_neg_path)
    samples = []
    for (user, pos), negs in zip(t_pos, t_neg):
        samples.append([user, pos, negs]) #first item is pos, others are negs
    return samples

def main():
    args = parse_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    worker_init = WorkerInitObj(args.seed)
    device, args = setup_training(args)
    test_data = prepare_test_data(args)
    model, optimizer, criterion = prepare_model_and_optimizer(args, device)

    pool = ProcessPoolExecutor(1)
    train_iter = ml_1mTrainDataLoader(path=args.train_path,
                                      num_negs= args.num_negs,
                                      batch_size=args.train_batch_size,
                                      seed=args.seed,
                                      worker_init=worker_init)

    print('-'*50 + 'args' + '-'*50 )
    for k in list(vars(args).keys()):
        print('{0}: {1}'.format(k, vars(args)[k]))
    print('-'*30)
    print(model)
    print('-'*50 + 'args' + '-'*50 )

    global_step = 0
    global_HR = 0.0; global_NDCG = 0.0

    s_time_train = time.time()
    for epoch in range(args.epoch):

        dataset_future = pool.submit(ml_1mTrainDataLoader,
                                     args.train_path,
                                     args.num_negs,
                                     args.train_batch_size,
                                     args.seed,
                                     worker_init)

        for step, batch in enumerate(train_iter):

            model.train()
            batch = [t.to(device) for t in batch]
            users, items, labels = batch

            logits = model(users, items)
            loss = criterion(logits, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #evaluate
            if global_step != 0 and global_step % args.eval_freq == 0:
                s_time_eval = time.time()
                model.eval()
                hits, ndcgs = evaluate(model, test_data, device, args.topk)
                e_time_eval = time.time()
                print('-' * 68)
                print('Epoch:[{0}] Step:[{1}] HR:[{2}] NDCG:[{3}] time:[{4}s]'.format(
                        epoch,
                        global_step,
                        format(hits, '.4f'),
                        format(ndcgs, '.4f'),
                        format(e_time_eval - s_time_eval, '.4f')))

                if hits > global_HR and ndcgs > global_NDCG:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_save_file = os.path.join(args.output_dir, "{}_hr_{}_ndcg_{}_step_{}_ckpt.pt".format(
                        args.model_name,
                        format(hits, '.4f'),
                        format(ndcgs, '.4f'),
                        global_step))

                    if os.path.exists(output_save_file):
                        os.system('rm -rf {}'.format(output_save_file))
                    torch.save({'model': model_to_save.state_dict(),
                                'name': args.model_name},
                                 output_save_file)
                    print('Epoch:[{0}] Step:[{1}] SavePath:[{2}]'.format(
                        epoch,
                        global_step,
                        output_save_file))
                    global_HR = hits; global_NDCG = ndcgs
                print('-' * 68)

             #log
            if global_step != 0 and global_step % args.log_freq == 0:
                e_time_train = time.time()
                print('Epoch:[{0}] Step:[{1}] Loss:[{2}] Lr:[{3}] time:[{4}s]'.format(
                    epoch,
                    global_step,
                    format(loss.item(), '.4f'),
                    format(optimizer.param_groups[0]['lr'], '.6'),
                    format(e_time_train - s_time_train, '.4f')))
                s_time_train = time.time()

            global_step += 1

        del train_iter
        train_iter = dataset_future.result(timeout=None)

if __name__ == '__main__':
    main()