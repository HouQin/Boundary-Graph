import os
import pandas as pd
import numpy as np
import argparse
from utils import *
from model import GNNs
from training import train_model
from earlystopping import stopping_args
from propagation import *
from load_data import *
from tqdm import tqdm

from training import normalize_attributes
from utils import matrix_to_torch

from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical
from skopt.utils import use_named_args
import skopt

search_space = list()
search_space.append(Real(1e-7, 5e-2, name='reg_lambda'))
search_space.append(Real(1e-7, 5e-2, name='reg_gamma'))
search_space.append(Real(0.001, 0.05, name='lr'))
search_space.append(Real(0.0, 0.95, name='dropout'))
search_space.append(Integer(0, 4, name='npow'))
search_space.append(Integer(0, 4, name='npow_attn'))
search_space.append(Integer(1, 10, name='niter'))
search_space.append(Integer(1, 10, name='niter_attn'))
search_space.append(Integer(1, 8, name='num_heads'))
search_space.append(Integer(8, 64, name='att_hidden'))

@use_named_args(search_space)
def evaluate_model(**params):
    args.reg_lambda = params['reg_lambda']
    args.reg_gamma = params['reg_gamma']
    args.lr = params['lr']
    args.dropout = params['dropout']
    args.npow = params['npow']
    args.npow_attn = params['npow_attn']
    args.niter = params['niter']
    args.niter_attn = params['niter_attn']
    args.num_heads = params['num_heads']
    args.att_hidden = params['att_hidden']

    print(args)

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    if args.dataset == 'acm':
        graph, idx_np = load_new_data_acm(args.labelrate)
    elif args.dataset == 'wiki':
        # graph, idx_np = load_new_data_wiki(args.labelrate)
        # # args.lr = 0.03
        # # args.reg_lambda = 5e-4
        graph, idx_np = load_fixed_wikics()
    elif args.dataset == 'ms':
        graph, idx_np = load_new_data_ms(args.labelrate)
    elif args.dataset in ['chameleon', 'squirrel', 'cornell', 'texas', 'wisconsin', 'film']:
        graph, idx_np = load_new_data(args.dataset, args.train_labelrate, args.val_labelrate, args.test_labelrate,
                                      args.random_seed)
    elif args.dataset in ['computers', 'photo']:
        graph, idx_np = load_Amazon(args.dataset)
    else:
        if args.dataset == 'cora':
            feature_dim = 1433
        elif args.dataset == 'citeseer':
            feature_dim = 3703
        elif args.dataset == 'pubmed':
            feature_dim = 500
        graph, idx_np = load_new_data_tkipf(args.dataset, feature_dim, args.labelrate)

    if args.dataset in ['chameleon', 'squirrel', 'cornell', 'texas', 'wisconsin', 'film', 'wiki', 'computers', 'photo']:
        fea_tensor = graph.attr_matrix
    else:
        fea_tensor = graph.attr_matrix.todense()
    fea_tensor = torch.from_numpy(fea_tensor).float().to(device)

    print_interval = 100
    test = True

    propagation = []
    results = []

    i_tot = 0
    # average_time: 每次实验跑average_time次取平均
    average_time = args.runs
    for _ in tqdm(range(average_time)):
        i_tot += 1

        propagation = PPRPowerIteration(graph.adj_matrix, num_heads=args.num_heads,
                                        nclasses=max(graph.labels) + 1, niter=args.niter, npow=args.npow, npow_attn=args.npow_attn, nalpha=1,
                                        num_feature=fea_tensor.shape[1], num_hidden=args.att_hidden,
                                        device=device, niter_attn=args.niter_attn)

        model_args = {
            'hiddenunits': [64],
            'drop_prob': args.dropout,
            'propagation': propagation}

        logging_string = f"Iteration {i_tot} of {average_time}"

        _, result = train_model(idx_np, args.dataset, GNNs, graph, model_args, args.lr, args.reg_lambda, args.reg_gamma, stopping_args,
                                test, device, None, print_interval)
        results.append({})
        results[-1]['stopping_accuracy'] = result['early_stopping']['accuracy']
        results[-1]['valtest_accuracy'] = result['valtest']['accuracy']
        results[-1]['runtime'] = result['runtime']
        results[-1]['runtime_perepoch'] = result['runtime_perepoch']
        tmp = propagation.linear1.weight.t().unsqueeze(1).squeeze()

    result_df = pd.DataFrame(results)
    result_df.head()

    stopping_acc = calc_uncertainty(result_df['stopping_accuracy'])
    valtest_acc = calc_uncertainty(result_df['valtest_accuracy'])
    runtime = calc_uncertainty(result_df['runtime'])
    runtime_perepoch = calc_uncertainty(result_df['runtime_perepoch'])

    print(
        "Early stopping: Accuracy: {:.2f} ± {:.2f}%\n"
        "{}: ACC: {:.2f} ± {:.2f}%\n"
        "Runtime: {:.3f} ± {:.3f} sec, per epoch: {:.2f} ± {:.2f}ms\n"
          .format(
            stopping_acc['mean'] * 100,
            stopping_acc['uncertainty'] * 100,
            'Test' if test else 'Validation',
            valtest_acc['mean'] * 100,
            valtest_acc['uncertainty'] * 100,
            runtime['mean'],
            runtime['uncertainty'],
            runtime_perepoch['mean'] * 1e3,
            runtime_perepoch['uncertainty'] * 1e3,
        ))

    return 1.0 - valtest_acc['mean']

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--dataset", help="dataset", type=str, default='pubmed')
    parse.add_argument("-l", "--labelrate", help="labeled data for train per class", type=int, default=60)
    parse.add_argument("-t", "--type", help="model for training, (PPNP=0, GNN-LF=1, GNN-HF=2)", type=int, default=0)
    parse.add_argument("--train_labelrate", help="labeled rate of training set", type=float, default=0.48)
    parse.add_argument("--val_labelrate", help="labeled data of validation set", type=float, default=0.32)
    parse.add_argument("--test_labelrate", help="labeled data of testing set", type=float, default=0.2)
    parse.add_argument("--seed", type=int, default=123, help="random seed")
    parse.add_argument("--random_seed", help="random seed", type=bool, default=False)
    parse.add_argument("-f", "--form", help="closed/iter form models (closed=0, iterative=1)", type=int, default=1)
    parse.add_argument('--cpu', action='store_true')
    parse.add_argument("--device", help="GPU device", type=str, default="2")
    parse.add_argument("--niter", help="times for iteration", type=int, default=10)
    parse.add_argument("--niter_attn", help="times for iteration", type=int, default=10)
    parse.add_argument("--num_heads", help="multi heads for attention", type=int, default=1)
    parse.add_argument("--reg_lambda", help="regularization", type=float, default=0.005)
    parse.add_argument("--reg_gamma", help="regularization", type=float, default=0.005)
    parse.add_argument("--lr", help="learning rate", type=float, default=0.01)
    parse.add_argument("--dropout", help="learning rate", type=float, default=0.8)
    parse.add_argument("--runs", help="learning rate", type=int, default=10)
    parse.add_argument('--npow', type=int, default=0, help="for APGNN gap")
    parse.add_argument('--npow_attn', type=int, default=0, help="for APGNN gap with attention")
    parse.add_argument("--att_hidden", help="attention hidden layer dimension", type=int, default=64)

    args = parse.parse_args()

    result = skopt.gp_minimize(evaluate_model, search_space, verbose=True, n_calls=256)

    print('Best Accuracy: %.3f' % (1.0 - result.fun))
    print('Best Parameters: %s' % (result.x))