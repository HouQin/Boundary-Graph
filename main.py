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


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--dataset", help="dataset", type=str, default='photo')
    parse.add_argument("-l", "--labelrate", help="labeled data for train per class", type=int, default=20)
    parse.add_argument("-t", "--type", help="model for training, (PPNP=0, GNN-LF=1, GNN-HF=2)", type=int, default=0)
    parse.add_argument("--train_labelrate", help="labeled rate of training set", type=float, default=0.48)
    parse.add_argument("--val_labelrate", help="labeled data of validation set", type=float, default=0.32)
    parse.add_argument("--test_labelrate", help="labeled data of testing set", type=float, default=0.2)
    parse.add_argument("--seed", type=int, default=123, help="random seed")
    parse.add_argument("--random_seed", help="random seed", type=bool, default=False)
    parse.add_argument("-f", "--form", help="closed/iter form models (closed=0, iterative=1)", type=int, default=1)
    parse.add_argument('--cpu', action='store_true')
    parse.add_argument("--device", help="GPU device", type=str, default="1")
    parse.add_argument("--niter", help="times for iteration", type=int, default=10)
    parse.add_argument("--num_heads", help="multi heads for attention", type=int, default=1)
    parse.add_argument("--reg_lambda", help="regularization", type=float, default=0.005)
    parse.add_argument("--lr", help="learning rate", type=float, default=0.01)
    parse.add_argument("--dropout", help="learning rate", type=float, default=0.8)
    parse.add_argument("--runs", help="learning rate", type=int, default=10)
    parse.add_argument('--npow', type=int, default=0, help="for APGNN gap")

    args = parse.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device
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
        graph, idx_np = load_new_data(args.dataset, args.train_labelrate, args.val_labelrate, args.test_labelrate, args.random_seed)
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
    # fea_tensor_np = normalize_attributes(graph.attr_matrix)
    # fea_tensor = matrix_to_torch(fea_tensor_np)
    # fea_tensor = fea_tensor.to(device)

    print_interval = 100
    test = True

    propagation = []
    
    para_list = [0.9]

    results = []

    i_tot = 0
    # average_time: 每次实验跑average_time次取平均
    average_time = args.runs
    for _ in tqdm(range(average_time)):
        i_tot += 1

        para1 = 1 / (1 + args.niter)
        for para2 in para_list:
            if args.type == 0:
                model_type = "PPNP"
                if args.form == 0:
                    model_form = "closed"
                    propagation = PPRExact(graph.adj_matrix, alpha=para1)
                else:
                    model_form = "itera"
                    propagation = PPRPowerIteration(graph.adj_matrix, num_heads=args.num_heads, nclasses=max(graph.labels)+1, niter=args.niter, npow=0, nalpha=1, device=device)

            model_args = {
                'hiddenunits': [64],
                'drop_prob': args.dropout,
                'propagation': propagation}

        logging_string = f"Iteration {i_tot} of {average_time}"
        print(logging_string)
        _, result = train_model(idx_np, args.dataset, GNNs, graph, model_args, args.lr, args.reg_lambda, stopping_args, test, device, None, print_interval)
        results.append({})
        results[-1]['stopping_accuracy'] = result['early_stopping']['accuracy']
        results[-1]['valtest_accuracy'] = result['valtest']['accuracy']
        results[-1]['runtime'] = result['runtime']
        results[-1]['runtime_perepoch'] = result['runtime_perepoch']
        tmp = propagation.linear1.weight.t().unsqueeze(1).squeeze()
        print(torch.tanh(tmp))
        print(result['valtest']['accuracy']*100)

    result_df = pd.DataFrame(results)
    result_df.head()

    stopping_acc = calc_uncertainty(result_df['stopping_accuracy'])
    valtest_acc = calc_uncertainty(result_df['valtest_accuracy'])
    runtime = calc_uncertainty(result_df['runtime'])
    runtime_perepoch = calc_uncertainty(result_df['runtime_perepoch'])

    f = open(str(args.dataset) + '_labelrate_' + str(args.labelrate) + '_model_' + str(model_type) + '_form_' + str(model_form) + '.txt','a+')

    print('beta is :' + str(args.niter))
    print("model_" + str(model_type) + "_form_" + str(model_form)  + "\n" 
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


    f.write("\nmodel_" + str(model_type) + "_form_" + str(model_form)  + "\n" 
          "Early stopping: Accuracy: {:.2f} ± {:.2f}%\n"
          "{}: ACC: {:.2f} ± {:.2f}%\n"
          "Runtime: {:.3f} ± {:.3f} sec, per epoch: {:.2f} ± {:.2f}ms\n\n"
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

