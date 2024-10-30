import os
import numpy as np

# reg_lambda = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]

# for threshold in np.arange(6, 11, 1):
# # threshold = 5
#     lr = 1e-2
#     npow = 0
#     for dropout in np.arange(0.5, 1, 0.1):
#         # dropout = 0.9
#         for i in range(len(reg_lambda)):
#             for niter in np.arange(10, 21, 1):
#                 os.system('/home/lhq/.conda/envs/GCN/bin/python main.py --dataset citeseer --type 0 --form 1 --device 0 --niter {} --reg_lambda {} --lr {} --dropout {} --runs 10 --threshold {} --npow {} >> mvgnn_citeseer_9_12.txt'.format(niter, reg_lambda[i], lr, dropout, threshold, npow))

# for threshold in np.arange(0, 5, 1):
# # threshold = 5
#     lr = 1e-2
#     npow = 0
#     for dropout in np.arange(0.5, 1, 0.1):
#         # dropout = 0.9
#         for i in range(len(reg_lambda)):
#             for niter in np.arange(10, 21, 1):
#                 os.system('/home/lhq/.conda/envs/GCN/bin/python main.py --dataset texas --type 0 --form 1 --device 1 --niter {} --reg_lambda {} --lr {} --dropout {} --runs 10 --threshold {} --npow {} >> mvgnn_texas_9_12.txt'.format(niter, reg_lambda[i], lr, dropout, threshold, npow))


# os.system('/home/lhq/.conda/envs/GCN/bin/python main.py --dataset cora --type 0 --form 1 --device 0 --niter 19 --reg_lambda 5e-4 --lr 1e-2 --dropout 0.8 --runs 10 --threshold 0.1 --npow 0 >> mvgnn_cora_9_6_2.txt'.format(niter, reg_lambda[i], lr, dropout, threshold, npow))

# for i in range(100):
#     os.system('/home/lhq/.conda/envs/GCN/bin/python main.py --dataset cora --type 0 --form 1 --device 0 --niter 19 --reg_lambda 1e-4 --lr 1e-2 --dropout 0.8 --runs 10 --threshold 0.1 --npow 0 >> mvgnn_cora_9_10.txt')

# for i in range(100):
#     os.system('/home/lhq/.conda/envs/GCN/bin/python main.py --dataset citeseer --type 0 --form 1 --device 1 --niter 16 --reg_lambda 0.05 --lr 1e-2 --dropout 0.9 --runs 10 --threshold 0.1 --npow 0 >> mvgnn_citeseer_9_10.txt')

for niter in np.arange(10, 21, 1):
    for num_heads in [1, 2, 4, 8]:
        for dropout in np.arange(0.3, 1, 0.1):
            for reg_lambda in [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]:
                os.system('/home/lhq/.conda/envs/GCN/bin/python main.py --dataset cora --type 0 --form 1 --device 0 --niter {} --num_heads {} --reg_lambda {} --lr 1e-2 --dropout {} --runs 10 --npow 0 >> gkd_cora_11_6.txt'.format(niter, num_heads, reg_lambda, dropout))
