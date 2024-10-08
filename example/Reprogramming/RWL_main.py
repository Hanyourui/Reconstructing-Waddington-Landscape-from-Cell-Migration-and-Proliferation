# -*- coding = utf-8 -*-
# @Time : 2024/8/6 10:26
# @Author : Yourui Han
# @File : RWL_main.py
# @Software : PyCharm


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utility import *


if __name__ == '__main__':
    arguments = input_args()

    torch.enable_grad()
    random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)

    device = torch.device('cuda:' + str(arguments.gpu)
                          if torch.cuda.is_available() else 'cpu')
    # load dataset
    data_train = loaddata(arguments, device)
    integral_time = arguments.pseudo_time

    time_pts = range(len(data_train))
    leave_1_out = []
    train_time = [x for i, x in enumerate(time_pts) if i != leave_1_out]

    # model
    func = RWL(in_out_dim=data_train[0].shape[1], GRN_dim_hidden=arguments.GRN_dim_hidden,
               GRN_num_hiddens=arguments.GRN_num_hiddens, BRD_dim_hidden=arguments.BRD_dim_hidden,
               BRD_num_hiddens=arguments.BRD_num_hiddens, activation=arguments.activation,
               decrease_multipleint=arguments.decrease_multipleint, sparsity_param=arguments.sparsity_param).to(device)
    func.apply(initialize_weights)

    # configure training options
    options = {}
    options.update({'method': 'Dopri5'})
    options.update({'h': None})
    options.update({'rtol': 1e-3})
    options.update({'atol': 1e-5})
    options.update({'print_neval': False})
    options.update({'neval_max': 1000000})
    options.update({'safety': None})

    optimizer = optim.Adam(func.parameters(), lr=arguments.lr, weight_decay=0.01)
    lr_adjust = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[arguments.num_iters - 400, arguments.num_iters - 200],
                                               gamma=0.5, last_epoch=-1)
    mse = nn.MSELoss()

    LOSS = []
    L2_1 = []
    L2_2 = []
    Trans = []
    Sigma = []

    if arguments.save_dir is not None:
        if not os.path.exists(arguments.save_dir):
            os.makedirs(arguments.save_dir)
        ckpt_path = os.path.join(arguments.save_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded ckpt from {}'.format(ckpt_path))

    try:
        sigma_now = 1
        for itr in range(1, arguments.num_iters + 1):
            optimizer.zero_grad()

            loss, loss1, sigma_now, L2_value1, L2_value2 = train_model(mse, func, arguments, data_train, train_time,
                                                                       integral_time, sigma_now, options, device, itr)

            loss.backward()
            optimizer.step()
            lr_adjust.step()

            LOSS.append(loss.item())
            Trans.append(loss1[-1].mean(0).item())
            Sigma.append(sigma_now)
            L2_1.append(L2_value1.tolist())
            L2_2.append(L2_value2.tolist())

            print('Iter: {}, loss: {:.4f}'.format(itr, loss.item()))

            if itr % 100 == 0:
                ckpt_path = os.path.join(arguments.save_dir, 'ckpt_itr{}.pth'.format(itr))
                torch.save({'func_state_dict': func.state_dict()}, ckpt_path)
                print('Iter {}, Stored ckpt at {}'.format(itr, ckpt_path))


    except KeyboardInterrupt:
        if arguments.save_dir is not None:
            ckpt_path = os.path.join(arguments.save_dir, 'ckpt.pth')
            torch.save({
                'func_state_dict': func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))
    print('Training complete after {} iters.'.format(itr))

    ckpt_path = os.path.join(arguments.save_dir, 'ckpt.pth')
    torch.save({
        'func_state_dict': func.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'LOSS': LOSS,
        'TRANS': Trans,
        'L2_1': L2_1,
        'L2_2': L2_2,
        'Sigma': Sigma
    }, ckpt_path)
    print('Stored ckpt at {}'.format(ckpt_path))
