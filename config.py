import os

class trainConfig:

    print_loss = False
    pre_lr = 0.0001
    batch_size = 8
    epoch = 150
    pretrain = False

    data_dir = '/home/charliedai/aim2020/Dataset'
    checkpoints = './saved_checkpoints'
    if not os.path.exists(checkpoints):
        os.makedirs(checkpoints)
    save_best = './best_weight'
    if not os.path.exists(save_best):
        os.makedirs(save_best)