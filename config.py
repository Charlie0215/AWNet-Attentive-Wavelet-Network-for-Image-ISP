import os


class trainConfig:
    learning_rate = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    print_loss = False

    batch_size = 3
    epoch = 50
    pretrain = True

    data_dir = '/home/charliedai/aim2020/Dataset'

    checkpoints = './saved_checkpoints'
    if not os.path.exists(checkpoints):
        os.makedirs(checkpoints)
    save_best = './best_weight'
    if not os.path.exists(save_best):
        os.makedirs(save_best)
