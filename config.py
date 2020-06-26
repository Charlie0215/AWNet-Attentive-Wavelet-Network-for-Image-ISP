import os

class trainConfig:
    learning_rate = [1e-4, 7e-5, 5e-5, 2e-5, 1e-5]
    print_loss = False
    # pre_lr = 0.0001

    batch_size = 12
    epoch = 50
    pretrain = False

    data_dir = '/home/charliedai/aim2020/Dataset'
    checkpoints = './saved_checkpoints'
    if not os.path.exists(checkpoints):
        os.makedirs(checkpoints)
    save_best = './best_weight'
    if not os.path.exists(save_best):
        os.makedirs(save_best)

    teacher_reload = True
    teacher_weight = './best_weight'
    if not os.path.exists(save_best):
        os.makedirs(save_best)