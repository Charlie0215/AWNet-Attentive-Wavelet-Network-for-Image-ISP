import os

class trainConfig:
    learning_rate = [2e-4, 1e-4, 5e-5, 2e-5, 1e-5,
                    8e-6, 7e-6, 6e-6, 5e-6, 4e-6,
                    3e-6, 2e-6, 1e-6, 2e-6, 1.5e-6]
    print_loss = False
    # pre_lr = 0.0001

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

    teacher_reload = True
    teacher_weight = './best_weight'
    if not os.path.exists(save_best):
        os.makedirs(save_best)