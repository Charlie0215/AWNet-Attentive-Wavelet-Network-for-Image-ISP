import os

class trainConfig:
    model_save = './saved_checkpoints/'
    if not os.path.exists(model_save):
        os.makedirs(model_save)
    
    print_loss = False

    pre_lr = 0.0001
    batch_size = 8
    epoch = 150
    pretrain = False

    data_dir = '/home/charliedai/aim2020/Dataset'