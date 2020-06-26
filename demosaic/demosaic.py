import imageio
import os
import numpy as np
import cv2

def extract_bayer_channels(raw):

    # Reshape the input bayer image

    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    ch_B = cv2.resize(ch_B, (ch_B.shape[1]*2, ch_B.shape[0]*2))
    ch_R = cv2.resize(ch_R, (ch_R.shape[1]*2, ch_R.shape[0]*2))
    ch_Gb = cv2.resize(ch_Gb, (ch_Gb.shape[1]*2, ch_Gb.shape[0]*2))
    ch_Gr = cv2.resize(ch_Gr, (ch_Gr.shape[1]*2, ch_Gr.shape[0]*2))

    ch_G = ch_Gb / 2 + ch_Gr / 2  
    RAW_combined = np.dstack((ch_B, ch_G, ch_R))
    RAW_norm = RAW_combined.astype(np.float32) / (3 * 255)

    return RAW_norm

def process_image(path, save, file):
    raw_image = np.asarray(imageio.imread(path))
    demosaic = extract_bayer_channels(raw_image)
    save_path = './{}/{}'.format(save, file)
    print(save_path)
    cv2.imwrite(save_path, demosaic*255)
    
def batch_process(root, save):
    # from joblib import Parallel, delayed
    for file in os.listdir(root):
        path = os.path.join(root, file)
        process_image(path, save, file)


if __name__ == '__main__':
    import rawpy
    path1 = '/home/charliedai/aim2020/Dataset/train/huawei_raw/'
    path2 = '/home/charliedai/aim2020/Dataset/test/huawei_raw/'
    batch_process(path1, 'train_vis')
    batch_process(path2, 'test_vis')