# -*- coding: utf-8 -*-
import argparse
import os

import cv2
import imageio
import numpy as np
import tqdm


def extract_bayer_channels(raw_image: np.ndarray) -> np.ndarray:
    # Reshape the input bayer image
    ch_B = raw_image[1::2, 1::2]
    ch_Gb = raw_image[0::2, 1::2]
    ch_R = raw_image[0::2, 0::2]
    ch_Gr = raw_image[1::2, 0::2]

    ch_B = cv2.resize(ch_B, (ch_B.shape[1] * 2, ch_B.shape[0] * 2))
    ch_R = cv2.resize(ch_R, (ch_R.shape[1] * 2, ch_R.shape[0] * 2))
    ch_Gb = cv2.resize(ch_Gb, (ch_Gb.shape[1] * 2, ch_Gb.shape[0] * 2))
    ch_Gr = cv2.resize(ch_Gr, (ch_Gr.shape[1] * 2, ch_Gr.shape[0] * 2))

    ch_G = ch_Gb / 2 + ch_Gr / 2
    RAW_combined = np.dstack((ch_B, ch_G, ch_R))
    RAW_norm = RAW_combined.astype(np.float32) / (3 * 255)

    return RAW_norm


def process_imgs(root_path: str, saving_path: str) -> None:
    os.makedirs(saving_path, exist_ok=True)
    for filename in tqdm.tqdm(os.listdir(root_path)):
        path = os.path.join(root_path, filename)
        raw_image = np.asarray(imageio.imread(path))
        demosaic = extract_bayer_channels(raw_image)
        cv2.imwrite(f'{saving_path}/{filename}', demosaic * 255)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="A function that demosaic n image from the RAW format to the BGR format.")
    parser.add_argument("-s", "--src", type=str, help="data directory of your raw images")
    parser.add_argument("-d", "--dst", type=str, help="save image folder")
    args = parser.parse_args()
    process_imgs(args.src, args.dst)
