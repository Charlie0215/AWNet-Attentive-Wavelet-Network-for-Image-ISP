# -*- coding: utf-8 -*-
import os

import cv2

if __name__ == "__main__":
    path1 = './result_fullres_4channel/'
    path2 = './result_fullres_3channel/'
    save_path = './final_result_fullres'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in os.listdir(path1):
        image_path1 = os.path.join(path1, file)
        image_path2 = os.path.join(path2, file)
        img1 = cv2.imread(image_path1)
        img2 = cv2.imread(image_path2)
        out = img1 / 2 + img2 / 2
        save_path1 = os.path.join(save_path, file)
        cv2.imwrite(save_path1, out)
