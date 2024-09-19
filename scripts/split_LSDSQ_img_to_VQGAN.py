# File: parse_ICDAR2013_img_to_VQGAN.py
# Created by Juan A. Rodriguez on 12/06/2022
# Goal: This script is intended to access the json files corresponding to the ICDAR13 dataset (train, val)
# and convert them to the format required by the VQGAN, that is, a txt file containing the image path,

import os
import argparse
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, default="LSDSQ_flattened_240_gray", help="Path to dataset root, containing image directories (train and test)")
args = parser.parse_args()

if __name__ == '__main__':
    count = 0
    path = args.path
    filenames = [os.path.join(path,fle) for fle in os.listdir(path) if fle.endswith(".png")]
    splits = {'train':filenames[:int(len(filenames)*0.9)], 'test':filenames[int(len(filenames)*0.9):]}
    for split_name, split in splits.items():
        for filename in split:
            # append to txt file
            with open(os.path.join(path, 'LSDSQ_flattened_240_gray_img_'+split_name+'.txt'), 'a') as f:
                f.write(filename + '\n')
            count += 1
    print(f"Stored {count} images in LSDSQ_flattened_240_gray_img_{split_name}.txt")
