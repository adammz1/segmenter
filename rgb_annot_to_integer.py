import argparse
import os
import tqdm
import mmcv
import numpy as np
from PIL import Image

palette = \
    {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (128, 128, 128),
        5: (255, 255, 0),
        6: (0, 255, 255),
    }

invert_palette = {v: k for k, v in palette.items()}


def convert_from_color(arr_3d, palette=invert_palette):
    """RGB-color encoding to grayscale labels."""
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d


def slide_crop_label(src_path, out_dir):
    label = mmcv.imread(src_path, channel_order='rgb')
    label = convert_from_color(label)
    output_path = os.path.join(out_dir, os.path.basename(src_path).split('.')[0] + '.png')
    label = Image.fromarray(label.astype(np.uint8))
    label.save(output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert iSAID dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='iSAID folder path')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    out_dir = args.out_dir

    src_path_list = [os.path.join(dataset_path, _) for _ in os.listdir(dataset_path)]

    for img_path in tqdm.tqdm(src_path_list):
        slide_crop_label(img_path, out_dir)

    print('Done!')


if __name__ == '__main__':
    main()
