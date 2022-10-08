import os
from argparse import ArgumentParser
from glob import glob

import cv2
import mmcv
from tqdm import tqdm

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import numpy as np


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', help='Image path or tested txt file with absolute img path',
                        default="/root/autodl-tmp/datasets/cartoon_fine_filter/test/imgs_list.txt")
    parser.add_argument("--suffixes", nargs="+", default="jpg png",
                        help="img suffix you want to test, like: jpg png")
    parser.add_argument("--ignore_classes", nargs="+", default="0 4 15 23",
                        help="img suffix you want to test, like: jpg png")
    parser.add_argument('--config', help='Config file',
                        default="work_dirs/segformer.b4.512x512.cartoon_25cls.240k/segformer.b4.512x512.cartoon.240k.py")
    parser.add_argument('--checkpoint', help='Checkpoint file',
                        default="work_dirs/segformer.b4.512x512.cartoon_25cls.240k/iter_200000.pth")
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument("--save_path", default=str)
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    suffixes = args.suffixes
    if isinstance(suffixes, str):
        suffixes = [suffixes]
    inputs = []
    input_file = args.input
    save_path = args.save_path
    if os.path.isfile(input_file):
        suffix = input_file.split(".")[-1]
        if suffix in ["txt"]:
            with open(input_file) as f:
                inputs = f.readlines()
            inputs = [path.strip() for path in inputs]
        if suffix in ["mp4", "avi", "mpeg"]:
            video = mmcv.VideoReader(input_file)
            inputs = video[:]
            vwriter = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), video.fps,
                                      video.resolution)
        if suffix in ["jpg", "png", "jpeg", "bmp"]:
            inputs = [input_file]
    elif os.path.isdir(input_file):
        inputs = glob("%s/*.*" % input_file)
        inputs = [path for path in inputs if path.split(".")[-1] in suffixes]
    else:
        raise ValueError("unrecognized input: %s" % input_file)
    # ignore background, clothes, neck, limbs
    # ignore_classes = [0, 4, 15, 23]
    ignore_classes = [int(x) for x in args.ignore_classes]
    for input_ in tqdm(inputs):
        if isinstance(input_, str):
            img = mmcv.imread(input_)
        else:
            img = input_
        # test a single image
        result = inference_segmentor(model, img)
        mask = result[0].copy()
        for label in range(np.min(mask), np.max(mask) + 1):
            idx = np.where(mask == label)
            if label in ignore_classes:
                mask[idx] = 0
            else:
                mask[idx] = 1
        # mask_rgb = np.zeros((*mask.shape, 3), dtype=mask.dtype)
        # import pdb
        # pdb.set_trace()
        mask = mask.astype(img.dtype)
        head_img = img.copy()
        for ch in range(img.shape[-1]):
            head_img[:, :, ch] = img[:, :, ch] * mask
        if save_path.split(".")[-1] in ["mp4", "avi", "mpeg"]:
            vwriter.write(head_img)
        else:
            # save_path_ = "%s/%s_head.jpg" % (save_path, os.path.basename(input_).split(".")[0])
            save_path_ = input_.replace("/imgs/", "/imgs_head/")
            mmcv.imwrite(head_img, save_path_)
    if save_path.split(".")[-1] in ["mp4", "avi", "mpeg"]:
        vwriter.release()


if __name__ == '__main__':
    main()
