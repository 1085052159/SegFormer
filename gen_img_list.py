import os
import random
import shutil

import cv2
from glob import glob

from tqdm import tqdm


def sample_img(root_path, sample_num=1):
    # sampled_rel_path = []
    sampled_abso_path = []
    vid_names = os.listdir(root_path)
    for vid in vid_names:
        vid_path = "%s/%s" % (root_path, vid)
        frames = glob("%s/*.*" % (vid_path))
        frame_paths = random.sample(frames, sample_num)
        # frame_names = [os.path.basename(frame_path) for frame_path in frame_paths]
        # rel_frame_paths = ["%s/%s" % (vid, frame_name) for frame_name in frame_names]
        # sampled_rel_path += rel_frame_paths
        sampled_abso_path += frame_paths
    return sampled_abso_path  # , sampled_rel_path


def save_sampled_img(sampled_abso_path, save_path, backup=True):
    for abso_path in tqdm(sampled_abso_path):
        rel_img_path = "/".join(abso_path.split("/")[-2:])
        dst_path = "%s/%s" % (save_path, rel_img_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(abso_path, dst_path)
        if backup:
            dst_backup_path = "%s_backup/%s" % (save_path, rel_img_path)
            os.makedirs(os.path.dirname(dst_backup_path), exist_ok=True)
            shutil.copy(abso_path, dst_backup_path)


def write_list(name, contents):
    with open(name, "w") as f:
        f.write("\n".join(contents))


def coarse_filter():
    root_path = "/root/autodl-tmp/datasets/cartoon_large_frames"
    save_path = "/root/autodl-tmp/datasets/cartoon_coarse_filter/img"
    sampled_abso_path = sample_img(root_path, sample_num=1)
    write_list("%s/imgs_list.txt", sampled_abso_path)
    save_sampled_img(sampled_abso_path, save_path)

    # TODO pred mask
    # TODO mv mask directory to img root path, like: xxx/img, xxx/mask
    # TODO extract binary image


def fine_filter():
    root_path = "/root/autodl-tmp/datasets/cartoon_large_frames"
    save_path = "/root/autodl-tmp/datasets/cartoon_fine_filter/img"
    sampled_abso_path = sample_img(root_path, sample_num=1)
    write_list("%s/imgs_list.txt", sampled_abso_path)
    save_sampled_img(sampled_abso_path, save_path)

    # TODO pred mask
    # TODO mv mask directory to img root path, like: xxx/img, xxx/mask
    # TODO extract binary image
    

coarse_filter()
# fine_filter()