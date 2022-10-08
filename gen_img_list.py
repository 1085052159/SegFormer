import os
import random
import shutil

import cv2
from glob import glob

from tqdm import tqdm

random.seed(0)


def sample_img(root_path, sample_num=1):
    print("sampled img start")
    sampled_rel_path = []
    sampled_abso_path = []
    sub_vid_dirs = os.listdir(root_path)
    for sub_vid_dir in sub_vid_dirs:
        sub_vid_path = "%s/%s" % (root_path, sub_vid_dir)
        if os.path.isfile(sub_vid_path):
            continue
        vid_names = os.listdir(sub_vid_path)
        for vid in vid_names:
            vid_path = "%s/%s" % (sub_vid_path, vid)
            if os.path.isfile(vid_path):
                continue
            frames = glob("%s/*.*" % (vid_path))
            sample_num = min(sample_num, len(frames))
            if sample_num == 0:
                print("no frames in %s" % vid_path)
            frame_paths = random.sample(frames, sample_num)
            frame_names = [os.path.basename(frame_path) for frame_path in frame_paths]
            rel_frame_paths = ["%s/%s/%s" % (sub_vid_dir, vid, frame_name) for frame_name in frame_names]
            sampled_rel_path += rel_frame_paths
            sampled_abso_path += frame_paths
    print("sampled img end")
    # print(len(sampled_abso_path), len(sampled_rel_path))
    return sampled_abso_path, sampled_rel_path


def save_sampled_img(sampled_abso_path, save_path, sampled_rel_path, backup=True):
    assert len(sampled_rel_path) == len(sampled_abso_path)
    for i in tqdm(range(len(sampled_rel_path))):
        abso_path = sampled_abso_path[i]
        rel_path = sampled_rel_path[i]
        dst_path = "%s/%s" % (save_path, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(abso_path, dst_path)
        if backup:
            dst_backup_path = "%s_backup/%s" % (save_path, rel_path)
            os.makedirs(os.path.dirname(dst_backup_path), exist_ok=True)
            shutil.copy(abso_path, dst_backup_path)


def write_list(name, contents):
    with open(name, "w") as f:
        f.write("\n".join(contents))


def coarse_filter():
    root_path = "/root/autodl-tmp/datasets/cartoon_large_frames/train"
    save_path = "/root/autodl-tmp/datasets/cartoon_coarse_filter/train/img"
    sampled_abso_path, sampled_rel_path = sample_img(root_path, sample_num=1)
    save_sampled_img(sampled_abso_path, save_path, sampled_rel_path, backup=True)
    saved_abso_path = ["%s/%s" % (save_path, rel_path) for rel_path in sampled_rel_path]
    write_list("%s/imgs_list.txt" % ("/".join(save_path.split("/")[: -1])), saved_abso_path)
    
    root_path = "/root/autodl-tmp/datasets/cartoon_large_frames/test"
    save_path = "/root/autodl-tmp/datasets/cartoon_coarse_filter/test/img"
    sampled_abso_path, sampled_rel_path = sample_img(root_path, sample_num=1)
    save_sampled_img(sampled_abso_path, save_path, sampled_rel_path, backup=True)
    saved_abso_path = ["%s/%s" % (save_path, rel_path) for rel_path in sampled_rel_path]
    write_list("%s/imgs_list.txt" % ("/".join(save_path.split("/")[: -1])), saved_abso_path)
    
    # TODO pred mask
    # TODO mv mask directory to img root path, like: xxx/img, xxx/mask
    # TODO extract binary image


def fine_filter():
    root_path = "/root/autodl-tmp/datasets/cartoon_large_frames/train"
    save_path = "/root/autodl-tmp/datasets/cartoon_coarse_filter/train/img"
    sampled_abso_path, sampled_rel_path = sample_img(root_path, sample_num=3)
    save_sampled_img(sampled_abso_path, save_path, sampled_rel_path, backup=True)
    saved_abso_path = ["%s/%s" % (save_path, rel_path) for rel_path in sampled_rel_path]
    write_list("%s/imgs_list.txt" % ("/".join(save_path.split("/")[: -1])), saved_abso_path)
    
    root_path = "/root/autodl-tmp/datasets/cartoon_large_frames/test"
    save_path = "/root/autodl-tmp/datasets/cartoon_coarse_filter/test/img"
    sampled_abso_path, sampled_rel_path = sample_img(root_path, sample_num=1)
    save_sampled_img(sampled_abso_path, save_path, sampled_rel_path, backup=True)
    saved_abso_path = ["%s/%s" % (save_path, rel_path) for rel_path in sampled_rel_path]
    write_list("%s/imgs_list.txt" % (save_path.split("/")[: -1]), saved_abso_path)
    
    # TODO pred mask
    # TODO mv mask directory to img root path, like: xxx/img, xxx/mask
    # TODO extract binary image


def delete_mask(root_path):
    modes = os.listdir(root_path)
    for mode in modes:
        mode_root = "%s/%s" % (root_path, mode)
        if os.path.isfile(mode_root):
            continue
        sub_vid_paths = os.listdir(mode_root)
        for sub_vid in sub_vid_paths:
            sub_vid_root = "%s/%s" % (mode_root, sub_vid)
            if os.path.isfile(sub_vid_root):
                continue
            vids = os.listdir(sub_vid_root)
            for vid in vids:
                vid_root = "%s/%s" % (sub_vid_root, vid)
                if os.path.isfile(vid_root):
                    continue
                frames = os.listdir(vid_root)
                for frame in frames:
                    if frame == "mask":
                        mask_root = "%s/mask" % vid_root
                        print(mask_root)
                        # shutil.rmtree(mask_root)


coarse_filter()
# fine_filter()
# delete_mask("/root/autodl-tmp/datasets/cartoon_large_frames")
