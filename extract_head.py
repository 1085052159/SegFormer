import cv2
import numpy as np
import os
from PIL import Image
from glob import glob


def read_img(img_name, gray=False):
    if not os.path.exists(img_name):
        print("not found: ", img_name)
        return None
    img = cv2.imread(img_name)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def write_img(save_name, img):
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    cv2.imwrite(save_name, img)


def gen_binary_mask(mask, del_rgb_color=None):
    h, w, _ = mask.shape
    new_mask = np.ones_like(mask)
    for i in range(h):
        for j in range(w):
            cur_rgb_color = "_".join(str(t) for t in mask[i, j, ::-1])
            # print(cur_rgb_color)
            if cur_rgb_color in del_rgb_color:
                new_mask[i, j] = [0, 0, 0]
    return new_mask


def save_binary_img(img_name, mask_name, del_rgb_color=None):
    img = read_img(img_name)
    mask = read_img(mask_name, gray=False)
    if del_rgb_color is None:
        # delete "background", "clothes", "neck"
        # color [0, 0, 0], [186, 173, 165], [177, 196, 202]
        del_rgb_color = ["0_0_0", "186_173_165", "177_196_202"]
    binary_mask = gen_binary_mask(mask, del_rgb_color)
    
    binary_img = img * binary_mask
    base_dir = os.path.dirname(img_name)
    base_img_name = os.path.basename(img_name)
    save_name = "%s_binary/%s" % (base_dir, base_img_name)
    write_img(save_name, binary_img)


class_names = ['background', 'hair', 'hair_accessories', 'lip', 'clothes',
               'eyebrow', 'upper_eyelid', 'lower_eyelid', 'nostril',
               'face', 'ear', 'pupil', 'highlight',
               'eyes_white', 'iris', 'neck', 'tongue',
               'lip_shadow', 'eye_socket', 'furrows_under_eyes', 'nose',
               'teeth', 'wrinkle', 'limbs', 'blush_sweating']
colors = [[0, 0, 0], [56, 66, 115], [254, 245, 188], [217, 175, 179], [186, 173, 165],
          [44, 97, 77], [80, 45, 52], [164, 149, 152], [212, 203, 188],
          [228, 221, 203], [119, 125, 113], [168, 49, 113], [255, 253, 226],
          [250, 212, 235], [211, 174, 122], [177, 196, 202], [204, 3, 12],
          [208, 197, 212], [56, 148, 228], [153, 204, 0], [128, 128, 128],
          [255, 255, 0], [128, 0, 0], [251, 56, 56], [255, 97, 0]]

# vid_names = [
#     "1d142abd-92aa-11eb-b186-58a02372a267####99",
#     "1d142abe-92aa-11eb-9336-58a02372a267####0",
#     "1d142abe-92aa-11eb-9336-58a02372a267####1",
#     "1d142abe-92aa-11eb-9336-58a02372a267####10",
#     "1d142abe-92aa-11eb-9336-58a02372a267####11",
#     "1d142abe-92aa-11eb-9336-58a02372a267####12",
#     "1d142abe-92aa-11eb-9336-58a02372a267####14",
# ]
# base_path = "/home/ubuntu/Desktop/tmp_videos"
base_path = "/root/autodl-tmp/datasets/cartoon_coarse_filter/img"
vid_names = os.listdir(base_path)
del_rgb_color = ["0_0_0", "186_173_165", "177_196_202"]
for vid_name in vid_names:
    vid_path = "%s/%s" % (base_path, vid_name)
    img_path = "%s/img" % vid_path
    mask_path = "%s/mask" % vid_path
    frames = os.listdir(img_path)
    frames_id = [frame.split(".")[0] for frame in frames]
    frames_id = sorted(frames_id)
    for frame_id in frames_id:
        img_name = "%s/%s.jpg" % (img_path, frame_id)
        mask_name = "%s/%s.png" % (mask_path, frame_id)
        save_binary_img(img_name, mask_name, del_rgb_color)
