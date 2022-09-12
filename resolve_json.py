import json
import base64
import os
import shutil
from glob import glob

from tqdm import tqdm


def gen_diff_list(base_path):
    names = glob("%s/jsons/*.json" % (base_path))
    names = [name.replace("\\", "/") for name in names]
    normal = []
    abnormal = []
    repair = []
    for json_name in tqdm(names):
        with open(json_name) as f:
            content = json.load(f)
        label = content["shapes"][0]["label"]
        img_name = content["imagePath"].replace("!!", "/")
        if label == "chin":
            normal.append(img_name)
        elif label == "eyebrow":
            abnormal.append(img_name)
        elif label == "circle":
            repair.append(img_name)
    print("normal count: %s; abnormal count: %s; circle: %s" % (len(normal), len(abnormal), len(repair)))
    return normal, abnormal, repair


def write_list(filename, contents):
    with open(filename, "w") as f:
        f.write("\n".join(contents))


def copy_file(src_abso_file, dst_abso_file):
    if not os.path.exists(src_abso_file):
        print("not found: ", src_abso_file)
    os.makedirs(os.path.dirname(dst_abso_file), exist_ok=True)
    shutil.copy(src_abso_file, dst_abso_file)


def copy_batch_files(src_path_prefix, src_rel_files, dst_path_prefix):
    for rel_file in tqdm(src_rel_files):
        src_abso_path = "%s/%s" % (src_path_prefix, rel_file)
        dst_abso_path = "%s/%s" % (dst_path_prefix, rel_file)
        copy_file(src_abso_path, dst_abso_path)


base_path = "D:\BaiduNetdiskDownload\分割质量标注chin正常-eyebrow异常-circle需要人工修复如字幕等"
normal, abnormal, repair = gen_diff_list(base_path)
write_list("%s/normal.txt" % base_path, normal)
write_list("%s/abnormal.txt" % base_path, abnormal)
write_list("%s/repair.txt" % base_path, repair)
