config="work_dirs/segformer.b4.512x512.cartoon_25cls.240k/segformer.b4.512x512.cartoon.240k.py"
#config="local_configs/segformer/B4/segformer.b4.512x512.cartoon.240k.py"
ckpt="work_dirs/segformer.b4.512x512.cartoon_25cls.240k/iter_200000.pth"
save_dir="./work_dirs/segformer.b4.512x512.cartoon_25cls.240k/train_masks_pred"
python tools/test.py $config $ckpt --show-dir $save_dir



