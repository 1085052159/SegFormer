input="/home/ubuntu/Desktop/tmp_videos/1d142abd-92aa-11eb-b186-58a02372a267####99/img"
input="/root/autodl-tmp/datasets/cartoon_fine_filter/test/imgs_list.txt"
input="/root/autodl-tmp/codes/LIA/test_videos_by_hezong/sources/source.txt"
save_path="/root/autodl-tmp/codes/LIA/test_videos_by_hezong/sources"
#input="/root/autodl-tmp/codes/LIA/test_videos_by_hezong/drivens/driven13_256_256.mp4"
#save_path="/root/autodl-tmp/codes/LIA/test_videos_by_hezong/drivens/driven13_256_256_head.mp4"
input="/media/ubuntu/win_software/BaiduNetdiskDownload/role_acculate/imgs.txt"
save_path="/media/ubuntu/win_software/BaiduNetdiskDownload/role_acculate"
suffixes="png"
ignore_classes="0 4 15 23"
config="work_dirs/segformer.b4.512x512.cartoon_25cls.240k/segformer.b4.512x512.cartoon.240k.py"
ckpt="work_dirs/segformer.b4.512x512.cartoon_25cls.240k/iter_200000.pth"
export CUDA_VISIBLE_DEVICES=0
python extract_head_video.py --input $input --suffixes $suffixes --ignore_classes $ignore_classes --config $config --checkpoint $ckpt --device cuda:0 --save_path $save_path