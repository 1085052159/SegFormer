img_path="/home/ubuntu/Desktop/tmp_videos/1d142abd-92aa-11eb-b186-58a02372a267####99/img"
img_path="/home/ubuntu/Desktop/tmp_videos/1d142abe-92aa-11eb-9336-58a02372a267####0/img"
img_path="/home/ubuntu/Desktop/tmp_videos/1d142abe-92aa-11eb-9336-58a02372a267####1/img"
img_path="/home/ubuntu/Desktop/tmp_videos/1d142abe-92aa-11eb-9336-58a02372a267####10/img"
img_path="/home/ubuntu/Desktop/tmp_videos/1d142abe-92aa-11eb-9336-58a02372a267####11/img"
img_path="/home/ubuntu/Desktop/tmp_videos/1d142abe-92aa-11eb-9336-58a02372a267####12/img"
img_path="/home/ubuntu/Desktop/tmp_videos/1d142abe-92aa-11eb-9336-58a02372a267####14/img"
img_path="/root/autodl-tmp/datasets/cartoon_coarse_filter/train/img/imgs_list_01.txt"
img_suffixes="jpg"
config="work_dirs/segformer.b4.512x512.cartoon_25cls.240k/segformer.b4.512x512.cartoon.240k.py"
ckpt="work_dirs/segformer.b4.512x512.cartoon_25cls.240k/iter_200000.pth"
export CUDA_VISIBLE_DEVICES=0
python demo/batch_image_demo.py $img_path $img_suffixes $config $ckpt --device cuda:0 --palette cartoon