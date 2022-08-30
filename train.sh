config="local_configs/segformer/B4/segformer.b4.512x512.LovaszLoss.cartoon.240k.py"
work_dir="work_dirs/segformer.b4.512x512.LovaszLoss.cartoon_25cls.240k"
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --master_port=9998 tools/train.py $config --launcher pytorch --work-dir $work_dir

#CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=9998 tools/train.py local_configs/segformer/B4/segformer.b4.512x512.cartoon.160k.py --launcher pytorch --work-dir work_dirs/segformer.b4.512x512.cartoon_25cls.160k

#CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --master_port=9998 tools/train.py local_configs/segformer/B4/segformer.b4.512x512.cartoon.160k.py --launcher pytorch --work-dir work_dirs/segformer.b4.512x512.cartoon_25cls.160k