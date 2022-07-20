device_id=1
config="local_configs/segformer/B4/segformer.b4.512x512.cartoon.160k.py"
#CUDA_VISIBLE_DEVICES=$device_id python tools/train.py $config
CUDA_VISIBLE_DEVICES=$device_id python tools/train.py local_configs/segformer/B4/segformer.b4.512x512.cartoon.160k.py
#CUDA_VISIBLE_DEVICES=2 python tools/train.py local_configs/segformer/B4/segformer.b4.512x512.cartoon.160k.py