import os
from argparse import ArgumentParser
from glob import glob

from tqdm import tqdm

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


def main():
    parser = ArgumentParser()
    parser.add_argument('img_path', help='Image path or tested txt file with absolute img path')
    parser.add_argument("img_suffixes", nargs="+",
                        help="img suffix you want to test, like: jpg png")
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    suffixes = args.img_suffixes
    if isinstance(suffixes, str):
        suffixes = [suffixes]
    img_paths = []
    # print(suffixes)
    img_path = args.img_path
    if os.path.isfile(img_path):
        with open(img_path) as f:
            img_paths = f.readlines()
        img_paths = [path.strip() for path in img_paths]
    if os.path.isdir(img_path):
        img_paths = glob("%s/*.*" % img_path)
    img_paths = [path for path in img_paths if path.split(".")[-1] in suffixes]
    for img_path in tqdm(img_paths):
        # test a single image
        result = inference_segmentor(model, img_path)
        # show the results
        show_result_pyplot(model, img_path, result, get_palette(args.palette))
    


if __name__ == '__main__':
    main()
