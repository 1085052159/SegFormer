from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CartoonDataset(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = (
        'background', 'hair', 'hair_accessories', 'lip', 'clothes',
        'eyebrow', 'upper_eyelid', 'lower_eyelid', 'nostril',
        'face', 'ear', 'pupil', 'highlight',
        'eyes_white', 'iris', 'neck', 'tongue',
        'lip_shadow', 'eye_socket', 'furrows_under_eyes', 'nose',
        'teeth', 'wrinkle', 'limbs', 'blush_sweating')
    
    PALETTE = [[0, 0, 0], [56, 66, 115], [254, 245, 188], [217, 175, 179], [186, 173, 165],
               [44, 97, 77], [80, 45, 52], [164, 149, 152], [212, 203, 188],
               [228, 221, 203], [119, 125, 113], [168, 49, 113], [255, 253, 226],
               [250, 212, 235], [211, 174, 122], [177, 196, 202], [204, 3, 12],
               [208, 197, 212], [56, 148, 228], [153, 204, 0], [128, 128, 128],
               [255, 255, 0], [128, 0, 0], [251, 56, 56], [255, 97, 0]]
    
    def __init__(self, **kwargs):
        super(CartoonDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)
