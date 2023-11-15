model = dict(
    type='SAMDetector',
    #sam_type='vit_h',
    #sam_weights_path='weights/sam_vit_h_4b8939.pth',
    sam_type='vit_b',
    sam_weights_path='weights/sam_vit_b_01ec64.pth',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        bgr_to_rgb=True,
        pad_size_divisor=1,
    ),
)
