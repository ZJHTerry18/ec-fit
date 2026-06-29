import numpy as np

IMAGE_SIZE = 640

# path to mano files
MANO_MODEL_PATH = 'static/MANO/mano_v1_2/models'

# mesh colors
COLOR_HUMAN_BLUE = [67, 135, 240, 255]
COLOR_OBJECT_RED = [255, 69, 0, 255]

# OSX camera setup
OSX_FOCAL_VIRTUAL = (5000, 5000)
OSX_INPUT_BODY_SHAPE = (256, 192)
OSX_PRINCPT = (OSX_INPUT_BODY_SHAPE[1] / 2, OSX_INPUT_BODY_SHAPE[0] / 2)

# boilerplate
SMPLX_LAYER_ARGS = {
    'create_global_orient': False,
    'create_body_pose': False,
    'create_left_hand_pose': False,
    'create_right_hand_pose': False,
    'create_jaw_pose': False,
    'create_leye_pose': False,
    'create_reye_pose': False,
    'create_betas': False,
    'create_expression': False,
    'create_transl': False
}

# Seal mesh faces for MANO (make it watertight)
SEAL_MANO_FACES = {
    "left": [
        (79, 108, 215),
        (79, 214, 78),
        (92, 234, 122),
        (108, 120, 119),
        (117, 118, 279),
        (118, 122, 239),
        (119, 117, 279),
        (122, 38, 92),
        (214, 121, 78),
        (215, 108, 119),
        (215, 119, 279),
        (215, 214, 79),
        (234, 239, 122),
        (239, 279, 118)
    ],
    "right": [
        (79, 78, 214),
        (92, 38, 122),
        (108, 79, 215),
        (117, 119, 279),
        (118, 117, 279),
        (118, 239, 122),
        (119, 120, 108),
        (121, 214, 78),
        (214, 215, 79),
        (215, 119, 108),
        (215, 279, 119),
        (234, 92, 122),
        (239, 234, 122),
        (279, 239, 118)
    ]
}

# object pose initialization strategies
RADNOM_MULTIPLE = [
    [-np.pi / 2, 0, 0],
    [np.pi / 2, 0, 0],
    [0, 0, 0],
    [np.pi, 0, 0],
]
RANDOM_PRIOR = 'static/object_pose_priors'

# output exp template
OUT_TEMPLATE = "con-p2-{}_con-p3-{}_silo{}-occ_peno{}_sc{}_silh{}_penh{}_reg{}{}"