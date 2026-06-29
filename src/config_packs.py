default_loss_weights = {
    'lw_contact_p2': 8.0,
    'lw_contact_p3': 8.0,
    'lw_silhouette': 0.3,
    'lw_silhouette_distance': 0.3,
    'lw_scale': 4,
    'lw_collision_p2': 100,
    'lw_collision_p3': 50,
    'lw_pose_reg': 0.05,
    'lw_silhouette_hand': 0.1,
}

epic_loss_weights = {
    'lw_contact_p2': 8.0,
    'lw_contact_p3': 80.0,
    'lw_silhouette': 0.03,
    'lw_silhouette_distance': 0.03,
    'lw_scale': 1.0,
    'lw_collision_p2': 0.01,
    'lw_collision_p3': 1.0,
    'lw_pose_reg': 0.5,
    'lw_silhouette_hand': 0.03,
}

class ConfigPack:
    def __init__(
        self,
        # input file names
        hand_inference_file: str = 'hand_posed_mesh.ply',
        # hand_detection_file: str = 'hand_detection.npz',
        object_mesh_file: str = 'object.obj',
        # object_detection_file: str = 'object_detection.npz',
        contact_mapping_file: str = 'corresponding_contacts.json',
        # support file names
        sparse_dense_mapping_file: str = 'sparse_dense_mapping.json',
        object_pose_init: str = 'single',
        # optimization nr of steps
        nr_phase_1_steps: int = 250,
        lr_rotation_phase_1: float = 0.04,
        lr_translation_phase_1: float = 0.02,
        skip_phase_1: bool = False,
        nr_phase_2_steps: int = 1500,
        lr_rotation_phase_2: float = 0.04,
        lr_translation_phase_2: float = 0.03,
        lr_scaling_phase_2: float = 0.02,
        skip_phase_2: bool = False,
        nr_phase_3_steps: int = 1000,
        lr_phase_3: float=0.01,
        phase_3_upd_trans: bool = False,
        phase_3_upd_rot: bool = False,
        phase_3_dyn_w: bool = False,
        skip_phase_3: bool = False,
        loss_weights: dict = default_loss_weights,
    ):
        self.hand_inference_file = hand_inference_file
        # self.hand_detection_file = hand_detection_file
        self.object_mesh_file = object_mesh_file
        # self.object_detection_file = object_detection_file
        self.contact_mapping_file = contact_mapping_file
        self.sparse_dense_mapping_file = sparse_dense_mapping_file
        self.object_pose_init = object_pose_init

        self.nr_phase_1_steps = nr_phase_1_steps
        self.lr_rotation_phase_1 = lr_rotation_phase_1
        self.lr_translation_phase_1 = lr_translation_phase_1
        self.skip_phase_1 = skip_phase_1
        self.nr_phase_2_steps = nr_phase_2_steps
        self.lr_rotation_phase_2 = lr_rotation_phase_2
        self.lr_translation_phase_2 = lr_translation_phase_2
        self.lr_scaling_phase_2 = lr_scaling_phase_2
        self.skip_phase_2 = skip_phase_2
        self.nr_phase_3_steps = nr_phase_3_steps
        self.lr_phase_3 = lr_phase_3
        self.phase_3_upd_trans = phase_3_upd_trans
        self.phase_3_upd_rot = phase_3_upd_rot
        self.phase_3_dyn_w = phase_3_dyn_w
        self.skip_phase_3 = skip_phase_3

        self.loss_weights = loss_weights

default_config = ConfigPack()

epic_config = ConfigPack(
    object_pose_init="multi-mixed", # choices: [single, multi-random, multi_prior, multi-mixed]
    nr_phase_1_steps=250,
    lr_rotation_phase_1=0.04,
    lr_translation_phase_1=0.02,
    skip_phase_1=False,
    nr_phase_2_steps=500,
    lr_rotation_phase_2=0.01,
    lr_translation_phase_2=0.005,
    lr_scaling_phase_2=0.005,
    skip_phase_2=False,
    nr_phase_3_steps=500,
    lr_phase_3=0.01,
    phase_3_upd_trans=True,
    phase_3_dyn_w=False,
    skip_phase_3=False,
    loss_weights=epic_loss_weights,
)
