import os
import os.path as osp
import numpy as np
from glob import glob
import copy
import torch
from torch.utils.data import Dataset

from src.constants import RADNOM_MULTIPLE, RANDOM_PRIOR
from src.utils.load_utils import load_hand_params, load_object_params
from src.utils.contact_mapping import load_contact_mapping, load_sparse_dense_mapping
from src.utils.geometry import axis_angle_to_matrix, rotation_matrix_to_angle_axis


class EpicDataset(Dataset):
    def __init__(self, data_dir : str, start_idx: int=0, end_idx: int=10**9, cfg=None):
        self.data_dir = data_dir
        self.cfg = cfg
        self.dataset_samples = []

        # Prepare data
        self.prepare_data_list(start_idx, end_idx)
    
    def prepare_data_list(self, start_idx, end_idx):
        for folder in sorted(os.listdir(self.data_dir)):
            folder_path = osp.join(self.data_dir, folder)
            if not osp.isdir(folder_path):
                continue
            contact_ann_file = osp.join(folder_path, "corresponding_contacts.json")
            if not osp.exists(contact_ann_file):
                continue
            self.dataset_samples.append(folder)
        
        self.dataset_samples = self.dataset_samples[start_idx:end_idx]
    
    def __len__(self,):
        return len(self.dataset_samples)

    def _check_file(self, file, isdir=False):
        return osp.exists(file)
    
    def __getitem__(self, index):
        folder_name = self.dataset_samples[index]
        folder_path = osp.join(self.data_dir, folder_name)
        obj_cat = folder_name[:-4].split("_")[5]

        # load the hand, object and contact mapping
        lr_flag = "left" if "left" in folder_name else "right"
        hand_mesh_path = osp.join(folder_path, f"{lr_flag}_hand_posed_mesh.ply")
        obj_mesh_path = osp.join(folder_path, "object.obj")
        contact_path = osp.join(folder_path, "corresponding_contacts.json")
        hand_npz_path = osp.join(folder_path, "wilor_output.pkl")
        hand_mask_path = osp.join(folder_path, "hand_mask.png")
        obj_mask_path = osp.join(folder_path, "object_mask.png")
        try:
            assert self._check_file(hand_mesh_path)
            assert self._check_file(obj_mesh_path), f"{obj_mesh_path} does not exist"
            assert self._check_file(contact_path)
            assert self._check_file(hand_npz_path), f"{hand_npz_path} does not exist"
        except Exception as e:
            print(f"{folder_name} has missing files: {e}. Skip.")
            return []

        # get camera intrinsic matrix
        cam_intrinsic = None
        hand_npz = None
        render_img_size = [456, 256]
        hand_npz = np.load(hand_npz_path, allow_pickle=True)[lr_flag]
        fl = hand_npz['focal_length'].item()
        cx, cy = hand_npz['img_size'][0], hand_npz['img_size'][1]
        render_img_size = [int(cy * 2), int(cx * 2)]
        cam_intrinsic = torch.FloatTensor([[fl, 0, cx], [0, fl, cy], [0, 0, 1]])

        # get objectmeta
        meta_info = {
            "cat": obj_cat
        }

        # load hand parameters, object parameters, contact
        load_hand_mask = (not (self.cfg.skip_phase_3 and self.cfg.skip_phase_2)) and (self._check_file(hand_mask_path))
        load_obj_mask = (not self.cfg.skip_phase_2) and (self._check_file(obj_mask_path))
        load_occ_mask = load_obj_mask
        load_mano = (not self.cfg.skip_phase_3)
        hand_params = load_hand_params(
            hand_mesh_path, hand_detection_file=hand_mask_path, imgsize=render_img_size, 
            lr_flag=lr_flag, center=True, load_hand_mask=load_hand_mask, load_mano=load_mano,
            cam_intrinsic=cam_intrinsic, hand_npz=hand_npz,
        )
        contact_mapping = load_contact_mapping(contact_path, convert_to_smplx=False)
        sparse_dense_mapping = load_sparse_dense_mapping("./sparse_dense_mapping.json")

        if self.cfg.object_pose_init == "single":
            object_params, _ = load_object_params(
                obj_mesh_path, object_detection_file=obj_mask_path, imgsize=render_img_size, 
                trans_mat=None, load_obj_mask=load_obj_mask, load_occ_mask=load_occ_mask, cam_intrinsic=cam_intrinsic
            )

            sample = dict()
            sample["hand_params"] = hand_params
            sample["object_params"] = object_params
            sample["contact_mapping"] = contact_mapping
            sample["sparse_dense_mapping"] = sparse_dense_mapping
            sample["render_size"] = render_img_size
            sample["cam_intrinsic"] = cam_intrinsic
            sample["meta_info"] = meta_info
            sample["metrics"] = dict()

            return sample, folder_name
        elif self.cfg.object_pose_init.startswith("multi"):
            if self.cfg.object_pose_init == "multi-random":
                init_poses = RADNOM_MULTIPLE
            elif self.cfg.object_pose_init in ["multi-prior", "multi-mixed"]:
                init_pose_dir = RANDOM_PRIOR
                obj_pose_files = sorted(glob(osp.join(init_pose_dir, obj_cat, "*.npy")))
                obj_pose_files = [x for x in obj_pose_files if lr_flag in x]
                obj_pose_mats = [torch.from_numpy(np.load(file))  for file in obj_pose_files]
                init_poses = []
                    
                # hand coordinate -> camera coordinate with hand centered: T_h2c
                T_h2c = torch.eye(4).to(torch.float64)
                global_orient = rotation_matrix_to_angle_axis(torch.from_numpy(hand_npz['rot'])[None, :]).squeeze()
                if lr_flag == "left":
                    global_orient[..., 1:] *= -1
                T_h2c[:3, :3] = axis_angle_to_matrix(global_orient)
                T_h2c[:3, 3] = torch.from_numpy(hand_npz["pred_cam_t"]) - hand_params.centroid_offset.cpu()
                for T_o2h in obj_pose_mats:
                    # object coordinate -> hand coordinate: T_o2h
                    T_o2c = torch.matmul(T_h2c, T_o2h)
                    init_pose = rotation_matrix_to_angle_axis(T_o2c[None, :3, :3]).squeeze().tolist()
                    init_poses.append(init_pose)

                if self.cfg.object_pose_init == "multi-mixed":
                    init_poses.extend(RADNOM_MULTIPLE)
            else:
                raise NotImplementedError(self.cfg.object_pose_init)
            samples = list()
            for inipose in init_poses:
                object_params, _ = load_object_params(
                    obj_mesh_path, object_detection_file=obj_mask_path, imgsize=render_img_size, 
                    trans_mat=inipose, load_obj_mask=load_obj_mask, load_occ_mask=load_occ_mask, cam_intrinsic=cam_intrinsic
                )

                sample = dict()
                sample["hand_params"] = copy.deepcopy(hand_params)
                sample["object_params"] = object_params
                sample["contact_mapping"] = contact_mapping
                sample["sparse_dense_mapping"] = sparse_dense_mapping
                sample["render_size"] = render_img_size
                sample["cam_intrinsic"] = cam_intrinsic
                sample["meta_info"] = meta_info
                sample["metrics"] = dict()
                samples.append((sample, folder_name))
            
            return samples
