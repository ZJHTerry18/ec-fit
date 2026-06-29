import cv2
import numpy as np
import trimesh
import os
import torch
import trimesh.transformations as tr
from scipy.spatial.transform import Rotation as R
from typing import Tuple

from src.constants import IMAGE_SIZE
from src.utils.structs import HandParams, ObjectParams
from src.utils.renderer_out import MySoftSilhouetteRenderer    

def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # resize image
    h, w = image.shape[:2]
    r = min(IMAGE_SIZE / w, IMAGE_SIZE / h)
    w = int(r * w)
    h = int(r * h)
    image = cv2.resize(image, (w, h))

    return image


def load_hand_params(hand_inference_file: str, lr_flag: str, hand_detection_file: str = None, imgsize: np.ndarray = None, 
                     center: bool=False, load_hand_mask: bool=False, load_mano: bool=False,
                     cam_intrinsic=None, hand_npz=None) -> HandParams:
    hand_mesh = trimesh.load(hand_inference_file)

    # load hand mask
    if load_hand_mask:
        if hand_detection_file:
            mask = cv2.imread(hand_detection_file) / 255.0
            mask = cv2.resize(mask, (int(hand_npz["img_size"][0]), int(hand_npz["img_size"][1])))
            # crop hand mask with the wilor bbox, to exclude arms
            hcx, hcy = hand_npz["box_center"]
            hl = hand_npz["box_size"] * 0.5
            x_min, y_min = max(0, int(hcx - hl * 0.5)), max(0, int(hcy - hl * 0.5))
            x_max, y_max = min(mask.shape[1], int(hcx + hl * 0.5)), min(mask.shape[0], int(hcy + hl * 0.5))
            new_mask = np.zeros_like(mask)
            new_mask[y_min:y_max, x_min:x_max, :] = mask[y_min:y_max, x_min:x_max, :]
            mask = cv2.resize(new_mask, (imgsize[1], imgsize[0]))[:, :, 0]
        else:
            faces = torch.from_numpy(hand_mesh.faces).float().cuda()
            vertices = torch.from_numpy(hand_mesh.vertices).float().cuda()
            renderer = MySoftSilhouetteRenderer(img_shape=imgsize, faces=faces, cam_intrinsic=cam_intrinsic)
            mask = renderer.render(vertices).cpu().numpy()
    else:
        mask = None
    
    if center:
        centroid_offset = hand_mesh.centroid
        hand_mesh.apply_translation(-centroid_offset)
    else:
        centroid_offset = np.array([0., 0., 0.])

    # Load MANO hand parameters
    mano_params = None
    if load_mano:
        global_orient = R.from_matrix(hand_npz['rot']).as_rotvec()
        if lr_flag == "left":
            global_orient[..., 1:] *= -1
        mano_params = {
            'betas': hand_npz['betas'],
            'hand_pose': hand_npz['hand_params_original'],
            'global_orient': global_orient,
            'transl': hand_npz['pred_cam_t'],
            'is_rhand': (lr_flag == 'right'),
        }

    hand_params = HandParams(
        vertices = hand_mesh.vertices,
        faces = hand_mesh.faces,
        centroid_offset = centroid_offset.copy(),
        left_right=lr_flag,
        # bbox = hand_npz['bbox'][0],
        mask = mask,
        mano_params = mano_params
    )

    hand_params.to_cuda()
    return hand_params
    

def load_object_params(object_mesh_file: str, object_detection_file: str = None, imgsize: np.ndarray = None,
                       trans_mat=None, load_obj_mask=False, load_occ_mask=False, cam_intrinsic=None) -> Tuple[ObjectParams, torch.Tensor]:
    obj_mesh = trimesh.load(object_mesh_file)
    gt_vertices = torch.from_numpy(obj_mesh.vertices).float()

    # render object mask
    mask = None
    occ_mask = None
    if load_obj_mask:
        if object_detection_file:
            mask = cv2.imread(object_detection_file) / 255.0
            mask = cv2.resize(mask, (imgsize[1], imgsize[0]))[:, :, 0]
            if load_occ_mask:
                occ_mask_file = object_detection_file.replace('object_mask.png', 'occ_obj_mask.png')
                if os.path.exists(occ_mask_file):
                    occ_mask = cv2.imread(occ_mask_file) / 255.0
                    occ_mask = cv2.resize(occ_mask, (imgsize[1], imgsize[0]))[: ,:, 0]
        else:
            faces = torch.from_numpy(obj_mesh.faces).float().cuda()
            vertices = torch.from_numpy(obj_mesh.vertices).float().cuda()
            renderer = MySoftSilhouetteRenderer(img_shape=imgsize, faces=faces, cam_intrinsic=cam_intrinsic)
            mask = renderer.render(vertices).cpu().numpy()

    # initialize object pose, according to how the HOI dataset defines it
    if trans_mat is None:
        # rotate object 90 degrees around x-axis (mostly upright in objaverse)
        rotation_offset = np.array([-np.pi / 2, 0, 0])
        centroid_offset = -obj_mesh.centroid
        obj_mesh.apply_transform(tr.rotation_matrix(-np.pi/2, [1, 0, 0]))
        # center the mesh
        obj_mesh.apply_translation(centroid_offset)
    elif isinstance(trans_mat, list): # if list, it is the multiple inits of EPIC
        angle = np.linalg.norm(trans_mat)
        axis = trans_mat / angle if angle != 0 else np.array([1.0, 0.0, 0.0])
        rotation_offset = np.array(trans_mat)
        centroid_offset = -obj_mesh.centroid
        obj_mesh.apply_transform(tr.rotation_matrix(angle, axis))
        obj_mesh.apply_translation(centroid_offset)
    else:
        rotation_offset = np.array([0., 0., 0.])
        centroid_offset = np.array([0., 0., 0.])
        rot_mat = torch.eye(4, dtype=torch.float)
        rot_mat[:3, :3] = trans_mat[:3, :3]
        trans_vec = trans_mat[:3, 3]
        obj_mesh.apply_translation(trans_vec)
        obj_mesh.apply_transform(rot_mat)

    object_params = ObjectParams(
        vertices = obj_mesh.vertices,
        faces = obj_mesh.faces,
        rotation_offset = rotation_offset.copy(), # rotation offset of the initial pose from the canonical pose
        centroid_offset = centroid_offset.copy(), # centroid offset of the initial pose from the canonical pose
        mask = mask,
        occ_mask = occ_mask,
        scale = obj_mesh.extents.max()
    )

    object_params.to_cuda()
    return object_params, gt_vertices

