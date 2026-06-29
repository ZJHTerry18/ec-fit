import torch
import trimesh
import os
import os.path as osp
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict

MESH_DIR = "/home/b5db/jiahezhao25.b5db/jiahe/data/epic-grasps/pico_v3/pico_stage3_fullvideos/2026-02-19_pico_stage3_allvideos_wilorspace_maskv3_newsf_multimixed_con-p2-8.0_con-p3-80.0_silo0.03-occ_peno0.01_sc1.0_silh0.03_penh1.0_reg0.5_upd-h-trans_selbest"
PHASE_SELECT_FILE = "/home/b5db/jiahezhao25.b5db/jiahe/data/epic-grasps/pico_v3/pico_stage3_fullvideos/majority_vote_labels.json"
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_epic_mesh(root_dir, phase_select=None):
    '''
    Load EPIC-Contact mesh
    '''
    epic_mesh_list = []
    for video in sorted(os.listdir(root_dir)):
        fpath = osp.join(root_dir, video)
        hoi_mesh_dict = dict()
        if phase_select is not None:
            sel = phase_select[video]
            if sel == "None" or sel == "NA":
                sel = "Phase 3"
            phase = int(sel.split()[-1].strip())
            hand_mesh_path = osp.join(fpath, f"pred_hand_mesh_phase{phase}.obj")
            object_mesh_path = osp.join(fpath, f"pred_obj_mesh_phase{phase}.obj")
            if not osp.exists(hand_mesh_path) or not osp.exists(object_mesh_path):
                continue
            hoi_mesh_dict[f"phase_sel"] = (hand_mesh_path, object_mesh_path)
        else:
            for phase in range(1, 4):
                hand_mesh_path = osp.join(fpath, f"pred_hand_mesh_phase{phase}.obj")
                object_mesh_path = osp.join(fpath, f"pred_obj_mesh_phase{phase}.obj")
                if not osp.exists(hand_mesh_path) or not osp.exists(object_mesh_path):
                    continue
                hoi_mesh_dict[f"phase{phase}"] = (hand_mesh_path, object_mesh_path)
        epic_mesh_list.append(hoi_mesh_dict)
    
    return epic_mesh_list

def run_hoi_pen(hand_mesh: trimesh.Trimesh, object_mesh: trimesh.Trimesh):
    '''
    Calculate the hand-object penetration for one example.
    hand_mesh, object_mesh: trimesh.Trimesh
    '''
    hand_verts = hand_mesh.vertices # [Vh, 3]
    hand_faces = hand_mesh.faces # [Fh, 3]
    obj_verts = object_mesh.vertices
    obj_faces = object_mesh.faces

    # get penetration mask
    obj_triangles = obj_verts[obj_faces] # [Fh, 3, 3]
    hand_verts_tensor = torch.from_numpy(hand_verts).float().to(device)
    obj_triangles_tensor = torch.from_numpy(obj_triangles).float().to(device)
    exterior = batch_mesh_contains_points(hand_verts_tensor[None, ...], obj_triangles_tensor[None, ...],
                                          torch.Tensor([0.4395064455, 0.617598629942, 0.652231566745]).to(device))
    penetr_mask = ~exterior.squeeze(dim=0).detach().cpu().numpy()

    # get penetration depth and volume
    if penetr_mask.sum() == 0:
        max_depth = 0.0
        volume = 0.0
    else:
        sample_info = {
            "sbj_verts": hand_verts,
            "obj_verts": obj_verts,
            "sbj_faces": hand_faces,
            "obj_faces": obj_faces
        }
        volume = get_sample_intersect_volume(sample_info, mode="voxels")
        volume = volume * 1e6

        max_depth = compute_max_depth(obj_faces, obj_verts, hand_verts, penetr_mask)
        max_depth = max_depth * 100
    
    return volume, max_depth


def intersect_vox(obj_mesh, hand_mesh, pitch=0.01):
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume

def intersect(obj_mesh, hand_mesh, engine="auto"):
    trimesh.repair.fix_normals(obj_mesh)
    inter_mesh = obj_mesh.intersection(hand_mesh, engine=engine)
    return inter_mesh

def get_sample_intersect_volume(sample_info, mode="voxels"):
    hand_mesh = trimesh.Trimesh(vertices=sample_info["sbj_verts"], faces=sample_info["sbj_faces"])
    obj_mesh = trimesh.Trimesh(vertices=sample_info["obj_verts"], faces=sample_info["obj_faces"])
    if mode == "engines":
        try:
            # sudo apt install openscad
            intersection = intersect(obj_mesh, hand_mesh, engine="scad")
            if intersection.is_watertight:
                volume = intersection.volume
            else:
                intersection = intersect(obj_mesh, hand_mesh, engine="blender")
                if intersection.vertices.shape[0] == 0:
                    volume = 0
                elif intersection.is_watertight:
                    volume = intersection.volume
                else:
                    volume = None
        except Exception:
            # the scad engine throws an exception if there is no intersection
            intersection = intersect(obj_mesh, hand_mesh, engine="blender")
            if intersection.is_empty:
                volume = 0
            elif intersection.is_watertight:
                volume = intersection.volume
            else:
                volume = None
    elif mode == "voxels":
        volume = intersect_vox(obj_mesh, hand_mesh, pitch=0.005)
    return volume

def pre_compute_closest_dist(obj_faces, obj_vertices, sbj_vertices):
    obj_mesh = trimesh.Trimesh(vertices=obj_vertices, faces=obj_faces)
    trimesh.repair.fix_normals(obj_mesh)
    _, _dist_to_closets_point_on_obj, _, = trimesh.proximity.closest_point(obj_mesh, sbj_vertices)
    return _dist_to_closets_point_on_obj

def compute_max_depth(obj_faces, obj_vertices, sbj_vertices, penetr_mask):
    """
    Original source: https://github.com/hwjiang1510/GraspTTA/tree/master/metric/penetration.py
    """
    obj_mesh = trimesh.Trimesh(vertices=obj_vertices, faces=obj_faces)
    trimesh.repair.fix_normals(obj_mesh)

    if penetr_mask.sum() == 0:
        max_depth = 0.0
    else:
        dist_to_closest_point_on_obj = pre_compute_closest_dist(obj_faces, obj_vertices, sbj_vertices)
        max_depth = dist_to_closest_point_on_obj[penetr_mask == 1].max()

    return max_depth

def batch_mesh_contains_points(ray_origins, obj_triangles,
                               direction=torch.Tensor([0.4395064455, 0.617598629942, 0.652231566745])):
    """Times efficient but memory greedy !
    Computes ALL ray/triangle intersections and then counts them to determine
    if point inside mesh
    Args:
    ray_origins: (batch_size x point_nb x 3)
    obj_triangles: (batch_size, triangle_nb, vertex_nb=3, vertex_coords=3)
    tol_thresh: To determine if ray and triangle are //
    Returns:
    exterior: (batch_size, point_nb) 1 if the point is outside mesh, 0 else
    """
    tol_thresh = 0.0000001
    # ray_origins.requires_grad = False
    # obj_triangles.requires_grad = False
    batch_size = obj_triangles.shape[0]
    triangle_nb = obj_triangles.shape[1]
    point_nb = ray_origins.shape[1]

    # Batch dim and triangle dim will flattened together
    batch_points_size = batch_size * triangle_nb
    # Direction is random but shared
    v0, v1, v2 = obj_triangles[:, :, 0], obj_triangles[:, :, 1], obj_triangles[:, :, 2]
    # Get edges
    v0v1 = v1 - v0
    v0v2 = v2 - v0

    # Expand needed vectors
    batch_direction = direction.view(1, 1, 3).expand(batch_size, triangle_nb, 3)

    # Compute ray/triangle intersections
    pvec = torch.cross(batch_direction, v0v2, dim=2)
    dets = torch.bmm(
        v0v1.view(batch_points_size, 1, 3), pvec.view(batch_points_size, 3, 1)
    ).view(batch_size, triangle_nb)

    # Check if ray and triangle are parallel
    parallel = abs(dets) < tol_thresh
    invdet = 1 / (dets + 0.1 * tol_thresh)

    # Repeat mesh info as many times as there are rays
    triangle_nb = v0.shape[1]
    v0 = v0.repeat(1, point_nb, 1)
    v0v1 = v0v1.repeat(1, point_nb, 1)
    v0v2 = v0v2.repeat(1, point_nb, 1)
    hand_verts_repeated = (
        ray_origins.view(batch_size, point_nb, 1, 3)
        .repeat(1, 1, triangle_nb, 1)
        .view(ray_origins.shape[0], triangle_nb * point_nb, 3)
    )
    pvec = pvec.repeat(1, point_nb, 1)
    invdet = invdet.repeat(1, point_nb)
    tvec = hand_verts_repeated - v0
    u_val = (
            torch.bmm(
                tvec.view(batch_size * tvec.shape[1], 1, 3),
                pvec.view(batch_size * tvec.shape[1], 3, 1),
            ).view(batch_size, tvec.shape[1])
            * invdet
    )
    # Check ray intersects inside triangle
    u_correct = (u_val > 0) * (u_val < 1)
    qvec = torch.cross(tvec, v0v1, dim=2)

    batch_direction = batch_direction.repeat(1, point_nb, 1)
    v_val = (
            torch.bmm(
                batch_direction.view(batch_size * qvec.shape[1], 1, 3),
                qvec.view(batch_size * qvec.shape[1], 3, 1),
            ).view(batch_size, qvec.shape[1])
            * invdet
    )
    v_correct = (v_val > 0) * (u_val + v_val < 1)
    t = (
            torch.bmm(
                v0v2.view(batch_size * qvec.shape[1], 1, 3),
                qvec.view(batch_size * qvec.shape[1], 3, 1),
            ).view(batch_size, qvec.shape[1])
            * invdet
    )
    # Check triangle is in front of ray_origin along ray direction
    t_pos = t >= tol_thresh
    parallel = parallel.repeat(1, point_nb)
    # # Check that all intersection conditions are met
    not_parallel = ~parallel
    final_inter = v_correct * u_correct * not_parallel * t_pos
    # Reshape batch point/vertices intersection matrix
    # final_intersections[batch_idx, point_idx, triangle_idx] == 1 means ray
    # intersects triangle
    final_intersections = final_inter.view(batch_size, point_nb, triangle_nb)
    # Check if intersection number accross mesh is odd to determine if point is
    # outside of mesh
    exterior = final_intersections.sum(2) % 2 == 0
    return exterior

if __name__ == "__main__":
    with open(PHASE_SELECT_FILE, "r") as f:
        phase_select_dat = json.load(f)
    mesh_list = load_epic_mesh(MESH_DIR, phase_select=phase_select_dat)

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    mesh_list_local = [v for i, v in enumerate(mesh_list) if i % num_tasks == task_id]
    
    all_volume_results = defaultdict(list)
    all_max_depth_results = defaultdict(list)
    for mesh_data in tqdm(mesh_list_local):
        pen_volumes = []
        pen_max_depths = []
        for key, val in mesh_data.items():
            hand_mesh_path, object_mesh_path = val
            hand_mesh = trimesh.load(hand_mesh_path)
            object_mesh = trimesh.load(object_mesh_path)
            volume, max_depth = run_hoi_pen(hand_mesh, object_mesh)

            pen_volumes.append(volume)
            pen_max_depths.append(max_depth)
            all_volume_results[key].append(volume)
            all_max_depth_results[key].append(max_depth)
        min_volume = np.min(pen_volumes)
        min_max_depth = np.min(pen_max_depths)
        all_volume_results["min"].append(min_volume)
        all_max_depth_results["min"].append(min_max_depth)
    
    for key, val in all_volume_results.items():
        avg = np.mean(val)
        print(f"{key} interpenetraion volume: {avg:.6f}")
    for key, val in all_max_depth_results.items():
        avg = np.mean(val)
        print(f"{key} interpenetration max depth: {avg:.6f}")
