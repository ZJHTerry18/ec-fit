import torch
import torch.nn as nn
import trimesh
from tqdm.auto import tqdm
import numpy as np
import copy
from smplx import MANO

from src.constants import MANO_MODEL_PATH
from src.utils.contact_mapping import calculate_hand_points, interpret_contact_points, select_pose_parameters
from src.utils.renderer_out import MySoftSilhouetteRenderer
from src.utils.structs import HandParams, ObjectParams
from src.utils.sdf.sdf.sdf_loss import SDFLoss


class Phase_3_Optimizer(nn.Module):
    def __init__(self,
        mano_params,
        hand_points,
        object_points,
        contact_mapping,
        body_pose_indices_to_opt, 
        hand_params,
        object_params,
        render_size,
        cam_intrinsic,
        sparse_dense_mapping,
        update_hand_trans=False,
        update_hand_rot=False,
    ):
        super(Phase_3_Optimizer, self).__init__()

        self.register_buffer('mano_betas', torch.tensor(mano_params['betas']).float().cuda())
        self.register_buffer('mano_global_orient', torch.tensor(mano_params['global_orient']).float().cuda())
        self.register_buffer('mano_transl', torch.tensor(mano_params['transl']).float().cuda())
        self.register_buffer('mano_pose', torch.tensor(mano_params['hand_pose']).float().cuda())
        self.hand_lr = hand_params.left_right
        self.sparse_dense_mapping = sparse_dense_mapping

        self.body_pose_indices_to_opt = body_pose_indices_to_opt

        self.mano_pose_opt = nn.Parameter(
            # torch.tensor(mano_params['hand_pose'][:, self.body_pose_indices_to_opt]).float().cuda(),
            torch.tensor(mano_params['hand_pose']).float().cuda(), 
            requires_grad=True
        )
        self.register_buffer('mano_pose_init', torch.tensor(mano_params['hand_pose']).float().cuda())
        if update_hand_trans:
            self.mano_transl_opt = nn.Parameter(
                torch.tensor(mano_params['transl']).float().cuda(),
                requires_grad=True
            )
            self.register_buffer('mano_transl_init', torch.tensor(mano_params['transl']).float().cuda())
        if update_hand_rot:
            self.mano_rot_opt = nn.Parameter(
                torch.tensor(mano_params['global_orient']).float().cuda(),
                requires_grad=True
            )
            self.register_buffer('mano_global_orient_init', torch.tensor(mano_params['global_orient']).float().cuda())

        self.register_buffer('hand_points', hand_points)
        self.register_buffer('object_points', object_points)
        self.contact_transfer_map = contact_mapping

        mano_model = MANO(MANO_MODEL_PATH, is_rhand=mano_params['is_rhand'], flat_hand_mean=True, use_pca=False)
        self.mano_model = mano_model.cuda()

        self.register_buffer('hum_vertices', hand_params.vertices)
        self.register_buffer('hum_faces', hand_params.faces)
        self.register_buffer('hum_centroid_offset', hand_params.centroid_offset)
        # self.register_buffer('hum_bbox', hand_params.bbox)
        self.register_buffer('hum_mask', hand_params.mask.float() if hand_params.mask is not None else None)
        self.register_buffer('obj_vertices', object_params.vertices)

        # # center the MANO mesh
        # newverts = self.get_hand_verts(remove_offset=False)
        # temp_mesh = trimesh.Trimesh(vertices=newverts.detach().cpu().numpy(), faces=hand_params.faces.detach().cpu().numpy())
        # self.mano_offset = torch.tensor(temp_mesh.centroid).float().cuda()

        self.renderer = MySoftSilhouetteRenderer(render_size, hand_params.faces, cam_intrinsic)

        # SDF collision loss setup
        self.sdf_loss = SDFLoss(object_params.faces, robustifier=1.0)

        self.upd_trans = update_hand_trans
        self.upd_rot = update_hand_rot

    def get_mano_params(self):
        hand_pose = self.mano_pose_opt
        transl = self.mano_transl_opt if self.upd_trans else self.mano_transl
        global_orient = self.mano_rot_opt if self.upd_rot else self.mano_global_orient
        return hand_pose, transl, global_orient

    def get_hand_verts(self, remove_offset=True):
        hand_pose, transl, global_orient = self.get_mano_params()
        output = self.mano_model(
            betas=self.mano_betas.unsqueeze(0),
            hand_pose=hand_pose.unsqueeze(0),
            global_orient=global_orient.unsqueeze(0),
            transl=transl.unsqueeze(0),
        )
        verts_hand = output.vertices[0]
        if remove_offset:
            verts_hand = verts_hand - self.hum_centroid_offset
        return verts_hand


    def calculate_contact_loss(self, upd_hand_vertices):
        new_hand_points = calculate_hand_points(
            upd_hand_vertices, self.contact_transfer_map, self.hand_lr, self.sparse_dense_mapping)
        loss = torch.nn.functional.mse_loss(new_hand_points, self.object_points)
        return {"loss_contact_p3": loss}

    def calculate_collision_loss(self, upd_hand_vertices):
        loss = self.sdf_loss.forward_upd_human(upd_hand_vertices, self.obj_vertices, scale_factor=0.01)
        return {"loss_collision_p3": loss}
    
    def calculate_pose_reg_loss(self):
        loss = torch.nn.functional.mse_loss(self.mano_pose_opt, self.mano_pose_init)
        return {"loss_pose_reg": loss}
    

    def calculate_silhouette_loss_iou(self, upd_hand_vertices):
        current_mask = self.renderer.render(
            upd_hand_vertices + self.hum_centroid_offset
        )
        intersection = torch.sum(current_mask * self.hum_mask)
        union = torch.sum((current_mask + self.hum_mask).clamp(0, 1))
        loss = 1 - intersection / union
        return {"loss_silhouette_hand": loss}

    def calculate_silhouette_loss_l2(self, upd_hand_vertices):
        loss = torch.tensor(0.0).cuda()
        pred_mask = self.renderer.render(upd_hand_vertices + self.hum_centroid_offset)
        loss = torch.nn.functional.mse_loss(pred_mask, self.hum_mask)
        return {"loss_silhouette_hand": loss}


    def forward(self, loss_weights: dict):
        upd_hand_vertices = self.get_hand_verts()

        loss_dict = {}
        if loss_weights["lw_contact_p3"] > 0:
            loss_dict.update(self.calculate_contact_loss(upd_hand_vertices))
        if loss_weights["lw_collision_p3"] > 0:
            loss_dict.update(self.calculate_collision_loss(upd_hand_vertices))
        if loss_weights["lw_pose_reg"] > 0:
            loss_dict.update(self.calculate_pose_reg_loss())
        if loss_weights["lw_silhouette_hand"] > 0 and self.hum_mask is not None:
            loss_dict.update(self.calculate_silhouette_loss_iou(upd_hand_vertices))

        return loss_dict
   

def optimize_phase3_hand(
    hand_params: HandParams, object_params: ObjectParams, contact_mapping: dict, render_size: list, cam_intrinsic: torch.Tensor,
    sparse_dense_mapping: dict=None, **kwargs,
):
    object_mesh = trimesh.Trimesh(vertices=object_params.vertices.detach().cpu().numpy(), faces=object_params.faces.detach().cpu().numpy())
    hand_mesh = trimesh.Trimesh(vertices=hand_params.vertices.detach().cpu().numpy(), faces=hand_params.faces.detach().cpu().numpy())

    hand_points, object_points = interpret_contact_points(
        contact_mapping, hand_mesh.vertices, object_mesh.vertices, object_mesh, hand_params.left_right,
        sparse_dense_map=sparse_dense_mapping,
    )

    opt_args = copy.deepcopy(kwargs)
    loss_weights = opt_args["loss_weights"]
    nr_phase_3_steps = opt_args["nr_phase_3_steps"]
    lr_phase_3 = opt_args.get("lr_phase_3", 0.01)
    # hard-code: For EPIC, increase penetration weight for objects with thin parts
    if opt_args["meta_info"].get("cat", None) in ["pan", "plate", "saucepan", "bowl"]:
        opt_args["loss_weights"]["lw_collision_p3"] *= 10.0
    update_hand_transl = opt_args.get("phase_3_upd_trans", False)
    update_hand_rot = opt_args.get("phase_3_upd_rot", False)

    model = Phase_3_Optimizer(
        hand_params.mano_params,
        hand_points,
        object_points,
        contact_mapping,
        None,
        hand_params,
        object_params,
        render_size,
        cam_intrinsic,
        sparse_dense_mapping,
        update_hand_trans=update_hand_transl,
        update_hand_rot=update_hand_rot
    )
    model.cuda()

    # optimizer params
    opt_params = [
        {'params': [model.mano_pose_opt], 'lr': lr_phase_3},
    ]
    if update_hand_transl:
        opt_params.append({'params': [model.mano_transl_opt], 'lr': lr_phase_3})
    if update_hand_rot:
        opt_params.append({'params': [model.mano_rot_opt], 'lr': lr_phase_3})

    # optimizer with separate learning rates for each parameter
    optimizer = torch.optim.Adam(opt_params)

    loop = tqdm(total=nr_phase_3_steps)
    for i in range(nr_phase_3_steps):
        optimizer.zero_grad()
        loss_dict = model(loss_weights)
        loss_dict_weighted = {
            k: loss_dict[k] * loss_weights[k.replace("loss", "lw")] for k in loss_dict
        }
        loss = sum(loss_dict_weighted.values())
        loss.backward() # remove retain_graph=True for memory reasons
        optimizer.step()

        if i % 50 == 0:
            loss_str = " | ".join([f"{k}: {loss_dict_weighted[k].item():.3g}" for k in loss_dict_weighted])
            print(loss_str)


    hand_parameters = {}
    updated_hand_vertices = model.get_hand_verts()
    hand_parameters["vertices"] = updated_hand_vertices.detach()
    hand_parameters["mano_pose_init"] = model.mano_pose_init
    hand_parameters["global_orient"] = model.mano_global_orient
    hand_parameters["transl"] = model.mano_transl
    hand_parameters["mano_pose_opt"] = model.mano_pose_opt.detach()
    if update_hand_transl:
        hand_parameters["transl_init"] = model.mano_transl
        hand_parameters["transl"] = model.mano_transl_opt.detach()
    if update_hand_rot:
        hand_parameters["global_orient_init"] = model.mano_global_orient
        hand_parameters["global_orient"] = model.mano_rot_opt.detach()
    hand_parameters["mano_betas"] = model.mano_betas
    hand_parameters["loss"] = {k: v.item() for k, v in loss_dict_weighted.items()} 

    return hand_parameters
