import torch
import torch.nn as nn
import trimesh
import numpy as np
from tqdm.auto import tqdm

from src.utils.contact_mapping import calculate_object_points, interpret_contact_points, apply_transformation
from src.utils.geometry import rot6d_to_matrix, rotation_matrix_to_angle_axis
from src.utils.structs import HandParams, ObjectParams


class Phase_1_Optimizer(nn.Module):
    def __init__(self,
        rotation_init,
        translation_init,
        hand_points,
        object_points,
        hand_vertices,
        obj_vertices,
        obj_faces,
        contact_mapping
    ):
        super(Phase_1_Optimizer, self).__init__()

        self.rotation = nn.Parameter(rotation_init.clone().float(), requires_grad=True)
        self.translation = nn.Parameter(translation_init.clone().float(), requires_grad=True)

        self.register_buffer('hand_points', hand_points)
        self.register_buffer('object_points', object_points)
        self.register_buffer('hand_vertices', hand_vertices)
        self.register_buffer('obj_vertices', obj_vertices)
        self.register_buffer('obj_faces', obj_faces)
        self.contact_transfer_map = contact_mapping

    def calculate_contact_loss(self, upd_obj_vertices):
        new_object_points = calculate_object_points(upd_obj_vertices, self.contact_transfer_map, self.obj_faces)
        return torch.nn.functional.mse_loss(self.hand_points, new_object_points)

    def forward(self):
        # print(f"{self.rotation.detach().cpu()} {self.translation.detach().cpu()}")
        new_obj_vertices = apply_transformation(self.obj_vertices, self.rotation, self.translation)
        loss = self.calculate_contact_loss(new_obj_vertices)
        return loss
   

def optimize_phase1_contact(
    hand_params: HandParams, object_params: ObjectParams, contact_mapping: dict, sparse_dense_mapping: dict = None,
    **kwargs
):
    object_mesh = trimesh.Trimesh(vertices=object_params.vertices.detach().cpu().numpy(), faces=object_params.faces.detach().cpu().numpy())
    hand_mesh = trimesh.Trimesh(vertices=hand_params.vertices.detach().cpu().numpy(), faces=hand_params.faces.detach().cpu().numpy())

    hand_points, object_points = interpret_contact_points(
        contact_mapping, hand_mesh.vertices, object_mesh.vertices, object_mesh, hand_params.left_right,
        sparse_dense_map=sparse_dense_mapping,
    )

    nr_phase_1_steps = kwargs["nr_phase_1_steps"]
    lr_rotation = kwargs["lr_rotation_phase_1"]
    lr_translation = kwargs["lr_translation_phase_1"]

    rotation_init = torch.tensor([1.01, 0.01, 0.01, 1.01, 0.01, 0.01], requires_grad=True).cuda()
    translation_init = torch.tensor([0.0, 0.0, 0.0], requires_grad=True).cuda()

    model = Phase_1_Optimizer(
        rotation_init,
        translation_init,
        hand_points,
        object_points,
        hand_params.vertices,
        object_params.vertices,
        object_params.faces,
        contact_mapping,
    )
    model.cuda()

    # optimizer with separate learning rates for each parameter
    optimizer = torch.optim.Adam([
        {'params': [model.rotation], 'lr': lr_rotation},
        {'params': [model.translation], 'lr': lr_translation},
    ])

    loop = tqdm(total=nr_phase_1_steps)
    for i in range(nr_phase_1_steps):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        # print(model.rotation.grad, model.translation.grad)
        optimizer.step()
        # loop.set_description(f'loss: {loss.item():.3g}')
        # loop.update()

        if i % 50 == 0:
            loss_str = f"loss_contact: {loss.item():.3g}"
            print(loss_str)

    object_parameters = {}
    object_parameters["rotation"] = rotation_matrix_to_angle_axis(rot6d_to_matrix(model.rotation).detach())
    object_parameters["translation"] = model.translation.unsqueeze(0).detach()
    transformed_obj_vertices = apply_transformation(object_params.vertices, model.rotation, model.translation)
    object_parameters['vertices'] = transformed_obj_vertices.detach()
    object_parameters['loss'] = {"loss_contact": loss.item()}

    return object_parameters
