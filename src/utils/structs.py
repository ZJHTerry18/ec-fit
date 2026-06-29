import torch

class HandParams:
    def __init__(
        self,
        vertices,
        faces,
        centroid_offset,
        left_right,
        # bbox,
        mask,
        mano_params,
    ):
        self.vertices = vertices
        self.faces = faces
        self.centroid_offset = centroid_offset
        self.left_right = left_right
        # self.bbox = bbox
        self.mask = mask
        self.mano_params = mano_params

    def to_cuda(self):
        self.vertices = torch.from_numpy(self.vertices).float().cuda()
        self.faces = torch.from_numpy(self.faces).float().cuda()
        self.centroid_offset = torch.from_numpy(self.centroid_offset).float().cuda()
        # self.bbox = torch.from_numpy(self.bbox).float().cuda()
        self.mask = torch.tensor(self.mask).cuda() if self.mask is not None else None
    
    def to_cpu(self):
        self.vertices = self.vertices.detach().cpu().numpy()
        self.faces = self.faces.detach().cpu().numpy()
        self.centroid_offset = self.centroid_offset.detach().cpu().numpy()
        self.mask = self.mask.detach().cpu().numpy() if self.mask is not None else None
    
    def __str__(self):
        return ('HumanParams:\n'
            f'  - Vertices: {self.vertices.shape}, {type(self.vertices)} {self.vertices[:5]}\n'
            f'  - Faces: {self.faces.shape}, {type(self.faces)} {self.faces[:5]}\n'
            f'  - Centroid offset: {type(self.centroid_offset)} {self.centroid_offset}\n'
            f'  - Left/Right: {type(self.left_right)} {self.left_right}\n'
            # f'  - Bbox: {type(self.bbox)} {self.bbox}\n'
            # f'  - Mask: {self.mask.shape}'
        )


class ObjectParams:
    def __init__(
        self,
        vertices,
        faces,
        rotation_offset,
        centroid_offset,
        mask,
        occ_mask,
        scale,
    ):
        self.vertices = vertices
        self.faces = faces
        self.rotation_offset = rotation_offset
        self.centroid_offset = centroid_offset
        self.mask = mask
        self.occ_mask = occ_mask
        self.scale = scale

    def to_cuda(self):
        self.vertices = torch.from_numpy(self.vertices).float().cuda()
        self.faces = torch.from_numpy(self.faces).float().cuda()
        self.rotation_offset = torch.from_numpy(self.rotation_offset).float().cuda()
        self.centroid_offset = torch.from_numpy(self.centroid_offset).float().cuda()
        self.mask = torch.tensor(self.mask).cuda() if self.mask is not None else None
        self.occ_mask = torch.tensor(self.occ_mask).cuda() if self.occ_mask is not None else None
        self.scale = torch.tensor(self.scale).float().cuda()
    
    def to_cpu(self):
        self.vertices = self.vertices.detach().cpu().numpy()
        self.faces = self.faces.detach().cpu().numpy()
        self.rotation_offset = self.rotation_offset.detach().cpu().numpy()
        self.centroid_offset = self.centroid_offset.detach().cpu().numpy()
        self.mask = self.mask.detach().cpu() if self.mask is not None else None
        self.occ_mask = self.occ_mask.detach().cpu() if self.occ_mask is not None else None
        self.scale = self.scale.detach().cpu().numpy()

    def __str__(self):
        return ('ObjectParams:\n'
            f'  - Vertices: {self.vertices.shape}, {type(self.vertices)} {self.vertices[:5]}\n'
            f'  - Faces: {self.faces.shape}, {type(self.faces)} {self.faces[:5]}\n'
            f'  - Rotation offset: {type(self.rotation_offset)} {self.rotation_offset}\n'
            f'  - Centroid offset: {type(self.centroid_offset)} {self.centroid_offset}\n'
            # f'  - Mask: {self.mask.shape}, {type(self.mask)} {self.mask[:5]}\n'
            f'  - Scale: {self.scale}')
