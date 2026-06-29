import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.ops import point_mesh_face_distance

def compute_sdf(points, obj_verts, obj_faces):
    """
    :param points: [B*T, N, 3] (Hand keypoints)
    :param obj_verts: [T, V, 3] (Object vertices)
    :param obj_faces: [F, 3] (Object faces)
    :return: dists [B*T, N] (Negative = inside, Positive = outside)
    """
    BT, N, _ = points.shape
    T, V, _ = obj_verts.shape
    B = BT // T

    # 1. Create a PyTorch3D Mesh object
    # We expand obj_verts to match the batch size of the points
    verts_expanded = obj_verts.unsqueeze(0).expand(B, -1, -1, -1).reshape(BT, V, 3)
    faces_expanded = obj_faces.unsqueeze(0).expand(BT, -1, -1)
    
    mesh = Meshes(verts=verts_expanded, faces=faces_expanded)

    # 2. Get Unsigned Distance
    # This returns the squared distance to the closest face
    # point_to_face shape: [BT, N]
    point_to_face = point_mesh_face_distance(mesh, points)
    unsigned_dist = torch.sqrt(point_to_face + 1e-8)

    # 3. Determine the Sign (Inside vs Outside)
    # We find the nearest face normal and check the dot product
    # with the vector from the face to the point.
    
    # Get centers and normals of all faces
    # faces_verts shape: [BT, F, 3, 3]
    faces_verts = mesh.verts_packed()[mesh.faces_packed()].reshape(BT, -1, 3, 3)
    face_centers = faces_verts.mean(dim=2) # [BT, F, 3]
    
    # Calculate normals via cross product
    v0, v1, v2 = faces_verts.unbind(2)
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    face_normals = F.normalize(face_normals, dim=-1) # [BT, F, 3]

    # For each point, find the index of the closest face
    # (In practice, point_mesh_face_distance doesn't return indices, 
    # so we often use knn_points or a custom op for this)
    # Here is a simplified logic:
    with torch.no_grad():
        # Find which face center is closest to each point as an approximation
        # for the sign check
        dist_to_centers = torch.cdist(points, face_centers) # [BT, N, F]
        closest_face_idx = dist_to_centers.argmin(dim=-1) # [BT, N]

    # Gather the normals of the closest faces
    # idx expanded for gather: [BT, N, 3]
    idx = closest_face_idx.unsqueeze(-1).expand(-1, -1, 3)
    nearest_normals = torch.gather(face_normals, 1, idx)

    # Vector from face center to point
    nearest_centers = torch.gather(face_centers, 1, idx)
    vec_face_to_pt = points - nearest_centers
    
    # Dot product: if > 0, point is "outside" (same direction as normal)
    # if < 0, point is "inside"
    sign = torch.sign(torch.sum(vec_face_to_pt * nearest_normals, dim=-1))
    
    # If sign is 0 (exactly on surface), treat as outside
    sign[sign == 0] = 1

    return unsigned_dist * sign