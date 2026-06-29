import numpy as np

def seal_mano(hand_mesh, seal_faces):
    '''
    Seal the mano faces to make the mesh watertight.
    
    :param hand_mesh: trimesh.Trimesh, hand mesh
    :param seal_faces: list, the faces for sealing the mesh
    '''
    faces_old = list(hand_mesh.faces)
    faces_old.extend(seal_faces)
    hand_mesh.faces = np.array(faces_old)
    return hand_mesh