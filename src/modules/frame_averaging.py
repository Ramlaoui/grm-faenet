import torch
from copy import deepcopy
from itertools import product
import random

def compute_frames(eigenvec, pos, cell, fa_method="random", pos_3D=None, det_index=0):
    """Compute all frames for a given graph.

    Args:
        eigenvec (tensor): eigenvectors matrix
        pos (tensor): centered position vector
        cell (tensor): cell direction (dxd)
        fa_method (str): the Frame Averaging (FA) inspired technique
            chosen to select frames: stochastic-FA (random), deterministic-FA (det),
            Full-FA (all) or SE(3)-FA (se3).
        pos_3D: for 2D FA, pass atoms' 3rd position coordinate.

    Returns:
        list: 3D position tensors of projected representation
    """
    dim = pos.shape[1]  # to differentiate between 2D or 3D case
    plus_minus_list = list(product([1, -1], repeat=dim))
    plus_minus_list = [torch.tensor(x) for x in plus_minus_list]
    all_fa_pos = []
    all_cell = []
    all_rots = []
    assert fa_method in {
        "all",
        "random",
        "det",
        "se3-all",
        "se3-random",
        "se3-det",
    }
    se3 = fa_method in {
        "se3-all",
        "se3-random",
        "se3-det",
    }
    fa_cell = deepcopy(cell)

    if fa_method == "det" or fa_method == "se3-det":
        sum_eigenvec = torch.sum(eigenvec, axis=0)
        plus_minus_list = [torch.where(sum_eigenvec >= 0, 1.0, -1.0)]

    for pm in plus_minus_list:
        # Append new graph positions to list
        new_eigenvec = pm * eigenvec

        # Consider frame if it passes above check
        fa_pos = pos @ new_eigenvec

        if pos_3D is not None:
            full_eigenvec = torch.eye(3)
            fa_pos = torch.cat((fa_pos, pos_3D.unsqueeze(1)), dim=1)
            full_eigenvec[:2, :2] = new_eigenvec
            new_eigenvec = full_eigenvec

        if cell is not None:
            fa_cell = cell @ new_eigenvec

        # Check if determinant is 1 for SE(3) case
        if se3 and not torch.allclose(
            torch.linalg.det(new_eigenvec), torch.tensor(1.0), atol=1e-03
        ):
            continue

        all_fa_pos.append(fa_pos)
        all_cell.append(fa_cell)
        all_rots.append(new_eigenvec.unsqueeze(0))

    # Handle rare case where no R is positive orthogonal
    if all_fa_pos == []:
        all_fa_pos.append(fa_pos)
        all_cell.append(fa_cell)
        all_rots.append(new_eigenvec.unsqueeze(0))

    # Return frame(s) depending on method fa_method
    if fa_method == "all" or fa_method == "se3-all":
        return all_fa_pos, all_cell, all_rots

    elif fa_method == "det" or fa_method == "se3-det":
        return [all_fa_pos[det_index]], [all_cell[det_index]], [all_rots[det_index]]

    index = random.randint(0, len(all_fa_pos) - 1)
    return [all_fa_pos[index]], [all_cell[index]], [all_rots[index]]


def check_constraints(eigenval, eigenvec, dim=3):
    """Check requirements for frame averaging are satisfied

    Args:
        eigenval (tensor): eigenvalues
        eigenvec (tensor): eigenvectors
        dim (int): 2D or 3D frame averaging
    """
    # Check eigenvalues are different
    if dim == 3:
        if (eigenval[1] / eigenval[0] > 0.90) or (eigenval[2] / eigenval[1] > 0.90):
            print("Eigenvalues are quite similar")
    else:
        if eigenval[1] / eigenval[0] > 0.90:
            print("Eigenvalues are quite similar")

    # Check eigenvectors are orthonormal
    if not torch.allclose(eigenvec @ eigenvec.T, torch.eye(dim), atol=1e-03):
        print("Matrix not orthogonal")

    # Check determinant of eigenvectors is 1
    if not torch.allclose(torch.linalg.det(eigenvec), torch.tensor(1.0), atol=1e-03):
        print("Determinant is not 1")

def frame_averaging_2D(pos, cell=None, fa_method="random", check=False):
    """Computes new positions for the graph atoms,
    based on a frame averaging building on PCA.

    Args:
        pos (tensor): positions of atoms in the graph
        cell (tensor): unit cell of the graph
        fa_method (str): FA method used (random, det, all, se3)
        check (bool): check if constraints are satisfied. Default: False.

    Returns:
        tensor: updated atom positions
        tensor: updated unit cell
        tensor: the rotation matrix used (PCA)
    """

    # Compute centroid and covariance
    pos_2D = pos[:, :2] - pos[:, :2].mean(dim=0, keepdim=True)
    C = torch.matmul(pos_2D.t(), pos_2D)

    # Eigendecomposition
    eigenval, eigenvec = torch.linalg.eigh(C)
    # Sort eigenvalues
    idx = eigenval.argsort(descending=True)
    eigenval = eigenval[idx]
    eigenvec = eigenvec[:, idx]

    # Check if constraints are satisfied
    if check:
        check_constraints(eigenval, eigenvec, 3)

    # Compute all frames
    fa_pos, fa_cell, fa_rot = compute_frames(
        eigenvec, pos_2D, cell, fa_method, pos[:, 2]
    )
    # No need to update distances, they are preserved.

    return fa_pos, fa_cell, fa_rot

def data_augmentation(g, d=3, *args):
    """Data augmentation where we randomly rotate each graph
    in the dataloader transform

    Args:
        g (data.Data): single graph
        d (int): dimension of the DA rotation (2D around z-axis or 3D)
        rotation (str, optional): around which axis do we rotate it.
            Defaults to 'z'.

    Returns:
        (data.Data): rotated graph
    """

    # Sampling a random rotation within [-180, 180] for all axes.
    if d == 3:
        transform = RandomRotate([-180, 180], [0, 1, 2])  # 3D
    else:
        transform = RandomRotate([-180, 180], [2])  # 2D around z-axis

    # Rotate graph
    graph_rotated, _, _ = transform(g)

    return graph_rotated

class Transform:
    def __call__(self, data):
        raise NotImplementedError

    def __str__(self):
        name = self.__class__.__name__
        items = [
            f"{k}={v}"
            for k, v in self.__dict__.items()
            if not callable(v) and k != "inactive"
        ]
        s = f"{name}({', '.join(items)})"
        if self.inactive:
            s = f"[inactive] {s}"
        return s


class FrameAveraging(Transform):
    r"""Frame Averaging (FA) Transform for (PyG) Data objects (e.g. 3D atomic graphs).
    Computes new atomic positions (`fa_pos`) for all datapoints, as well as new unit
    cells (`fa_cell`) attributes for crystal structures, when applicable. The rotation
    matrix (`fa_rot`) used for the frame averaging is also stored.

    Args:
        frame_averaging (str): Transform method used.
            Can be 2D FA, 3D FA, Data Augmentation or no FA, respectively denoted by
            (`"2D"`, `"3D"`, `"DA"`, `""`)
        fa_method (str): the actual frame averaging technique used.
            "stochastic" refers to sampling one frame at random (at each epoch), "det"
            to chosing deterministically one frame, and "all" to using all frames. The
            prefix "se3-" refers to the SE(3) equivariant version of the method. ""
            means that no frame averaging is used. (`""`, `"stochastic"`, `"all"`,
            `"det"`, `"se3-stochastic"`, `"se3-all"`, `"se3-det"`)

    Returns:
        (data.Data): updated data object with new positions (+ unit cell) attributes
        and the rotation matrices used for the frame averaging transform.
    """

    def __init__(self, frame_averaging=None, fa_method=None):
        self.fa_method = (
            "random" if (fa_method is None or fa_method == "") else fa_method
        )
        self.frame_averaging = "" if frame_averaging is None else frame_averaging
        self.inactive = not self.frame_averaging
        assert self.frame_averaging in {
            "",
            "2D",
            # "DA",
        }
        assert self.fa_method in {
            "",
            "random",
            "det",
            "all",
            "se3-random",
            "se3-det",
            "se3-all",
        }

        if self.frame_averaging:
            if self.frame_averaging == "2D":
                self.fa_func = frame_averaging_2D
            # elif self.frame_averaging == "DA":
            #     self.fa_func = data_augmentation
            else:
                raise ValueError(f"Unknown frame averaging: {self.frame_averaging}")

    def __call__(self, data):
        if self.inactive:
            return data
        elif self.frame_averaging == "DA":
            return self.fa_func(data, self.fa_method)
        else:
            data.fa_pos, data.fa_cell, data.fa_rot = self.fa_func(
                data.pos, data.cell if hasattr(data, "cell") else None, self.fa_method
            )
            return data



