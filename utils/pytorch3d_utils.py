# utils functions from pytorch3d
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn.functional as F


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str="XYZ") -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str="XYZ") -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

import torch


def quaternion_to_matrix(quat: torch.Tensor, axis: str) -> torch.Tensor:
    """
    Converts a quaternion to a rotation matrix using PyTorch.

    Args:
        quat: A PyTorch tensor of shape (4,) representing the quaternion.
              The tensor should be in the format (w, x, y, z).
        axis: A string representing the order of rotations in the Euler angle
              convention. Valid values are 'XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', and 'ZYX'.

    Returns:
        A PyTorch tensor of shape (3, 3) representing the rotation matrix and its form is specified by the axis input.
    """
    # Extract the scalar and vector components of the quaternion
    w, x, y, z = quat

    # Calculate the matrix elements
    xx = x * x
    xy = x * y
    xz = x * z
    yy = y * y
    yz = y * z
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z

    # Construct the rotation matrix based on the specified Euler angle convention
    if axis == 'XYZ':
        matrix = torch.stack([
            torch.stack([1-2*(yy+zz), 2*(xy+wz), 2*(xz-wy)]),
            torch.stack([2*(xy-wz), 1-2*(xx+zz), 2*(yz+wx)]),
            torch.stack([2*(xz+wy), 2*(yz-wx), 1-2*(xx+yy)])
        ])
    elif axis == 'XZY':
        matrix = torch.stack([
            torch.stack([1-2*(yy+zz), 2*(xy+wz), 2*(xz+wy)]),
            torch.stack([2*(xy-wz), 1-2*(xx+zz), 2*(yz-wx)]),
            torch.stack([2*(xz-wy), 2*(yz+wx), 1-2*(xx+yy)])
        ])
    elif axis == 'YZX':
        matrix = torch.stack([
            torch.stack([1-2*(xx+zz), 2*(xy-wz), 2*(xz+wy)]),
            torch.stack([2*(xy+wz), 1-2*(yy+zz), 2*(yz-wx)]),
            torch.stack([2*(xz-wy), 2*(yz+wx), 1-2*(xx+yy)])
        ])
    elif axis == 'YXZ':
        matrix = torch.stack([
            torch.stack([1-2*(zz+yy), 2*(xy-wz), 2*(xz+wy)]),
            torch.stack([2*(xy+wz), 1-2*(zz+xx), 2*(yz-wx)]),
            torch.stack([2*(xz-wy), 2*(yz+wx), 1-2*(yy+xx)])
        ])

    elif axis == 'ZYX':
        matrix = torch.stack([
            torch.stack([1-2*(yy+zz), 2*(xz+wy), 2*(xy-wz)]),
            torch.stack([2*(xz-wy), 1-2*(xx+zz), 2*(yz+wx)]),
            torch.stack([2*(xy+wz), 2*(yz-wx), 1-2*(xx+yy)])
        ])

    elif axis == 'ZXY':
        matrix = torch.stack([
            torch.stack([1-2*(yy+zz), 2*(yz-wx), 2*(xy+wz)]),
            torch.stack([2*(yz+wx), 1-2*(zz+xx), 2*(xz-wy)]),
            torch.stack([2*(xy-wz), 2*(xz+wy), 1-2*(yy+xx)])
        ])
    else:
        raise ValueError(f"Invalid Euler angle convention '{axis}'")
    
    return matrix


def matrix_to_quaternion(matrix: torch.Tensor, axis: str) -> torch.Tensor:
    """
    Converts a rotation matrix to a quaternion using PyTorch.

    Args:
        matrix: A PyTorch tensor of shape (3, 3) representing the rotation matrix.
        axis: A string specifying the order of axes for the rotation matrix.
              The string should be composed of three letters representing the axes,
              with the first letter corresponding to the axis of rotation in the first
              rotation operation, and so on. For example, 'XYZ' means the rotation
              is performed first about the X axis, then the Y axis, then the Z axis.

    Returns:
        A PyTorch tensor of shape (4,) representing the quaternion.
    """
    # Check that the input matrix is valid
    if matrix.shape != (3, 3):
        raise ValueError("Invalid shape for rotation matrix. Expected (3, 3).")

    # Determine the order of axes for the rotation matrix
    if axis == 'XYZ':
        order = [0, 1, 2]
    elif axis == 'XZY':
        order = [0, 2, 1]
    elif axis == 'YXZ':
        order = [1, 0, 2]
    elif axis == 'YZX':
        order = [1, 2, 0]
    elif axis == 'ZXY':
        order = [2, 0, 1]
    elif axis == 'ZYX':
        order = [2, 1, 0]
    else:
        raise ValueError("Invalid axis order. Expected one of 'XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', or 'ZYX'.")

    # Extract the rotation angles from the rotation matrix
    r = torch.zeros(3, dtype=matrix.dtype, device=matrix.device)
    for i in range(3):
        j, k = order[i], order[(i+1)%3]
        r[i] = torch.atan2(matrix[j,k], matrix[i,i])

    # Construct the quaternion from the rotation angles
    qw = torch.cos(r[0]/2) * torch.cos(r[1]/2) * torch.cos(r[2]/2) + \
         torch.sin(r[0]/2) * torch.sin(r[1]/2) * torch.sin(r[2]/2)
    qx = torch.sin(r[0]/2) * torch.cos(r[1]/2) * torch.cos(r[2]/2) - \
         torch.cos(r[0]/2) * torch.sin(r[1]/2) * torch.sin(r[2]/2)
    qy = torch.cos(r[0]/2) * torch.sin(r[1]/2) * torch.cos(r[2]/2) + \
         torch.sin(r[0]/2) * torch.cos(r[1]/2) * torch.sin(r[2]/2)
    qz = torch.cos(r[0]/2) * torch.cos(r[1]/2) * torch.sin(r[2]/2) - \
         torch.sin(r[0]/2) * torch.sin(r[1]/2) * torch.cos(r[2]/2)

    # Combine the scalar and vector components of the quaternion
    quat = torch.tensor([qw, qx, qy, qz], dtype=matrix.dtype, device=matrix.device)

    return quat
