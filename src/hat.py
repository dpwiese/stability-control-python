"""
Collection of a few things for rigid body motion
"""

import math
from typing import Annotated, Literal, TypedDict
import numpy as np
import numpy.typing as npt
from numpy import (
    dot,
    eye,
    sin,
    cos,
    tan,
    arcsin,
    arccos,
    arctan,
    arctan2,
    trace,
    zeros
)
from numpy import linalg as la
# from scipy.spatial.transform import Rotation

# TODO@dpwiese - type all arrays with their actual size
# TODO@dpwiese - make custom types for stuff?
# TODO@dpwiese - np.float vs float???

# https://stackoverflow.com/questions/71109838/numpy-typing-with-specific-shape-and-datatype
Vec3 = Annotated[npt.NDArray[np.float64], Literal[3]]
Quat = Annotated[npt.NDArray[np.float64], Literal[4]]
Mat3x3 = Annotated[npt.NDArray[np.float64], Literal[3, 3]]
Mat4x4 = Annotated[npt.NDArray[np.float64], Literal[4, 4]]

class EquivalentAxis(TypedDict):
    """Exponential coordinats of rotation"""
    omega: Vec3
    theta: np.float64

class EulerAngles(TypedDict):
    """Euler angles"""
    psi:    np.float64
    theta:  np.float64
    phi:    np.float64

class TwistCoords(TypedDict):
    """Twist coordinats: angular velocity omega and linear velocity"""
    omega:      Vec3
    velocity:   Vec3
    theta:      np.float64

# TODO@dpwiese - handle bad inputs better
def hat(vec_in: Vec3) -> Mat3x3:
    """Take a 3 dimensional vector as an input and determine hat matrix"""

    # Make sure input is column
    assert vec_in.shape == (3,1)

    # Reshape to one dimensional array to make element access easier
    vec_in = vec_in.reshape(3,)

    # Build and return hat matrix
    return np.array([
        [0, -vec_in[2], vec_in[1]],
        [vec_in[2], 0, -vec_in[0]],
        [-vec_in[1], vec_in[0], 0]
    ])

# TODO@dpwiese - handle bad inputs better
def inv_hat(hat_in: Mat3x3) -> Vec3:
    """Takes a 3x3 skew symmetric matrix and returns vector of length 3"""

    # Make sure input is square 3x3 matrix
    assert hat_in.shape == (3,3)

    # Make sure input is skew-symmetric
    assert (hat_in.transpose() == -hat_in).all()

    # With skew-symmetric matrix, build output vector
    return np.array([[hat_in[2,1]], [-hat_in[2,0]], [hat_in[1,0]]])

def mrp_to_eqax(mrp: Vec3) -> EquivalentAxis:
    """
    Computes equivalent axis representation from Modified Rodrigues Parameters
    Inputs: Modified Rodrigues Parameters
    Output: angular velocity vector omega, theta
    """
    assert mrp.shape == (3,1)

    mrp_norm = la.norm(mrp)
    theta = 4 * arctan(la.norm(mrp))

    return {"omega": mrp.reshape(3,1) / mrp_norm, "theta": theta}

def eqax_to_mrp(eqax: EquivalentAxis) -> Vec3:
    """
    Computes Modified Rodrigues Parameters from equivalent axis representation
    Inputs: angular velocity vector omega, theta
    Output: Modified Rodrigues Parameters
    """

    omega = eqax["omega"]
    theta = eqax["theta"]

    # Make sure input is a 3x1 column vector
    assert omega.shape == (3,1)

    out: Vec3 = omega * tan(theta/4)
    return out

def eqax_to_rotation_matrix(eqax: EquivalentAxis) -> Mat3x3:
    """
    Computes rotation matrix from constant angular velocity vector over theta
    Inputs: angular velocity vector omega, theta
    Output: rotation matrix R
    """

    omega = eqax["omega"]
    theta = eqax["theta"]

    # Make sure input is a 3x1 column vector
    assert omega.shape == (3,1)

    # Angular velocity is zero, there is no rotation
    if (omega == np.array([0, 0, 0])).all():
        return eye(3, dtype=float)

    # If time over which the angular velocity occurs is zero, there is no rotation
    if theta == 0:
        return eye(3, dtype=float)

    # Calculate rotation matrix
    first_term: float = sin(la.norm(omega,2)*theta) * (hat(omega)/la.norm(omega))

    second_term_a = (1-cos(la.norm(omega,2)*theta))
    second_term_b = (la.matrix_power(hat(omega),2)/math.pow(la.norm(omega,2), 2))
    second_term: float = second_term_a * second_term_b

    # Return
    return eye(3, dtype=float) + first_term + second_term

def rotation_matrix_to_eqax(rot_mat: Mat3x3) -> EquivalentAxis:
    """
    Input: rotation matrix R
    Output: angular velocity vector omega, theta
    """

    # Make sure input is square 3x3 matrix
    assert rot_mat.shape == (3,3)

    theta = arccos((trace(rot_mat)-1)/2)

    if theta == 0:
        return {"omega": zeros((3,1)), "theta": theta}

    w_hat = (theta / (2 * sin(theta))) * (rot_mat - rot_mat.transpose())
    w_vector = inv_hat(w_hat)
    omega = w_vector / la.norm(w_vector, 2)

    # Return
    return {"omega": omega, "theta": theta}

def rotation_matrix_to_quat(rot_mat: Mat3x3) -> Quat:
    """
    Rotation matrix to quaternions
    Quaternion is in scalar-last format
    """
    eq_ax = rotation_matrix_to_eqax(rot_mat)

    # Scalar is q0, vector is q_bar
    q_scalar = np.array(cos(eq_ax["theta"]/2))
    q_vector = np.array(eq_ax["omega"] * sin(eq_ax["theta"]/2))

    # Return in scalar-last format
    return np.append(q_vector, q_scalar)

def quat_to_rotation_matrix(quat: Quat) -> Mat3x3:
    """
    Quaternions to rotation matrix
    Quaternion is in scalar-last format
    """

    # Make sure it is column
    assert quat.shape == (4,1)

    # Reshape to one dimensional array to make element access easier
    quat = quat.reshape(4,)

    q_scalar = quat[3]
    q_vector = quat[0:3]

    theta = 2 * arccos(q_scalar)

    if theta == 0:
        return eqax_to_rotation_matrix({"omega": zeros((3,1)), "theta": theta})

    omega = q_vector / sin(theta/2)

    return eqax_to_rotation_matrix({"omega": omega.reshape((3,1)), "theta": theta})

def rotation_matrix_to_euler(rot_mat: Mat3x3) -> EulerAngles:
    """
    Rotation matrix to euler angles
    Sequence is
    1. yaw (psi)
    2. pitch (theta)
    3. roll (phi)
    """
    r11 = rot_mat[0,0]
    r21 = rot_mat[1,0]
    r31 = rot_mat[2,0]
    r32 = rot_mat[2,1]
    r33 = rot_mat[2,2]

    return {
        "psi": arctan2(r21, r11),
        "theta": -arcsin(r31),
        "phi": arctan2(r32, r33)
        }

def euler_to_rotation_matrix(euler: EulerAngles) -> Mat3x3:
    """
    Euler angles to rotation matrix
    """

    # Input
    psi = euler["psi"]
    theta = euler["theta"]
    phi = euler["phi"]

    # First row
    r11 = cos(psi) * cos(theta)
    r12 = cos(psi) * sin(phi) * sin(theta) - cos(phi) * sin(psi)
    r13 = sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta)

    # Second row
    r21 = cos(theta) * sin(psi)
    r22 = cos(phi) * cos(psi) + sin(phi) * sin(psi) * sin(theta)
    r23 = cos(phi) * sin(psi) * sin(theta) - cos(psi) * sin(phi)

    # Third row
    r31 = -sin(theta)
    r32 = cos(theta) * sin(phi)
    r33 = cos(phi) * cos(theta)

    # Return
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def twist_coords_to_g(twist: TwistCoords) -> Mat4x4:
    """
    Twist coordinates of angular and linear velocity (omega, v) to the 'twist hat' 4x4 matrix
    """

    # Parse twist coordinates
    omega   = twist["omega"]
    velocity= twist["velocity"]
    theta   = twist["theta"]

    # Check inputs are the right shape
    assert omega.shape == (3,1)
    assert velocity.shape == (3,1)

    # Angular velocity is zero, there is no rotation
    if (omega == np.array([0, 0, 0])).all():
        top = np.append(eye(3, dtype=float), velocity * theta, axis=1)
        return np.append(top, np.array([[0, 0, 0, 1]]), axis=0)

    # TODO@dpwiese - If omega != 0, then t must first be rescaled such that norm(omega) = 1
    omega = omega / la.norm(omega)
    theta = theta * la.norm(omega)

    # Calculate each block of g matrix
    rot_mat = eqax_to_rotation_matrix({"omega": omega, "theta": theta})
    top_right_second_term = dot(omega.reshape(3,), velocity.reshape(3,))*omega*theta
    top_right = ((eye(3, dtype=float)-rot_mat) @ (hat(omega) @ velocity)) + top_right_second_term
    top = np.append(rot_mat, top_right, axis=1)

    # Return
    return np.append(top, np.array([[0, 0, 0, 1]]), axis=0)

# def g_to_twist_coords(g_mat: pt.NDArray[np.float64]) -> :
#     """
#     Twist coordinates of angular and linear velocity (omega, v) to the 'twist hat' 4x4 matrix
#     """

if __name__ == "__main__":
    print("Hello, world!")
