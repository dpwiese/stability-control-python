"""
Collection of a few things for rigid body motion
"""

import math
from typing import Annotated, Literal, TypedDict
import numpy as np
import numpy.typing as npt
import numpy.linalg as la

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
    # TODO@dpwiese - uncomment the below
    # assert (hat_in.transpose() == -hat_in).all()

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
    theta = 4 * np.arctan(la.norm(mrp))

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

    out: Vec3 = omega * np.tan(theta/4)
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
        return np.eye(3, dtype=float)

    # If time over which the angular velocity occurs is zero, there is no rotation
    if theta == 0:
        return np.eye(3, dtype=float)

    # Calculate rotation matrix
    first_term: float = np.sin(la.norm(omega,2)*theta) * (hat(omega)/la.norm(omega))

    second_term_a = (1-np.cos(la.norm(omega,2)*theta))
    second_term_b = (la.matrix_power(hat(omega),2)/math.pow(la.norm(omega,2), 2))
    second_term: float = second_term_a * second_term_b

    # Return
    return np.eye(3, dtype=float) + first_term + second_term

def rotation_matrix_to_eqax(rot_mat: Mat3x3) -> EquivalentAxis:
    """
    Input: rotation matrix R
    Output: angular velocity vector omega, theta
    """

    # Make sure input is square 3x3 matrix
    assert rot_mat.shape == (3,3)

    theta = np.arccos((np.trace(rot_mat)-1)/2)

    if theta == 0:
        return {"omega": np.zeros((3,1)), "theta": theta}

    w_hat = (theta / (2 * np.sin(theta))) * (rot_mat - rot_mat.transpose())
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
    q_scalar = np.array(np.cos(eq_ax["theta"]/2))
    q_vector = np.array(eq_ax["omega"] * np.sin(eq_ax["theta"]/2))

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

    theta = 2 * np.arccos(q_scalar)

    if theta == 0:
        return eqax_to_rotation_matrix({"omega": np.zeros((3,1)), "theta": theta})

    omega = q_vector / np.sin(theta/2)

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
        "psi": np.arctan2(r21, r11),
        "theta": -np.arcsin(r31),
        "phi": np.arctan2(r32, r33)
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
    r11 = np.cos(psi) * np.cos(theta)
    r12 = np.cos(psi) * np.sin(phi) * np.sin(theta) - np.cos(phi) * np.sin(psi)
    r13 = np.sin(phi) * np.sin(psi) + np.cos(phi) * np.cos(psi) * np.sin(theta)

    # Second row
    r21 = np.cos(theta) * np.sin(psi)
    r22 = np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(psi) * np.sin(theta)
    r23 = np.cos(phi) * np.sin(psi) * np.sin(theta) - np.cos(psi) * np.sin(phi)

    # Third row
    r31 = -np.sin(theta)
    r32 = np.cos(theta) * np.sin(phi)
    r33 = np.cos(phi) * np.cos(theta)

    # Return
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def twist_coords_to_config_g(twist: TwistCoords) -> Mat4x4:
    """
    Twist coordinates of angular and linear velocity (omega, v) to the 4x4 configuration matrix g.
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
        top = np.append(np.eye(3, dtype=float), velocity * theta, axis=1)
        return np.append(top, np.array([[0, 0, 0, 1]]), axis=0)

    # TODO@dpwiese - If omega != 0, then t must first be rescaled such that norm(omega) = 1
    omega = omega / la.norm(omega)
    theta = theta * la.norm(omega)

    # Calculate each block of g matrix
    rot_mat = eqax_to_rotation_matrix({"omega": omega, "theta": theta})
    top_right_second_term = np.dot(omega.reshape(3,), velocity.reshape(3,))*omega*theta
    top_right = ((np.eye(3, dtype=float)-rot_mat) @ (hat(omega) @ velocity)) + top_right_second_term
    top = np.append(rot_mat, top_right, axis=1)

    # Return
    return np.append(top, np.array([[0, 0, 0, 1]]), axis=0)

def config_g_to_twist_coords(g_mat: npt.NDArray[np.float64]) -> TwistCoords:
    """
    4x4 configuration matrix g to twist coordinates of angular and linear velocity (omega, v)
    """

    rot_mat = g_mat[0:3,0:3]

    if (rot_mat == np.eye(3, dtype=float)).all():
        omega = np.zeros((3,1))
        velocity = g_mat[0:3,3].reshape(3,1)
        theta = 0
    else:
        eq_ax = rotation_matrix_to_eqax(rot_mat)
        omega = eq_ax["omega"]
        theta = eq_ax["theta"]

        # pylint: disable-next=C0301
        velocity = inv((np.eye(3, dtype=float)-rot_mat) * hat(omega) + (np.dot(omega, omega))*theta) * g_mat[0:3,3]
        # velocity = velocity.reshape(3,1)

    return {"omega": omega, "velocity": velocity, "theta": theta}

if __name__ == "__main__":
    print("Hello, world!")
