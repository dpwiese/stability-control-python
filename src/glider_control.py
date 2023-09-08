"""
Stabilizing glider - 16.333 HW #2
"""

import math
import control
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import rigidbody as rb
import numpy.linalg as la

def mass_prop(geo, xyz):

    # qc - quarter chord

    ################################################################################################

    # Calculate mass of wing, horizontal tail, and vertical tail: add together with tip mass to get total airplane mass
    geo["wing"]["m"] = geo["dens"]["bal"] * geo["wing"]["S"]
    geo["htai"]["m"] = geo["dens"]["bal"] * geo["htai"]["S"]
    geo["vtai"]["m"] = geo["dens"]["bal"] * geo["vtai"]["S"]
    geo["totl"]["m"] = geo["wing"]["m"] + geo["htai"]["m"] + geo["vtai"]["m"] + geo["tipm"]["m"]

    # Calculate distances to surfaces relative to coordinates xyz at tip
    # Distance in x-direction
    xyz["wing"]["xdcg"] = xyz["wing"]["xdle"] - 0.5 * geo["wing"]["c"]
    xyz["htai"]["xdcg"] = xyz["htai"]["xdle"] - 0.5 * geo["htai"]["c"]
    xyz["vtai"]["xdcg"] = xyz["vtai"]["xdle"] - 0.5 * geo["vtai"]["c"]
    xyz["tipm"]["xdcg"] = 0

    # Distance in z-direction
    xyz["vtai"]["zdcg"] = -0.5 * geo["vtai"]["h"]

    # # Calculate CG location wrt to coordinates xyz at airplane tip
    xyz["cg"]["x"] = (xyz["wing"]["xdcg"]*geo["wing"]["m"])+(xyz["htai"]["xdcg"]*geo["htai"]["m"])+(xyz["vtai"]["xdcg"]*geo["vtai"]["m"])+(xyz["tipm"]["xdcg"]*geo["tipm"]["m"]) / geo["totl"]["m"]
    xyz["cg"]["y"] = 0
    xyz["cg"]["z"] = (xyz["vtai"]["zdcg"]*geo["vtai"]["m"]) / geo["totl"]["m"]
    xyz["cg"]["totl"] = np.array([xyz["cg"]["x"], xyz["cg"]["y"], xyz["cg"]["z"]])

    ################################################################################################

    # Calculate distances to surfaces relative to coordinates XYZ at CG
    XYZ = {
        "cg": {
            "totl": np.array([0, 0, 0])
        },
        # Tip mass
        "tipm": {
            "xdcg": xyz["cg"]["x"]
        },
        "wing": {},
        "htai": {},
        "vtai": {}
    }

    # Wing
    XYZ["wing"]["xdle"] = XYZ["tipm"]["xdcg"] + xyz["wing"]["xdle"]
    XYZ["wing"]["xdqc"] = XYZ["wing"]["xdle"] - 0.25*geo["wing"]["c"]
    XYZ["wing"]["xdcg"] = XYZ["wing"]["xdle"] - 0.5*geo["wing"]["c"]

    # Horizontal tail
    XYZ["htai"]["xdle"] = XYZ["tipm"]["xdcg"] + xyz["htai"]["xdle"]
    XYZ["htai"]["xdqc"] = XYZ["htai"]["xdle"] - 0.25*geo["htai"]["c"]
    XYZ["htai"]["xdcg"] = XYZ["htai"]["xdle"] - 0.5*geo["htai"]["c"]

    # Vertical tail
    XYZ["vtai"]["xdle"] = XYZ["tipm"]["xdcg"] + xyz["vtai"]["xdle"]
    XYZ["vtai"]["xdqc"] = XYZ["vtai"]["xdle"] - 0.25*geo["vtai"]["c"]
    XYZ["vtai"]["xdcg"] = XYZ["vtai"]["xdle"] - 0.5*geo["vtai"]["c"]

    # Y distances: all zero

    # Z distances
    XYZ["tipm"]["zdcg"] = -xyz["cg"]["z"]
    XYZ["wing"]["zdcg"] = -xyz["cg"]["z"]
    XYZ["htai"]["zdcg"] = -xyz["cg"]["z"]
    XYZ["vtai"]["zdcg"] = xyz["vtai"]["zdcg"]-xyz["cg"]["z"]

    # Radial vectors from cg to aerodynamics surface quarter chords
    XYZ["wing"]["rqc"] = np.array([XYZ["wing"]["xdqc"], 0, XYZ["wing"]["zdcg"]])
    XYZ["htai"]["rqc"] = np.array([XYZ["htai"]["xdqc"], 0, XYZ["htai"]["zdcg"]])
    XYZ["vtai"]["rqc"] = np.array([XYZ["vtai"]["xdqc"], 0, XYZ["vtai"]["zdcg"]])

    ################################################################################################

    # Distances are from CG to plate

    J_XX_wing = (1/12)*geo["wing"]["m"] * math.pow(geo["wing"]["b"], 2) + geo["wing"]["m"] * math.pow(XYZ["wing"]["zdcg"], 2)
    J_XX_htai = (1/12)*geo["htai"]["m"] * math.pow(geo["htai"]["b"], 2) + geo["htai"]["m"] * math.pow(XYZ["htai"]["zdcg"], 2)
    J_XX_vtai = (1/12)*geo["vtai"]["m"] * math.pow(geo["vtai"]["h"], 2) + geo["vtai"]["m"] * math.pow(XYZ["vtai"]["zdcg"], 2)
    J_XX_mtip = geo["tipm"]["m"] * math.pow(XYZ["tipm"]["zdcg"], 2)
    JXX = J_XX_wing + J_XX_htai + J_XX_vtai + J_XX_mtip

    J_YY_wing = (1/12)*geo["wing"]["m"] * math.pow(geo["wing"]["c"], 2) + geo["wing"]["m"] * (math.pow(XYZ["wing"]["xdcg"], 2) + math.pow(XYZ["wing"]["zdcg"], 2))
    J_YY_htai = (1/12)*geo["htai"]["m"] * math.pow(geo["htai"]["c"], 2) + geo["htai"]["m"]*(math.pow(XYZ["htai"]["xdcg"], 2)+math.pow(XYZ["htai"]["zdcg"], 2))
    J_YY_vtai = (1/12)*geo["vtai"]["m"] * (math.pow(geo["vtai"]["c"], 2) + math.pow(geo["vtai"]["h"], 2)) + geo["vtai"]["m"]*(math.pow(XYZ["vtai"]["xdcg"], 2)+math.pow(XYZ["vtai"]["zdcg"], 2))
    J_YY_mtip = geo["tipm"]["m"]*(math.pow(XYZ["tipm"]["xdcg"], 2) + math.pow(XYZ["tipm"]["zdcg"], 2))
    JYY = J_YY_wing + J_YY_htai + J_YY_vtai + J_YY_mtip

    J_ZZ_wing = (1/12)*geo["wing"]["m"] * (math.pow(geo["wing"]["c"], 2) + math.pow(geo["wing"]["b"], 2))+geo["wing"]["m"]*math.pow(XYZ["wing"]["xdcg"], 2)
    J_ZZ_htai = (1/12)*geo["htai"]["m"] * (math.pow(geo["htai"]["c"], 2) + math.pow(geo["htai"]["b"], 2))+geo["htai"]["m"]*math.pow(XYZ["htai"]["xdcg"], 2)
    J_ZZ_vtai = (1/12)*geo["vtai"]["m"] * math.pow(geo["vtai"]["c"], 2) + geo["vtai"]["m"]*math.pow(XYZ["vtai"]["xdcg"], 2)
    J_ZZ_mtip = geo["tipm"]["m"]*math.pow(XYZ["tipm"]["xdcg"], 2)
    JZZ = J_ZZ_wing + J_ZZ_htai + J_ZZ_vtai + J_ZZ_mtip

    J_xy_wing = 0
    J_xy_htai = 0
    J_xy_vtai = 0
    J_xy_mtip = 0
    Jxy = J_xy_wing + J_xy_htai + J_xy_vtai + J_xy_mtip
    Jyx = Jxy

    J_xz_wing = geo["wing"]["m"] * (XYZ["wing"]["xdcg"])*(XYZ["wing"]["zdcg"])
    J_xz_htai = geo["htai"]["m"] * (XYZ["htai"]["xdcg"])*(XYZ["htai"]["zdcg"])
    J_xz_vtai = geo["vtai"]["m"] * (XYZ["vtai"]["xdcg"])*(XYZ["vtai"]["zdcg"])
    J_xz_mtip = geo["tipm"]["m"] * (XYZ["tipm"]["xdcg"])*(XYZ["tipm"]["zdcg"])
    Jxz = J_xz_wing+J_xz_htai+J_xz_vtai+J_xz_mtip

    # TODO@dpwiese - check this
    Jxz = -Jxz
    Jzx = Jxz

    J_yz_wing = 0
    J_yz_htai = 0
    J_yz_vtai = 0
    J_yz_mtip = 0
    Jyz = J_yz_wing + J_yz_htai + J_yz_vtai + J_yz_mtip
    Jzy = Jyz

    geo["totl"]["J"] = np.array([
        [JXX, Jxy, Jxz],
        [Jyx, JYY, Jyz],
        [Jzx, Jzy, JZZ]
    ])

    return { "geo": geo, "xyz": xyz, "XYZ": XYZ }

def aero_forces(geo, XYZ, xihat_b, aer):

    vel = {
        "wing": {},
        "htai": {},
        "vtai": {}
    }

    vel["totl"]     = np.array([xihat_b[0,3], xihat_b[1,3], xihat_b[2,3]])
    vel["u"]        = vel["totl"][0]
    vel["v"]        = vel["totl"][1]
    vel["w"]        = vel["totl"][2]
    vel["omega"]    = rb.inv_hat(xihat_b[0:3,0:3])
    vel["p"]        = vel["omega"][0]
    vel["q"]        = vel["omega"][1]
    vel["r"]        = vel["omega"][2]
    vel["VT"]       = la.norm(vel["totl"])
    vel["alpha"]    = np.arctan2(vel["w"], vel["u"])
    vel["beta"]     = np.arcsin(vel["v"] / vel["VT"])

    ################################################################################################
    # LONGITUDINAL DYNAMICS

    # Calculate effective uvw velocity components at wing (in body axes, NOT including surface deflections)
    vel["wing"]["totl"]     = vel["totl"] + np.cross(vel["omega"], XYZ["wing"]["rqc"].reshape(3,1), axisa=0, axisb=0).reshape(3)
    vel["wing"]["u"]        = vel["wing"]["totl"][0]
    vel["wing"]["v"]        = vel["wing"]["totl"][1]
    vel["wing"]["w"]        = vel["wing"]["totl"][2]
    vel["wing"]["VT"]       = la.norm(vel["wing"]["totl"])
    vel["wing"]["alpha"]    = np.arctan2(vel["wing"]["w"], vel["wing"]["u"])
    vel["wing"]["beta"]     = np.arcsin(vel["wing"]["v"]/vel["wing"]["VT"])

    # Calculate effective uvw velocity components at horizontal tail (in body axes, including surface deflections)
    vel["htai"]["totl"]     = vel["totl"] + np.cross(vel["omega"], XYZ["htai"]["rqc"].reshape(3,1), axisa=0, axisb=0).reshape(3)
    vel["htai"]["u"]        = vel["htai"]["totl"][0]
    vel["htai"]["v"]        = vel["htai"]["totl"][1]
    vel["htai"]["w"]        = vel["htai"]["totl"][2]
    vel["htai"]["VT"]       = la.norm(vel["htai"]["totl"])
    vel["htai"]["alpha"]    = np.arctan2(vel["htai"]["w"], vel["htai"]["u"]) + aer["delt"]["e"]
    vel["htai"]["beta"]     = np.arcsin(vel["htai"]["v"] / vel["htai"]["VT"])

    # Calculate lift coefficient and total lift force for wing
    aer["wing"]["CL"]   = (np.pi*geo["wing"]["AR"]*vel["wing"]["alpha"])/(1+math.sqrt(1+math.pow(geo["wing"]["AR"]/2, 2)))
    aer["wing"]["L"]    = 0.5*geo["dens"]["air"]*math.pow(vel["wing"]["VT"], 2)*geo["wing"]["S"]*aer["wing"]["CL"]

    # Calculate lift coefficient and total lift force for horizontal tail
    aer["htai"]["CL"]   = (np.pi*geo["htai"]["AR"]*(vel["htai"]["alpha"]))/(1+math.sqrt(1+math.pow(geo["htai"]["AR"]/2, 2)))
    aer["htai"]["L"]    = 0.5*geo["dens"]["air"]*math.pow(vel["htai"]["VT"], 2)*geo["htai"]["S"]*aer["htai"]["CL"]

    # Calculate drag coefficient and total drag force for wing
    aer["wing"]["CD"]   = math.pow(aer["wing"]["CL"], 2)/(np.pi*geo["wing"]["AR"]*aer["wing"]["e"])
    aer["wing"]["D"]    = 0.5*geo["dens"]["air"]*math.pow(vel["wing"]["VT"], 2)*geo["wing"]["S"]*aer["wing"]["CD"]

    # Calculate drag coefficient and total drag force for horizontal tail
    aer["htai"]["CD"]   = math.pow(aer["htai"]["CL"], 2)/(np.pi*geo["htai"]["AR"]*aer["htai"]["e"])
    aer["htai"]["D"]    = 0.5*geo["dens"]["air"]*math.pow(vel["htai"]["VT"], 2)*geo["htai"]["S"]*aer["htai"]["CD"]

    # Calculate X components of lift and drag for wing
    aer["wing"]["FXL"]  = aer["wing"]["L"] * np.sin(vel["alpha"])
    aer["wing"]["FXD"]  = -aer["wing"]["D"] * np.cos(vel["alpha"])

    # Calculate X components of lift and drag for horizontal tail
    aer["htai"]["FXL"]  = aer["htai"]["L"] * np.sin(vel["alpha"])
    aer["htai"]["FXD"]  = -aer["htai"]["D"] * np.cos(vel["alpha"])

    # Calculate Z components of lift and drag for wing
    aer["wing"]["FZL"]  = -aer["wing"]["L"] * np.cos(vel["alpha"])
    aer["wing"]["FZD"]  = -aer["wing"]["D"] * np.sin(vel["alpha"])

    # Calculate Z components of lift and drag for horizontal tail
    aer["htai"]["FZL"]  = -aer["htai"]["L"] * np.cos(vel["alpha"])
    aer["htai"]["FZD"]  = -aer["htai"]["D"] * np.sin(vel["alpha"])

    # Calculate moment MY about Y-axis due to wing and horizontal tail force components FZ_ and FX_
    aer["wing"]["MYL"]  = aer["wing"]["FZL"] * XYZ["wing"]["xdqc"] + aer["wing"]["FXL"] * XYZ["wing"]["zdcg"]
    aer["wing"]["MYD"]  = aer["wing"]["FZD"] * XYZ["wing"]["xdqc"] + aer["wing"]["FXD"] * XYZ["wing"]["zdcg"]
    aer["htai"]["MYL"]  = -aer["htai"]["FZL"] * XYZ["htai"]["xdqc"] + aer["htai"]["FXL"] * XYZ["htai"]["zdcg"]
    aer["htai"]["MYD"]  = -aer["htai"]["FZD"] * XYZ["htai"]["xdqc"] + aer["htai"]["FXD"] * XYZ["htai"]["zdcg"]

    ################################################################################################
    # LATERAL DYNAMICS

    # Calculate effective uvw velocity components at vertical tail (in body axes, including surface deflections)
    vel["vtai"]["totl"]     = vel["totl"] + np.cross(vel["omega"], XYZ["vtai"]["rqc"].reshape(3,1), axisa=0, axisb=0).reshape(3)
    vel["vtai"]["u"]        = vel["vtai"]["totl"][0]
    vel["vtai"]["v"]        = vel["vtai"]["totl"][1]
    vel["vtai"]["w"]        = vel["vtai"]["totl"][2]
    vel["vtai"]["VT"]       = la.norm(vel["vtai"]["totl"])
    vel["vtai"]["alpha"]    = np.arcsin(vel["vtai"]["v"]/vel["vtai"]["VT"]) - aer["delt"]["r"]

    # Calculate lift coefficient and lift force for vertical tail
    aer["vtai"]["CL"]   = (np.pi*geo["vtai"]["AR"]*vel["vtai"]["alpha"])/(1+math.sqrt(1+math.pow(geo["vtai"]["AR"]/2, 2)))
    aer["vtai"]["L"]    = 0.5 * geo["dens"]["air"] * math.pow(vel["vtai"]["VT"], 2) * geo["vtai"]["S"] * aer["vtai"]["CL"]

    # Calculate drag coefficient and drag and drag force for vertical tail
    aer["vtai"]["CD"]   = math.pow(aer["vtai"]["CL"], 2) / (np.pi*geo["vtai"]["AR"] * aer["vtai"]["e"])
    aer["vtai"]["D"]    = 0.5 * geo["dens"]["air"] * math.pow(vel["vtai"]["VT"], 2) * geo["vtai"]["S"] * aer["vtai"]["CD"]

    # Calculate X components of lift and drag for vertical tail
    aer["vtai"]["FXL"]  = aer["vtai"]["L"] * np.sin(vel["vtai"]["alpha"])
    aer["vtai"]["FXD"]  = -aer["vtai"]["D"] * np.cos(vel["vtai"]["alpha"])

    # Calculate Y components of lift and drag for vertical tail
    aer["vtai"]["FYL"]  = -aer["vtai"]["L"] * np.cos(vel["vtai"]["alpha"])
    aer["vtai"]["FYD"]  = -aer["vtai"]["D"] * np.cos(vel["vtai"]["alpha"])

    # Calculate moment about X-axis: MX
    aer["wing"]["dCL"]  = (np.pi * geo["wing"]["AR"] * aer["delt"]["a"]) / (1+math.sqrt(1+math.pow(geo["wing"]["AR"]/2, 2)))
    aer["wing"]["MX"]   = (0.5*geo["dens"]["air"]*math.pow(vel["wing"]["VT"], 2)*geo["wing"]["S"]*aer["wing"]["dCL"]*0.5*geo["wing"]["b"]) # *sign(aer["delt"]["a"])

    # Calculate moment about Z-axis: MZ
    aer["vtai"]["MZ"]   = aer["vtai"]["FYL"] * XYZ["vtai"]["xdqc"] + aer["vtai"]["FYD"] * XYZ["vtai"]["xdqc"]

    ################################################################################################
    # DAMPING

    aer["damp"]["MX"]   = -0.01 * vel["omega"][0]
    aer["damp"]["MY"]   = -0.1 * vel["omega"][1]
    aer["damp"]["MZ"]   = -0.01 * vel["omega"][2]

    ################################################################################################
    # SUM TOTALS

    aer["totl"]["FX"] = aer["wing"]["FXL"] + aer["wing"]["FXD"] + aer["htai"]["FXL"] + aer["htai"]["FXD"]
    aer["totl"]["FY"] = aer["vtai"]["FYL"] + aer["vtai"]["FYD"]
    aer["totl"]["FZ"] = aer["wing"]["FZL"] + aer["wing"]["FZD"] + aer["htai"]["FZL"] + aer["htai"]["FZD"]

    aer["totl"]["MX"] = aer["wing"]["MX"]
    aer["totl"]["MY"] = aer["wing"]["MYL"] + aer["wing"]["MYD"] + aer["htai"]["MYL"] + aer["htai"]["MYD"]
    aer["totl"]["MZ"] = aer["vtai"]["MZ"]

    # OUTPUTS
    fB   = np.array([aer["totl"]["FX"], aer["totl"]["FY"], aer["totl"]["FZ"]])
    tauB = np.array([aer["totl"]["MX"], aer["totl"]["MY"], aer["totl"]["MZ"]])

    return { "fB": fB, "tauB": tauB, "vel": vel, "aer": aer }

# def moment_equations_state(_t, x_state, u_input, _params):
#     """State is omega_b input is tau_b"""

#     # Parse input (tau_b) and state of moment equations (omega_b)
#     tau_b   = np.array([u_input[0], u_input[1], u_input[2]]).reshape(3,1)
#     omega_b = np.array([x_state[0], x_state[1], x_state[2]]).reshape(3,1)

#     # Calculate and return omega_b_dot
#     return inv(J_B) @ ((-rb.hat(omega_b) @ J_B) @ omega_b + tau_b)

# def moment_equations_output(_t, x_state, u_input, _params):
#     """Output is omega_b_dot"""

#     # Return omega_b_dot
#     return x_state

# def orientation_equations_state(_t, x_state, u_input, _params):
#     """State is euler angles (phi, theta, psi) input is omega_b"""

#     # Parse input
#     phi     = x_state[0]
#     theta   = x_state[1]

#     # Build output
#     row_1 = np.array([1, np.tan(theta) * np.sin(phi), np.tan(theta) * np.cos(phi)])
#     row_2 = np.array([0, np.cos(theta), -np.sin(phi)])
#     row_3 = np.array([0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)])

#     # Calculate and return euler_dot
#     return np.array([row_1, row_2, row_3]) @ u_input.reshape(3,1)

# def orientation_equations_output(_t, x_state, u_input, _params):
#     """Output is time derivative of Euler angles (phi_dot, theta_dot, psi_dot)"""

#     # Return Euler time derivatives
#     return x_state

# def velocity_equations_state(_t, x_state, u_input, _params):
#     """State is v_b, input is f_b, quat, omega_b"""

#     # Parse input (f_b, quat, omega_b)
#     # Quaternion is in scalar-last format
#     f_b   = np.array([u_input[0], u_input[1], u_input[2]]).reshape(3,1)
#     quat   = np.array([u_input[3], u_input[4], u_input[5], u_input[6]]).reshape(4,1)
#     omega_b   = np.array([u_input[7], u_input[8], u_input[9]]).reshape(3,1)

#     # Parse state of moment equations (v_b)
#     v_b = np.array([x_state[0], x_state[1], x_state[2]]).reshape(3,1)

#     # TODO@dpwiese
#     gbar = np.array([0, 0, 9.8]).reshape(3,1)
#     mass = 1

#     # Calculate and return v_b_dot
#     return -rb.hat(omega_b) @ v_b + rb.quat_to_rotation_matrix(quat) @ gbar + f_b / mass

# def velocity_equations_output(_t, x_state, u_input, _params):
#     """Output is v_b_dot"""

#     # Return v_b_dot
#     return x_state

# def position_equations_state(_t, x_state, u_input, _params):
#     """State is position Delta (x, y, z) input is quat, v_b"""

#     # Parse input
#     # Quaternion is in scalar-last format
#     quat    = np.array([u_input[0], u_input[1], u_input[2], u_input[3]]).reshape(4,1)
#     v_b     = np.array([u_input[4], u_input[5], u_input[6]]).reshape(3,1)

#     # Calculate and return Delta_dot
#     return rb.quat_to_rotation_matrix(quat) @ v_b

# def position_equations_output(_t, x_state, u_input, _params):
#     """Output is time derivative of position Delta_dot (x_dot, y_dot, z_dot)"""

#     # Return position time derivatives
#     return x_state

# # System parameters: moment equations
# # Input: tau_b
# # State: omega_b
# # Output: omega_b
# IO_MOMENT_EQUATIONS = control.NonlinearIOSystem(
#     moment_equations_state,
#     moment_equations_output,
#     inputs=3,
#     outputs=3,
#     states=3,
#     name='moment',
#     dt=0
# )

# # System parameters: orientation equations
# # Input: omega_b
# # State: euler
# # Output: euler
# IO_ORIENTATION_EQUATIONS = control.NonlinearIOSystem(
#     orientation_equations_state,
#     orientation_equations_output,
#     inputs=3,
#     outputs=3,
#     states=3,
#     name='orientation',
#     dt=0
# )

# # System parameters: velocity equations
# # Input: f_b, quat, omega_b
# # State: v_b
# # Output: v_b
# IO_VELOCITY_EQUATIONS = control.NonlinearIOSystem(
#     velocity_equations_state,
#     velocity_equations_output,
#     inputs=10,
#     outputs=3,
#     states=3,
#     name='velocity',
#     dt=0
# )

# # System parameters: position equations
# # Input: quat, v_b
# # State: position Delta (x, y, z)
# # Output: position Delta (x, y, z)
# IO_POSITION_EQUATIONS = control.NonlinearIOSystem(
#     position_equations_state,
#     position_equations_output,
#     inputs=7,
#     outputs=3,
#     states=3,
#     name='position',
#     dt=0
# )

# # System parameters: combined open-loop system
# # Input: tau_b, f_b
# # State: [omega_b, euler, v_b, Delta]
# # Output: [omega_b, euler, v_b, Delta]
# IO_OPEN_LOOP = control.InterconnectedSystem(
#     (IO_MOMENT_EQUATIONS, IO_ORIENTATION_EQUATIONS, IO_VELOCITY_EQUATIONS, IO_POSITION_EQUATIONS),
#     connections=(
#         ('orientation.u[0]', 'moment.y[0]'),
#         ('orientation.u[1]', 'moment.y[1]'),
#         ('orientation.u[2]', 'moment.y[2]')
#     ),
#     inplist=('moment.u[0]', 'moment.u[1]', 'moment.u[2]', 'velocity.u[0]', 'velocity.u[1]', 'velocity.u[2]'),
#     outlist=('moment.y[0]', 'moment.y[1]', 'moment.y[2]', 'orientation.y[0]', 'orientation.y[1]', 'orientation.y[2]', 'velocity.y[0]', 'velocity.y[1]', 'velocity.y[2]', 'position.y[0]', 'position.y[1]', 'position.y[2]'),
#     dt=0
# )

# Input surface dimensions (span and chords), masses, densities, gravity, unit conversions
geo = {
    "wing": {
        "b": 0.45,      # m (wingspan)
        "c": 0.075      # m (wing chord)
    },
    "htai": {
        "b": 0.18,      # m (horizontal tail span)
        "c": 0.04       # m (horitontal tail chord)
    },
    "vtai": {
        "h": 0.09,      # m (vertical tail height)
        "c": 0.04       # m (vertical tail chord)
    },
    "tipm": {
        "m": 0.010      # kg (weight at tip)
    },
    "dens": {
        "bal": 0.38,    # kg/m^2 (density of balsa per unit area)
        "air": 1.204    # kg/m^3 (density of air) 1.1644
    },
    "totl": {}
}

# Calculate surface areas and aspect ratios
geo["wing"]["S"]  = geo["wing"]["b"] * geo["wing"]["c"]               # m^2 (wing planform area)
geo["htai"]["S"]  = geo["htai"]["b"] * geo["htai"]["c"]               # m^2 (horizontail tail area)
geo["vtai"]["S"]  = geo["vtai"]["h"] * geo["vtai"]["c"]               # m^2 (vertical tail area)
geo["wing"]["AR"] = math.pow(geo["wing"]["b"], 2) / geo["wing"]["S"]  # dimensionless (Aspect ratio of wing)
geo["htai"]["AR"] = math.pow(geo["htai"]["b"], 2) / geo["htai"]["S"]  # dimensionless (Aspect ratio of horizontal tail)
geo["vtai"]["AR"] = math.pow(geo["vtai"]["h"], 2) / geo["vtai"]["S"]  # dimensionless (Aspect ratio of horizontal tail)

gbar            = np.array([0, 0, 9.8])   # kg-m/s^2 gravity in inertial axes: defined z_{I} positive down

# Input distances in xyz coordinates
xyz = {
    "wing": {
        # m (distance from nose to LE of wing)
        "xdle": -0.025
    },
    "htai": {
        # m (distance from nose to LE of horizontal tail)
        "xdle": -0.23
    },
    "vtai": {
        # m (distance from nose to LE of vertical tail)
        "xdle": -0.23
    },
    "tipm": {
        "xd": 0
    },
    "cg": {}
}

# Input aerodynamic efficiencies
aer = {
    "wing": {
        # dimensionless (oswald efficiency of wing)
        "e": 0.9
    },
    "htai": {
        # dimensionless (oswald efficiency of horizontal tail)
        "e": 0.9
    },
    "vtai": {
        # dimensionless (oswald efficiency of vertical tail)
        "e": 0.9
    },
    "delt": {},
    "damp": {},
    "totl": {}
}

# Calculate mass properties of glider
mass_prop_out = mass_prop(geo, xyz)

geo = mass_prop_out["geo"]
xyz = mass_prop_out["xyz"]
XYZ = mass_prop_out["XYZ"]

####################################################################################################

deltae_deg = -0.035
deltaa_deg = 0
deltar_deg = 0

# Control surface deflections in radians
aer["delt"]["e"] = math.radians(deltae_deg)
aer["delt"]["a"] = math.radians(deltaa_deg)
aer["delt"]["r"] = math.radians(deltar_deg)

# ####################################################################################################
# # Control Design

# # Vehicle moment of inertia
# J_B = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 5]])

# # Equilibrium point about which to linearize
# R_EQ = np.eye(3, dtype=float)
# X_EQ = np.zeros(6)
# U_EQ = np.zeros(3)

# # Rotation matrix that describes rotation matrix from inertial frame to desired orientation
# R_D = R_EQ

# # Linearize nonlinear spacecraft dynamics
# linsys = control.linearize(IO_OPEN_LOOP, X_EQ, U_EQ)

# # Make LQR controller
# Q = np.eye(6, dtype=float)
# R = np.eye(3, dtype=float)
# K, S, E = control.lqr(linsys, Q, R)

# # Controller
# # input: x (vehicle state)
# # state: nothing, but TODO@dpwiese - I don't know yet if I can have zero state?
# # output: -K * x
# A_C = np.zeros((1,1))
# B_C = np.zeros((1,6))
# C_C = np.zeros((3,1))
# D_C = -K

# # Define controller
# IO_LINEAR_CONTROL = control.LinearIOSystem(
#     control.StateSpace(A_C, B_C, C_C, D_C),
#     inputs=6,
#     outputs=3,
#     states=1,
#     name='linear_control'
# )

# def nonlinear_controller_state(_t, x_state, u_input, _params):
#     """Controller doesn't have internal state, input is plant state"""

#     # Calculate and return the controller state, it won't be used anyway
#     return x_state

# def nonlinear_controller_output(_t, x_state, u_input, _params):
#     """Input is the plant state, controller doesn't have state, output is tau_b"""

#     # Parse input to get omega
#     omega = np.array([u_input[0], u_input[1], u_input[2]]).reshape(3,1)

#     # pylint: disable-next=C0301
#     rot_mat = rb.euler_to_rotation_matrix({"phi": u_input[3], "theta": u_input[4], "psi": u_input[5]})
#     omega_d = np.zeros((3,1))
#     R_D = np.eye(3, dtype=float)

#     # Get velocity and proportional parts of gain
#     K_V = K[:, :3]
#     K_P = K[:, 3:]

#     term_1 = -np.transpose(rb.inv_hat(K_P @ np.transpose(R_D) @ rot_mat))
#     term_2 = -K_V @ (omega - rot_mat @ np.transpose(R_D) @ omega_d)

#     # pylint: disable-next=C0301
#     tau_b = term_1.reshape(3,1) + term_2

#     # Return control
#     return tau_b

# # System parameters: moment equations
# # Input: tau_b
# # State: omega_b
# # Output: omega_b
# IO_NONLINEAR_CONTROL = control.NonlinearIOSystem(
#     nonlinear_controller_state,
#     nonlinear_controller_output,
#     inputs=6,
#     outputs=3,
#     states=1,
#     name='nonlinear_control',
#     dt=0
# )

# # System parameters: combined open-loop system
# # Input: None
# # State: [omega_b, euler]
# # Output: [omega_b, euler]
# IO_CLOSED_LOOP_LINEAR = control.InterconnectedSystem(
#     (IO_MOMENT_EQUATIONS, IO_ORIENTATION_EQUATIONS, IO_LINEAR_CONTROL),
#     connections=(
#         ('orientation.u[0]',          'moment.y[0]'),
#         ('orientation.u[1]',          'moment.y[1]'),
#         ('orientation.u[2]',          'moment.y[2]'),
#         ('moment.u[0]',         'linear_control.y[0]'),
#         ('moment.u[1]',         'linear_control.y[1]'),
#         ('moment.u[2]',         'linear_control.y[2]'),
#         ('linear_control.u[0]', 'moment.y[0]'),
#         ('linear_control.u[1]', 'moment.y[1]'),
#         ('linear_control.u[2]', 'moment.y[2]'),
#         ('linear_control.u[3]', 'orientation.y[0]'),
#         ('linear_control.u[4]', 'orientation.y[1]'),
#         ('linear_control.u[5]', 'orientation.y[2]'),
#     ),
#     inplist=(),
#     outlist=('moment.y[0]', 'moment.y[1]', 'moment.y[2]', 'orientation.y[0]', 'orientation.y[1]', 'orientation.y[2]'),
#     dt=0
# )

# # System parameters: combined open-loop system
# # Input: None
# # State: [omega_b, euler]
# # Output: [omega_b, euler]
# IO_CLOSED_LOOP_NONLINEAR = control.InterconnectedSystem(
#     (IO_MOMENT_EQUATIONS, IO_ORIENTATION_EQUATIONS, IO_NONLINEAR_CONTROL),
#     connections=(
#         ('orientation.u[0]',              'moment.y[0]'),
#         ('orientation.u[1]',              'moment.y[1]'),
#         ('orientation.u[2]',              'moment.y[2]'),
#         ('moment.u[0]',             'nonlinear_control.y[0]'),
#         ('moment.u[1]',             'nonlinear_control.y[1]'),
#         ('moment.u[2]',             'nonlinear_control.y[2]'),
#         ('nonlinear_control.u[0]',  'moment.y[0]'),
#         ('nonlinear_control.u[1]',  'moment.y[1]'),
#         ('nonlinear_control.u[2]',  'moment.y[2]'),
#         ('nonlinear_control.u[3]',  'orientation.y[0]'),
#         ('nonlinear_control.u[4]',  'orientation.y[1]'),
#         ('nonlinear_control.u[5]',  'orientation.y[2]'),
#     ),
#     inplist=(),
#     outlist=('moment.y[0]', 'moment.y[1]', 'moment.y[2]', 'orientation.y[0]', 'orientation.y[1]', 'orientation.y[2]'),
#     dt=0
# )

def eom_integrator(geo, xihatB, tauB, fB, gbar, g, dt):

    # Convert input xihat into omega and v
    omegaB  = rb.inv_hat(xihatB[0:3, 0:3])
    v_B     = xihatB[0:3,3]
    RIB     = g[0:3,0:3]
    JB      = geo["totl"]["J"]
    M       = geo["totl"]["m"]

    v_B = v_B.reshape(3,1)
    gbar = gbar.reshape(3,1)
    tauB = tauB.reshape(3,1)
    fB = fB.reshape(3,1)

    assert omegaB.shape == (3,1)
    assert gbar.shape == (3,1)
    assert tauB.shape == (3,1)
    assert fB.shape == (3,1)

    # Calculate all the next time steps based on the previous
    # Omega
    K1 = dt * inv(JB) @ (np.cross(-omegaB, JB @ omegaB, axisa=0, axisb=0).reshape(3,1)+tauB)
    K2 = dt * inv(JB) @ (np.cross(-(omegaB+0.5*K1), JB @ (omegaB+0.5*K1), axisa=0, axisb=0).reshape(3,1)+tauB)
    K3 = dt * inv(JB) @ (np.cross(-(omegaB+0.5*K2), JB @ (omegaB+0.5*K2), axisa=0, axisb=0).reshape(3,1)+tauB)
    K4 = dt * inv(JB) @ (np.cross(-(omegaB+K3), JB @ (omegaB+K3), axisa=0, axisb=0).reshape(3,1)+tauB)
    omegaB = omegaB + (1/6)*(K1 + 2*K2 + 2*K3 + K4)

    # Velocity
    K1 = dt * (np.cross(-omegaB, (v_B), axisa=0, axisb=0).reshape(3,1) + np.transpose(RIB) @ gbar+(fB/M))
    K2 = dt * (np.cross(-omegaB, (v_B+0.5*K1), axisa=0, axisb=0).reshape(3,1) + np.transpose(RIB) @ gbar+(fB/M))
    K3 = dt * (np.cross(-omegaB, (v_B+0.5*K2), axisa=0, axisb=0).reshape(3,1) + np.transpose(RIB) @ gbar+(fB/M))
    K4 = dt * (np.cross(-omegaB, (v_B+K3), axisa=0, axisb=0).reshape(3,1) + np.transpose(RIB) @ gbar+(fB/M))
    v_B = v_B + (1/6) * (K1 + 2*K2 + 2*K3 + K4)

    # Configuration: g
    K1 = dt * g @ xihatB
    K2 = dt * (g+0.5*K1) @ xihatB
    K3 = dt * (g+0.5*K2) @ xihatB
    K4 = dt * (g+K3) @ xihatB
    g = g + (1/6) * (K1 + 2*K2 + 2*K3 + K4)

    xihatB = np.append(
        np.append(rb.hat(omegaB), v_B, axis=1),
        np.zeros((1,4)),
        axis=0
    )

    return { "g": g, "xihatB": xihatB }

####################################################################################################
# INTEGRATE

# Simulation times
t0      = 0
tf      = 100
dt      = 0.001
nt      = int((tf-t0)/dt+1)
time    = np.linspace(t0, tf, nt)
n_int   = int(((tf-t0)/dt)+1)

# Initialize integrator
g0 = np.eye(4, dtype=float)
xihatB0 = np.array([
    [1, 0, 0, 11],
    [0, 1, 0, 0],
    [0, 0, 1, 0.3],
    [0, 0, 0, 0]
])
g = g0
xihatB = xihatB0

Deltao = np.array([])
vbo = np.array([])

# Run integrator
for i in range(1, n_int-1):
    aero_forces_out = aero_forces(geo, XYZ, xihatB, aer)
    fB      = aero_forces_out["fB"]
    tauB    = aero_forces_out["tauB"]
    vel     = aero_forces_out["vel"]
    aer     = aero_forces_out["aer"]

    eom_integrator_output = eom_integrator(geo, xihatB, tauB, fB, gbar, g, dt)

    g = eom_integrator_output["g"]
    xihatB = eom_integrator_output["xihatB"]

    # Save the output in a format for plotting
    Deltao_append = np.array([np.array([g[0,3], g[1,3], g[2,3]])])
    vbo_append = np.array([np.array([xihatB[0,3], xihatB[1,3], xihatB[2,3]])])

    if Deltao.size == 0:
        Deltao = Deltao_append
    else:
        Deltao = np.append(Deltao, Deltao_append, axis=0)

    if vbo.size == 0:
        vbo = vbo_append
    else:
        vbo = np.append(vbo, vbo_append, axis=0)

    # Deltao[0,i] = g[0,3]
    # Deltao[1,i] = g[1,3]
    # Deltao[2,i] = g[2,3]
    # R           = g[0:3, 0:3]
    # eulr[:,i]   = rb.rotation_matrix_to_euler(R)
    # vely[0,i]   = xihatB[0,3]
    # vely[1,i]   = xihatB[1,3]
    # vely[2,i]   = xihatB[2,3]
    # OMEGA       = rb.hat(xihatB[1:3,1:3])
    # omega[0,i]  = OMEGA[0]
    # omega[1,i]  = OMEGA[1]
    # omega[2,i]  = OMEGA[2]

    # Bank angle hold for descending spiral
    # aer["delt"]["a"] = (math.radians(60)-eulr[0,i])*0.01
    # vel["delt"]["e"]=vel["delt"]["e"]+((math.radians(0.0032)-eulr(2,i))*-0.1)

    # Pitch stabilizer for straight descent
    # vel["delt"]["e"]=vel["delt"]["e"]+((math.radians(pitchdes)-eulr(2,i))*-0.001)+((0-omega(2,i))*-0.001)

    # Lower pitch controller gains
    # vel["delt"]["e"]=vel["delt"]["e"]+((math.radians(pitchdes)-eulr(2,i))*-0.001)+((0-omega(2,i))*-0.001)

####################################################################################################

# print(Deltao[:,1])

RED = '#f62d73'
BLUE = '#1269d3'
WHITE = '#ffffff'
GREEN = '#2df643'
BLACK = '#000000'

FIG = plt.figure(1, figsize=(6, 6), dpi=100, facecolor='w', edgecolor='k')

AX_1 = FIG.add_subplot(1, 2, 1)
AX_1.plot(Deltao[:,0], label=r'$x$', color=BLACK)
AX_1.plot(Deltao[:,1], label=r'$y$', color=BLUE)
AX_1.plot(Deltao[:,2], label=r'$z$', color=RED)
AX_1.legend(loc="lower right", bbox_to_anchor=(1, 0), fontsize=9)
AX_1.set_facecolor(WHITE)

AX_2 = FIG.add_subplot(1, 2, 2)
AX_2.plot(vbo[:,0], label=r'$v_{x}$', color=BLACK)
AX_2.plot(vbo[:,1], label=r'$v_{y}$', color=BLUE)
AX_2.plot(vbo[:,2], label=r'$v_{z}$', color=RED)
AX_2.legend(loc="lower right", bbox_to_anchor=(1, 0), fontsize=9)
AX_2.set_facecolor(WHITE)

plt.show()

# # Set simulation duration and time steps
# N_POINTS = 1000
# T_F = 15

# # Set initial conditions
# X0 = np.zeros((6, 1))
# X0 = np.array([1.2, -0.5, 0, 0, 0, np.pi/6, 0])

# # Define simulation time span and control input
# T = np.linspace(0, T_F, N_POINTS)

# # There is no input to the closed-loop system
# U = np.zeros((0, N_POINTS))

# # Simulate the system
# T_OUT_LIN, Y_OUT_LIN = control.input_output_response(IO_CLOSED_LOOP_LINEAR, T, U, X0)
# T_OUT_NON, Y_OUT_NON = control.input_output_response(IO_CLOSED_LOOP_NONLINEAR, T, U, X0)

# # Plot the response
# plt.rc('text', usetex=True)
# plt.rc('font', family='sans')

# # FIG = plt.figure(1, figsize=(6, 6), dpi=300, facecolor='w', edgecolor='k')
# FIG = plt.figure(1, figsize=(6, 6), dpi=100, facecolor='w', edgecolor='k')

# AX_1 = FIG.add_subplot(1, 1, 1)
# AX_1.plot(T_OUT_LIN, Y_OUT_LIN[3], label=r'$\phi$ (linear)', color=BLACK, linestyle='dashed')
# AX_1.plot(T_OUT_LIN, Y_OUT_LIN[4], label=r'$\theta$ (linear)', color=BLUE, linestyle='dashed')
# AX_1.plot(T_OUT_LIN, Y_OUT_LIN[5], label=r'$\psi$ (linear)', color=RED, linestyle='dashed')
# AX_1.plot(T_OUT_NON, Y_OUT_NON[3], label=r'$\phi$ (nonlinear)', color=BLACK)
# AX_1.plot(T_OUT_NON, Y_OUT_NON[4], label=r'$\theta$ (nonlinear)', color=BLUE)
# AX_1.plot(T_OUT_NON, Y_OUT_NON[5], label=r'$\psi$ (nonlinear)', color=RED)
# AX_1.set_xlabel(r'time ($t$)', fontname="Times New Roman", fontsize=9, fontweight=100)
# AX_1.legend(loc="lower right", bbox_to_anchor=(1, 0), fontsize=9)
# AX_1.set_facecolor(WHITE)

# plt.show()

# class AirplaneGeometry:
#   x = 5

# class DotNotation(dict):
#     __getattr__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__

# geo = DotNotation(acgeo)
