"""
Kinematics
"""

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from hat import (
    hat,
    twist_coords_to_g,
    rotation_matrix_to_euler
)

def run_kinematics_analytic():
    """Move initial conditions through four changes in position and orientation"""

    # Times for each segment
    t0 = 0
    t1 = t0 + 2
    t2 = t1 + (3*np.pi)/2
    t3 = t2 + 2
    t4 = t3 + (3*np.pi)/2

    # Linear and angular velocity: segment 1
    omega_B1    = np.array([0, 0, 0]).reshape(3,1)
    v_B1        = np.array([1, 0, 0.2]).reshape(3,1)

    # Linear and angular velocity: segment 2
    omega_B2    = np.array([0, 0, 1]).reshape(3,1)
    v_B2        = np.array([1, 0, 0]).reshape(3,1)

    # Linear and angular velocity: segment 3
    omega_B3    = np.array([0, 0, 0]).reshape(3,1)
    v_B3        = np.array([1, 0, -0.2]).reshape(3,1)

    # Linear and angular velocity: segment 4
    omega_B4    = np.array([0, 0, -1]).reshape(3,1)
    v_B4        = np.array([1, 0, 0]).reshape(3,1)

    # Initial state
    p0I = np.eye(4, dtype=float)

    # First twist
    g1  = twist_coords_to_g({"omega": omega_B1, "velocity": v_B1, "theta": t1});
    p1I = g1 @ p0I

    # Second twist
    g2  = twist_coords_to_g({"omega": omega_B2, "velocity": v_B2, "theta": t2 - t1});
    p2I = g2 @ p1I

    # Third twist
    g3  = twist_coords_to_g({"omega": omega_B3, "velocity": v_B3, "theta": t3 - t2});
    p3I = g3 @ p2I

    # Fourth twist
    g4  = twist_coords_to_g({"omega": omega_B4, "velocity": v_B4, "theta": t4 - t3});
    p4I = g4 @ p3I

    # Print and check that final state matches initial state
    print(p0I)
    print(p4I)

def kinematics_body(t0, tf, omega_B, v_B, dt, RIB, Delta):
    """Kinematics"""

    # Make sure shapes of input are correct
    assert omega_B.shape == (3,1)
    assert v_B.shape == (3,1)
    assert RIB[-1].shape == (3,3)
    assert Delta[-1].shape == (3,1)

    # Initial conditions
    n_int = int(((tf-t0)/dt)+1)

    # Assemble initial g matrix
    top_row = np.append(RIB[-1], Delta[-1], axis=1)
    bottom_row = np.zeros((1,4))
    g_matrix = np.append(top_row, bottom_row, axis=0)
    g_matrix = np.array([g_matrix])

    # Calculate xihat
    xihatB_top_row = np.append(hat(omega_B), v_B, axis=1)
    xihatB_bottom_row = np.zeros((1,4))
    xihatB = np.append(xihatB_top_row, xihatB_bottom_row, axis=0)

    # 4th Order Runge-Kutta
    for i in range(0, n_int-1):
        K1 = dt * g_matrix[i] @ xihatB
        K2 = dt * (g_matrix[i]+0.5*K1) @ xihatB
        K3 = dt * (g_matrix[i]+0.5*K2) @ xihatB
        K4 = dt * (g_matrix[i]+K3) @ xihatB

        # Append g matrix with increment for each timestep
        g_matrix_append = np.array([g_matrix[i] + (1/6) * (K1 + 2*K2 + 2*K3 + K4)])
        g_matrix = np.append(g_matrix, g_matrix_append, axis=0)

    # Return g matrix, R_IB matrix, and Delta matrix
    return (
        g_matrix,
        g_matrix[:, :3, :3],
        g_matrix[:, :3, 3].reshape(-1,3,1))

def run_kinematics_body():
    """Kinematics"""

    # Times for each segment
    t0 = 0
    t1 = t0 + 2
    t2 = t1 + (3*np.pi)/2
    t3 = t2 + 2
    t4 = t3 + (3*np.pi)/2

    # Stuff
    dt = 0.01
    RIB0 = np.array([np.eye(3,3)])
    Delta0 = np.array([np.zeros((3,1))])
    # Delta0 = np.zeros((1,3,1))

    # Linear and angular velocity: segment 1
    omega_B1    = np.array([0, 0, 0]).reshape(3,1)
    v_B1        = np.array([1, 0, 0.2]).reshape(3,1)

    # Linear and angular velocity: segment 2
    omega_B2    = np.array([0, 0, 1]).reshape(3,1)
    v_B2        = np.array([1, 0, 0]).reshape(3,1)

    # Linear and angular velocity: segment 3
    omega_B3    = np.array([0, 0, 0]).reshape(3,1)
    v_B3        = np.array([1, 0, -0.2]).reshape(3,1)

    # Linear and angular velocity: segment 4
    omega_B4    = np.array([0, 0, -1]).reshape(3,1)
    v_B4        = np.array([1, 0, 0]).reshape(3,1)

    print(Delta0[-1])

    (g_out1, RIB_out1, Delta_out1) = kinematics_body(t0, t1, omega_B1, v_B1, dt, RIB0, Delta0)

    print(Delta_out1[-1])

    ree1 = np.array([RIB_out1[-1]])
    dee1 = np.array([Delta_out1[-1]])

    (g_out2, RIB_out2, Delta_out2) = kinematics_body(t1, t2, omega_B2, v_B2, dt, ree1, dee1)

    print(Delta_out2[-1])

    ree2 = np.array([RIB_out2[-1]])
    dee2 = np.array([Delta_out2[-1]])

    (g_out3, RIB_out3, Delta_out3) = kinematics_body(t2, t3, omega_B3, v_B3, dt, ree2, dee2)

    print(Delta_out3[-1])

    ree3 = np.array([RIB_out3[-1]])
    dee3 = np.array([Delta_out3[-1]])

    (g_out4, RIB_out4, Delta_out4) = kinematics_body(t3, t4, omega_B4, v_B4, dt, ree3, dee3)

    print(Delta_out4[-1])

def wrenchint(t0, tf, dt, J_B, M, omega_B0, v_B0, tau_B0, f_B0, RIB0, gbar, DeltaIB0):
    """Wrench"""

    # Make sure shapes of input are correct
    assert J_B.shape == (3,3)
    assert omega_B0.shape == (3,1)
    assert v_B0.shape == (3,1)
    assert tau_B0.shape == (3,1)
    assert f_B0.shape == (3,1)
    assert RIB0.shape == (3,3)
    assert DeltaIB0.shape == (3,1)

    # Step size
    n_int = int(((tf-t0)/dt)+1)

    # Initial velocity (translational and rotational)
    omega_B = np.array([omega_B0])
    v_B = np.array([v_B0])

    # Initial position and orientation
    R = np.array([RIB0])
    Delta = np.array([DeltaIB0])
    tau_B = tau_B0
    f_B = f_B0

    # Dissipative consant
    # k_diss = 0.0
    k_diss = 0.05

    # 4th Order Runge-Kutta
    for i in range(0, n_int-1):
        K1 = dt * inv(J_B) @ (np.cross(-omega_B[i], J_B @ omega_B[i], axisa=0, axisb=0).reshape(3,1) + tau_B)
        K2 = dt * inv(J_B) @ (np.cross(-(omega_B[i]+0.5*K1),J_B@(omega_B[i]+0.5*K1), axisa=0, axisb=0).reshape(3,1)+tau_B)
        K3 = dt * inv(J_B) @ (np.cross(-(omega_B[i]+0.5*K2),J_B@(omega_B[i]+0.5*K2), axisa=0, axisb=0).reshape(3,1)+tau_B)
        K4 = dt * inv(J_B) @ (np.cross(-(omega_B[i]+K3),J_B@(omega_B[i]+K3), axisa=0, axisb=0).reshape(3,1)+tau_B)

        # Append omega with increment for each timestep
        omega_B_append = np.array([omega_B[i] + (1/6) * (K1 + 2*K2 + 2*K3 + K4)])
        omega_B = np.append(omega_B, omega_B_append, axis=0)

        K1 = dt * (np.cross(-omega_B[i],(v_B[i]), axisa=0, axisb=0).reshape(3,1)+np.transpose(R[i]) @ gbar+(f_B/M))
        K2 = dt * (np.cross(-omega_B[i],(v_B[i]+0.5*K1), axisa=0, axisb=0).reshape(3,1)+np.transpose(R[i]) @ gbar+(f_B/M))
        K3 = dt * (np.cross(-omega_B[i],(v_B[i]+0.5*K2), axisa=0, axisb=0).reshape(3,1)+np.transpose(R[i]) @ gbar+(f_B/M))
        K4 = dt * (np.cross(-omega_B[i],(v_B[i]+K3), axisa=0, axisb=0).reshape(3,1)+np.transpose(R[i]) @ gbar+(f_B/M))

        # Append v_B with increment for each timestep
        v_B_append = np.array([v_B[i] + (1/6) * (K1 + 2*K2 + 2*K3 + K4)])
        v_B = np.append(v_B, v_B_append, axis=0)

        K1 = dt * (R[i]) @ hat(omega_B[i])
        K2 = dt * (R[i]+0.5*K1) @ hat(omega_B[i])
        K3 = dt * (R[i]+0.5*K2) @ hat(omega_B[i])
        K4 = dt * (R[i]+K3) @ hat(omega_B[i])

        # Append R with increment for each timestep
        R_append = np.array([R[i] + (1/6) * (K1 + 2*K2 + 2*K3 + K4)])
        R = np.append(R, R_append, axis=0)

        K1 = dt * R[i] @ (v_B[i])
        K2 = dt * R[i] @ (v_B[i]+0.5*K1)
        K3 = dt * R[i] @ (v_B[i]+0.5*K1)
        K4 = dt * R[i] @ (v_B[i]+K3)

        # Append Delta with increment for each timestep
        Delta_append = np.array([Delta[i] + (1/6) * (K1 + 2*K2 + 2*K3 + K4)])
        Delta = np.append(Delta, Delta_append, axis=0)

        tau_B = -k_diss * omega_B[i];

    return (v_B, Delta, R)

def run_wrenchint():
    """Wrench"""

    # Simulation times
    t0 = 0
    tf = 20
    dt = 0.01
    nt = int((tf-t0)/dt+1)
    time = np.linspace(t0, tf, nt)

    # Constants
    # J_B = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 2]])
    J_B = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 5]])
    M = 1
    gbar = np.array([0, 0, 0]).reshape(3,1)

    # Initial velocity (translational and rotational)
    omega_B0 = np.array([1, 0, 0.2]).reshape(3,1)
    v_B0 = np.array([0, 0, 0]).reshape(3,1)

    # Wrench acting on rigid body
    tau_B0 = np.array([0, 0, 0]).reshape(3,1)
    f_B0 = np.array([0, 0, 0]).reshape(3,1)

    # Initial position and orientation of rigid body
    RIB0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    DeltaIB0 = np.array([0, 0, 0]).reshape(3,1)

    # Run integrator
    [v_Bout, Delta_out, R_out] = wrenchint(t0, tf, dt, J_B, M,omega_B0, v_B0, tau_B0, f_B0, RIB0, gbar, DeltaIB0)

    # Convert to Euler angles
    euler_angles = []
    for i in range(0, len(R_out)-1):
        euler_angles = np.append(euler_angles, rotation_matrix_to_euler(R_out[i]))

    t_plot = time[0:len(time)-1].reshape(-1)

    psi = np.array([eulr['psi'] for eulr in euler_angles])
    theta = np.array([eulr['theta'] for eulr in euler_angles])
    phi = np.array([eulr['phi'] for eulr in euler_angles])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(t_plot, psi.reshape(-1), label="Yaw (psi)")
    ax.plot(t_plot, theta.reshape(-1), label="Pitch (theta)")
    ax.plot(t_plot, phi.reshape(-1), label="Roll (phi)")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    # run_kinematics_analytic()
    # run_kinematics_body()
    run_wrenchint()
