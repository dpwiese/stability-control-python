"""
Kinematics
"""

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import rigidbody as rb

def run_kinematics_analytic():
    """Move initial conditions through four changes in position and orientation"""

    # Times for each segment
    t_0 = 0
    t_1 = t_0 + 2
    t_2 = t_1 + (3*np.pi)/2
    t_3 = t_2 + 2
    t_4 = t_3 + (3*np.pi)/2

    # Linear and angular velocity: segment 1
    omega_b_1   = np.array([0, 0, 0]).reshape(3,1)
    v_b_1       = np.array([1, 0, 0.2]).reshape(3,1)

    # Linear and angular velocity: segment 2
    omega_b_2   = np.array([0, 0, 1]).reshape(3,1)
    v_b_2       = np.array([1, 0, 0]).reshape(3,1)

    # Linear and angular velocity: segment 3
    omega_b_3   = np.array([0, 0, 0]).reshape(3,1)
    v_b_3        = np.array([1, 0, -0.2]).reshape(3,1)

    # Linear and angular velocity: segment 4
    omega_b_4   = np.array([0, 0, -1]).reshape(3,1)
    v_b_4        = np.array([1, 0, 0]).reshape(3,1)

    # Initial state
    p_0_i = np.eye(4, dtype=float)

    # First twist
    g_1  = rb.twist_coords_to_config_g({"omega": omega_b_1, "velocity": v_b_1, "theta": t_1})
    p_1_i = g_1 @ p_0_i

    # Second twist
    g_2  = rb.twist_coords_to_config_g({"omega": omega_b_2, "velocity": v_b_2, "theta": t_2 - t_1})
    p_2_i = g_2 @ p_1_i

    # Third twist
    g_3  = rb.twist_coords_to_config_g({"omega": omega_b_3, "velocity": v_b_3, "theta": t_3 - t_2})
    p_3_i = g_3 @ p_2_i

    # Fourth twist
    g_4  = rb.twist_coords_to_config_g({"omega": omega_b_4, "velocity": v_b_4, "theta": t_4 - t_3})
    p_4_i = g_4 @ p_3_i

    # Print and check that final state matches initial state
    print(p_0_i)
    print(p_4_i)

    # TODO@dpwiese - return stuff here to plot

def kinematics_body(t_0, t_f, omega_b, v_b, dt, rot_i_b, delta):
    """Kinematics"""

    # Make sure shapes of input are correct
    assert omega_b.shape == (3,1)
    assert v_b.shape == (3,1)
    assert rot_i_b[-1].shape == (3,3)
    assert delta[-1].shape == (3,1)

    # Initial conditions
    n_int = int(((t_f-t_0)/dt)+1)

    # Assemble initial g matrix
    top_row = np.append(rot_i_b[-1], delta[-1], axis=1)
    bottom_row = np.zeros((1,4))
    g_matrix = np.append(top_row, bottom_row, axis=0)
    g_matrix = np.array([g_matrix])

    # Calculate xihat
    xihat_b_top_row = np.append(rb.hat(omega_b), v_b, axis=1)
    xihat_b_bottom_row = np.zeros((1,4))
    xihat_b = np.append(xihat_b_top_row, xihat_b_bottom_row, axis=0)

    # 4th Order Runge-Kutta
    for i in range(0, n_int-1):
        k_1 = dt * g_matrix[i] @ xihat_b
        k_2 = dt * (g_matrix[i]+0.5*k_1) @ xihat_b
        k_3 = dt * (g_matrix[i]+0.5*k_2) @ xihat_b
        k_4 = dt * (g_matrix[i]+k_3) @ xihat_b

        # Append g matrix with increment for each timestep
        g_matrix_append = np.array([g_matrix[i] + (1/6) * (k_1 + 2*k_2 + 2*k_3 + k_4)])
        g_matrix = np.append(g_matrix, g_matrix_append, axis=0)

    # Return g matrix, R_IB matrix, and delta matrix
    return (
        g_matrix,
        g_matrix[:, :3, :3],
        g_matrix[:, :3, 3].reshape(-1,3,1))

def run_kinematics_body():
    """Kinematics"""

    # Times for each segment
    t_0 = 0
    t_1 = t_0 + 2
    t_2 = t_1 + (3*np.pi)/2
    t_3 = t_2 + 2
    t_4 = t_3 + (3*np.pi)/2

    # Stuff
    dt = 0.01
    rot_i_b_0 = np.array([np.eye(3,3)])
    delta_0 = np.array([np.zeros((3,1))])
    # delta_0 = np.zeros((1,3,1))

    # Linear and angular velocity: segment 1
    omega_b_1    = np.array([0, 0, 0]).reshape(3,1)
    v_b_1        = np.array([1, 0, 0.2]).reshape(3,1)

    # Linear and angular velocity: segment 2
    omega_b_2    = np.array([0, 0, 1]).reshape(3,1)
    v_b_2        = np.array([1, 0, 0]).reshape(3,1)

    # Linear and angular velocity: segment 3
    omega_b_3    = np.array([0, 0, 0]).reshape(3,1)
    v_b_3        = np.array([1, 0, -0.2]).reshape(3,1)

    # Linear and angular velocity: segment 4
    omega_b_4    = np.array([0, 0, -1]).reshape(3,1)
    v_b_4        = np.array([1, 0, 0]).reshape(3,1)

    print(delta_0[-1])

    # pylint: disable-next=C0301
    (_g_out1, rot_i_b_out_1, delta_out1) = kinematics_body(t_0, t_1, omega_b_1, v_b_1, dt, rot_i_b_0, delta_0)

    print(delta_out1[-1])

    ree1 = np.array([rot_i_b_out_1[-1]])
    dee1 = np.array([delta_out1[-1]])

    # pylint: disable-next=C0301
    (_g_out2, rot_i_b_out_2, delta_out2) = kinematics_body(t_1, t_2, omega_b_2, v_b_2, dt, ree1, dee1)

    print(delta_out2[-1])

    ree2 = np.array([rot_i_b_out_2[-1]])
    dee2 = np.array([delta_out2[-1]])

    # pylint: disable-next=C0301
    (_g_out3, rot_i_b_out_3, delta_out3) = kinematics_body(t_2, t_3, omega_b_3, v_b_3, dt, ree2, dee2)

    print(delta_out3[-1])

    ree3 = np.array([rot_i_b_out_3[-1]])
    dee3 = np.array([delta_out3[-1]])

    # pylint: disable-next=C0301
    (_g_out4, _rot_i_b_out_4, delta_out4) = kinematics_body(t_3, t_4, omega_b_4, v_b_4, dt, ree3, dee3)

    print(delta_out4[-1])

    # TODO@dpwiese - return stuff here to plot

# pylint: disable-next=C0301
def wrenchint(t_0, t_f, dt, j_b, mass, omega_b_0, v_b_0, tau_b_0, f_b_0, rot_i_b_0, gbar, delta_i_b_0):
    """Wrench"""

    # Make sure shapes of input are correct
    assert j_b.shape == (3,3)
    assert omega_b_0.shape == (3,1)
    assert v_b_0.shape == (3,1)
    assert tau_b_0.shape == (3,1)
    assert f_b_0.shape == (3,1)
    assert rot_i_b_0.shape == (3,3)
    assert delta_i_b_0.shape == (3,1)

    # Step size
    n_int = int(((t_f-t_0)/dt)+1)

    # Initial velocity (translational and rotational)
    omega_b = np.array([omega_b_0])
    v_b = np.array([v_b_0])

    # Initial position and orientation
    rot_mat = np.array([rot_i_b_0])
    delta = np.array([delta_i_b_0])
    tau_b = tau_b_0
    f_b = f_b_0

    # Dissipative consant
    # k_diss = 0.0
    k_diss = 0.05

    # 4th Order Runge-Kutta
    for i in range(0, n_int-1):
        # pylint: disable=C0301
        k_1 = dt * inv(j_b) @ (np.cross(-omega_b[i], j_b @ omega_b[i], axisa=0, axisb=0).reshape(3,1) + tau_b)
        k_2 = dt * inv(j_b) @ (np.cross(-(omega_b[i]+0.5*k_1),j_b@(omega_b[i]+0.5*k_1), axisa=0, axisb=0).reshape(3,1)+tau_b)
        k_3 = dt * inv(j_b) @ (np.cross(-(omega_b[i]+0.5*k_2),j_b@(omega_b[i]+0.5*k_2), axisa=0, axisb=0).reshape(3,1)+tau_b)
        k_4 = dt * inv(j_b) @ (np.cross(-(omega_b[i]+k_3),j_b@(omega_b[i]+k_3), axisa=0, axisb=0).reshape(3,1)+tau_b)

        # Append omega with increment for each timestep
        omega_b_append = np.array([omega_b[i] + (1/6) * (k_1 + 2*k_2 + 2*k_3 + k_4)])
        omega_b = np.append(omega_b, omega_b_append, axis=0)

        k_1 = dt * (np.cross(-omega_b[i],(v_b[i]), axisa=0, axisb=0).reshape(3,1)+np.transpose(rot_mat[i]) @ gbar+(f_b/mass))
        k_2 = dt * (np.cross(-omega_b[i],(v_b[i]+0.5*k_1), axisa=0, axisb=0).reshape(3,1)+np.transpose(rot_mat[i]) @ gbar+(f_b/mass))
        k_3 = dt * (np.cross(-omega_b[i],(v_b[i]+0.5*k_2), axisa=0, axisb=0).reshape(3,1)+np.transpose(rot_mat[i]) @ gbar+(f_b/mass))
        k_4 = dt * (np.cross(-omega_b[i],(v_b[i]+k_3), axisa=0, axisb=0).reshape(3,1)+np.transpose(rot_mat[i]) @ gbar+(f_b/mass))
        # pylint: enable=C0301

        # Append v_b with increment for each timestep
        v_b_append = np.array([v_b[i] + (1/6) * (k_1 + 2*k_2 + 2*k_3 + k_4)])
        v_b = np.append(v_b, v_b_append, axis=0)

        k_1 = dt * (rot_mat[i]) @ rb.hat(omega_b[i])
        k_2 = dt * (rot_mat[i]+0.5*k_1) @ rb.hat(omega_b[i])
        k_3 = dt * (rot_mat[i]+0.5*k_2) @ rb.hat(omega_b[i])
        k_4 = dt * (rot_mat[i]+k_3) @ rb.hat(omega_b[i])

        # Append rot_mat with increment for each timestep
        rot_mat_append = np.array([rot_mat[i] + (1/6) * (k_1 + 2*k_2 + 2*k_3 + k_4)])
        rot_mat = np.append(rot_mat, rot_mat_append, axis=0)

        k_1 = dt * rot_mat[i] @ (v_b[i])
        k_2 = dt * rot_mat[i] @ (v_b[i]+0.5*k_1)
        k_3 = dt * rot_mat[i] @ (v_b[i]+0.5*k_1)
        k_4 = dt * rot_mat[i] @ (v_b[i]+k_3)

        # Append delta with increment for each timestep
        delta_append = np.array([delta[i] + (1/6) * (k_1 + 2*k_2 + 2*k_3 + k_4)])
        delta = np.append(delta, delta_append, axis=0)

        tau_b = -k_diss * omega_b[i]

    return (v_b, delta, rot_mat)

def run_wrenchint():
    """Wrench"""

    # Simulation times
    t_0     = 0
    t_f     = 20
    dt      = 0.01
    nt      = int((t_f-t_0)/dt+1)
    time    = np.linspace(t_0, t_f, nt)

    # Constants
    j_b         = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 5]])
    mass        = 1
    gbar        = np.array([0, 0, 0]).reshape(3,1)

    # Initial velocity (translational and rotational)
    omega_b_0   = np.array([1, 0, 0.2]).reshape(3,1)
    v_b_0       = np.array([0, 0, 0]).reshape(3,1)

    # Wrench acting on rigid body
    tau_b_0     = np.array([0, 0, 0]).reshape(3,1)
    f_b_0       = np.array([0, 0, 0]).reshape(3,1)

    # Initial position and orientation of rigid body
    rot_i_b_0   = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    delta_i_b_0 = np.array([0, 0, 0]).reshape(3,1)

    # Run integrator
    # pylint: disable-next=C0301
    [_v_bout, _delta_out, rot_mat_out] = wrenchint(t_0, t_f, dt, j_b, mass, omega_b_0, v_b_0, tau_b_0, f_b_0, rot_i_b_0, gbar, delta_i_b_0)

    # Convert to Euler angles
    euler_angles = []
    for i in range(0, len(rot_mat_out)-1):
        euler_angles = np.append(euler_angles, rb.rotation_matrix_to_euler(rot_mat_out[i]))

    t_plot = time[0:len(time)-1].reshape(-1)

    psi = np.array([eulr['psi'] for eulr in euler_angles])
    theta = np.array([eulr['theta'] for eulr in euler_angles])
    phi = np.array([eulr['phi'] for eulr in euler_angles])

    _fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(t_plot, psi.reshape(-1), label="Yaw (psi)")
    ax.plot(t_plot, theta.reshape(-1), label="Pitch (theta)")
    ax.plot(t_plot, phi.reshape(-1), label="Roll (phi)")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    run_kinematics_analytic()
    run_kinematics_body()
    run_wrenchint()

    # TODO@dpwiese - get return values from the above and plot here
