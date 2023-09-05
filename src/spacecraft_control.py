"""
Stabilizing spacecraft - 16.333 HW #2
This script models the rotational dynamics of a tumbling spacecraft and designs and compares the
performance of two controllers in regulating the vehicle state back to the origin. The state vector
is of length 6 and includes the angular velocity about each of the body axes, and Euler angles to
describe the vehicle's orientation. The input is torque about each of the three body axes.
"""

import control
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import rigidbody as rb

def moment_equations_state(_t, x_state, u_input, _params):
    """State is omega_b input is tau_b"""

    # Parse input (tau_b) and state of moment equations (omega_b)
    tau_b   = np.array([u_input[0], u_input[1], u_input[2]]).reshape(3,1)
    omega_b = np.array([x_state[0], x_state[1], x_state[2]]).reshape(3,1)

    # Calculate and return omega_b_dot
    return inv(J_B) @ ((-rb.hat(omega_b) @ J_B) @ omega_b + tau_b)

def moment_equations_output(_t, x_state, _u_input, _params):
    """Output is omega_b_dot"""

    # Return omega_b_dot
    return x_state

def orientation_equations_state(_t, x_state, u_input, _params):
    """State is euler angles (phi, theta, psi) input is omega_b"""

    # Parse state
    phi     = x_state[0]
    theta   = x_state[1]

    # Build output
    row_1 = np.array([1, np.tan(theta) * np.sin(phi), np.tan(theta) * np.cos(phi)])
    row_2 = np.array([0, np.cos(theta), -np.sin(phi)])
    row_3 = np.array([0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)])

    # Calculate and return euler_dot
    return np.array([row_1, row_2, row_3]) @ u_input.reshape(3,1)

def orientation_equations_output(_t, x_state, _u_input, _params):
    """Output is time derivative of Euler angles (phi_dot, theta_dot, psi_dot)"""

    # Return Euler time derivatives
    return x_state

# System parameters: moment equations
# Input: tau_b
# State: omega_b
# Output: omega_b
IO_MOMENT_EQUATIONS = control.NonlinearIOSystem(
    moment_equations_state,
    moment_equations_output,
    inputs=3,
    outputs=3,
    states=3,
    name='moment',
    dt=0
)

# System parameters: Euler equations
# Input: omega_b
# State: euler
# Output: euler
IO_ORIENTATION_EQUATIONS = control.NonlinearIOSystem(
    orientation_equations_state,
    orientation_equations_output,
    inputs=3,
    outputs=3,
    states=3,
    name='orientation',
    dt=0
)

# System parameters: combined open-loop system
# Input: tau_b
# State: [omega_b, euler]
# Output: [omega_b, euler]
IO_OPEN_LOOP = control.InterconnectedSystem(
    (IO_MOMENT_EQUATIONS, IO_ORIENTATION_EQUATIONS),
    connections=(
        ('orientation.u[0]', 'moment.y[0]'),
        ('orientation.u[1]', 'moment.y[1]'),
        ('orientation.u[2]', 'moment.y[2]')
    ),
    inplist=('moment.u[0]', 'moment.u[1]', 'moment.u[2]'),
    outlist=(
        'moment.y[0]',
        'moment.y[1]',
        'moment.y[2]',
        'orientation.y[0]',
        'orientation.y[1]',
        'orientation.y[2]'
    ),
    dt=0
)

# Vehicle moment of inertia
J_B = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 5]])

# Equilibrium point about which to linearize
# TODO@dpwiese - resolve redundancy in specifying both R_EQ and X_EQ
R_EQ = np.eye(3, dtype=float)
X_EQ = np.zeros(6)
U_EQ = np.zeros(3)

# Rotation matrix that describes rotation matrix from inertial frame to desired orientation
R_D = R_EQ

# Linearize nonlinear spacecraft dynamics
linsys = control.linearize(IO_OPEN_LOOP, X_EQ, U_EQ)

# Make LQR controller
Q = np.eye(6, dtype=float)
R = np.eye(3, dtype=float)
K, S, E = control.lqr(linsys, Q, R)

# Controller
# input: x (vehicle state)
# state: nothing, but TODO@dpwiese - I don't know yet if I can have zero state?
# output: -K * x
A_C = np.zeros((1,1))
B_C = np.zeros((1,6))
C_C = np.zeros((3,1))
D_C = -K

# Define controller
IO_LINEAR_CONTROL = control.LinearIOSystem(
    control.StateSpace(A_C, B_C, C_C, D_C),
    inputs=6,
    outputs=3,
    states=1,
    name='linear_control'
)

def nonlinear_controller_state(_t, x_state, _u_input, _params):
    """Controller doesn't have internal state, input is plant state"""

    # Calculate and return the controller state, it won't be used anyway
    return x_state

def nonlinear_controller_output(_t, _x_state, u_input, _params):
    """Input is the plant state, controller doesn't have state, output is tau_b"""

    # Parse input to get omega
    omega = np.array([u_input[0], u_input[1], u_input[2]]).reshape(3,1)

    # pylint: disable-next=C0301
    rot_mat = rb.euler_to_rotation_matrix({"phi": u_input[3], "theta": u_input[4], "psi": u_input[5]})
    omega_d = np.zeros((3,1))
    rot_mat_d = np.eye(3, dtype=float)

    # Get velocity and proportional parts of gain
    kv_gain = K[:, :3]
    kp_gain = K[:, 3:]

    term_1 = -np.transpose(rb.inv_hat(kp_gain @ np.transpose(rot_mat_d) @ rot_mat))
    term_2 = -kv_gain @ (omega - rot_mat @ np.transpose(rot_mat_d) @ omega_d)

    # Return control
    return term_1.reshape(3,1) + term_2

# System parameters: moment equations
# Input: tau_b
# State: omega_b
# Output: omega_b
IO_NONLINEAR_CONTROL = control.NonlinearIOSystem(
    nonlinear_controller_state,
    nonlinear_controller_output,
    inputs=6,
    outputs=3,
    states=1,
    name='nonlinear_control',
    dt=0
)

# System parameters: combined open-loop system
# Input: None
# State: [omega_b, euler]
# Output: [omega_b, euler]
IO_CLOSED_LOOP_LINEAR = control.InterconnectedSystem(
    (IO_MOMENT_EQUATIONS, IO_ORIENTATION_EQUATIONS, IO_LINEAR_CONTROL),
    connections=(
        ('orientation.u[0]',    'moment.y[0]'),
        ('orientation.u[1]',    'moment.y[1]'),
        ('orientation.u[2]',    'moment.y[2]'),
        ('moment.u[0]',         'linear_control.y[0]'),
        ('moment.u[1]',         'linear_control.y[1]'),
        ('moment.u[2]',         'linear_control.y[2]'),
        ('linear_control.u[0]', 'moment.y[0]'),
        ('linear_control.u[1]', 'moment.y[1]'),
        ('linear_control.u[2]', 'moment.y[2]'),
        ('linear_control.u[3]', 'orientation.y[0]'),
        ('linear_control.u[4]', 'orientation.y[1]'),
        ('linear_control.u[5]', 'orientation.y[2]'),
    ),
    inplist=(),
    outlist=(
        'moment.y[0]',
        'moment.y[1]',
        'moment.y[2]',
        'orientation.y[0]',
        'orientation.y[1]',
        'orientation.y[2]'
    ),
    dt=0
)

# System parameters: combined open-loop system
# Input: None
# State: [omega_b, euler]
# Output: [omega_b, euler]
IO_CLOSED_LOOP_NONLINEAR = control.InterconnectedSystem(
    (IO_MOMENT_EQUATIONS, IO_ORIENTATION_EQUATIONS, IO_NONLINEAR_CONTROL),
    connections=(
        ('orientation.u[0]',        'moment.y[0]'),
        ('orientation.u[1]',        'moment.y[1]'),
        ('orientation.u[2]',        'moment.y[2]'),
        ('moment.u[0]',             'nonlinear_control.y[0]'),
        ('moment.u[1]',             'nonlinear_control.y[1]'),
        ('moment.u[2]',             'nonlinear_control.y[2]'),
        ('nonlinear_control.u[0]',  'moment.y[0]'),
        ('nonlinear_control.u[1]',  'moment.y[1]'),
        ('nonlinear_control.u[2]',  'moment.y[2]'),
        ('nonlinear_control.u[3]',  'orientation.y[0]'),
        ('nonlinear_control.u[4]',  'orientation.y[1]'),
        ('nonlinear_control.u[5]',  'orientation.y[2]'),
    ),
    inplist=(),
    outlist=(
        'moment.y[0]',
        'moment.y[1]',
        'moment.y[2]',
        'orientation.y[0]',
        'orientation.y[1]',
        'orientation.y[2]'
    ),
    dt=0
)

# Set simulation duration and time steps
N_POINTS = 1000
T_F = 15

# Set initial conditions
X0 = np.zeros((6, 1))
X0 = np.array([1.2, -0.5, 0, 0, 0, np.pi/6, 0])

# Define simulation time span and control input
T = np.linspace(0, T_F, N_POINTS)

# There is no input to the closed-loop system
U = np.zeros((0, N_POINTS))

# Simulate the system
T_OUT_LIN, Y_OUT_LIN = control.input_output_response(IO_CLOSED_LOOP_LINEAR, T, U, X0)
T_OUT_NON, Y_OUT_NON = control.input_output_response(IO_CLOSED_LOOP_NONLINEAR, T, U, X0)

# Plot the response
plt.rc('text', usetex=True)
plt.rc('font', family='sans')

# FIG = plt.figure(1, figsize=(6, 6), dpi=300, facecolor='w', edgecolor='k')
FIG = plt.figure(1, figsize=(6, 6), dpi=100, facecolor='w', edgecolor='k')

RED = '#f62d73'
BLUE = '#1269d3'
WHITE = '#ffffff'
GREEN = '#2df643'
BLACK = '#000000'

AX_1 = FIG.add_subplot(1, 1, 1)
AX_1.plot(T_OUT_LIN, Y_OUT_LIN[3], label=r'$\phi$ (linear)', color=BLACK, linestyle='dashed')
AX_1.plot(T_OUT_LIN, Y_OUT_LIN[4], label=r'$\theta$ (linear)', color=BLUE, linestyle='dashed')
AX_1.plot(T_OUT_LIN, Y_OUT_LIN[5], label=r'$\psi$ (linear)', color=RED, linestyle='dashed')
AX_1.plot(T_OUT_NON, Y_OUT_NON[3], label=r'$\phi$ (nonlinear)', color=BLACK)
AX_1.plot(T_OUT_NON, Y_OUT_NON[4], label=r'$\theta$ (nonlinear)', color=BLUE)
AX_1.plot(T_OUT_NON, Y_OUT_NON[5], label=r'$\psi$ (nonlinear)', color=RED)
AX_1.set_xlabel(r'time ($t$)', fontname="Times New Roman", fontsize=9, fontweight=100)
AX_1.legend(loc="lower right", bbox_to_anchor=(1, 0), fontsize=9)
AX_1.set_facecolor(WHITE)

plt.show()
