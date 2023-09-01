"""
Test the collection of a few things for rigid body motion
"""
import unittest
import numpy as np
import numpy.typing as npt
from numpy import eye
from scipy.spatial.transform import Rotation
from src.hat import (
    hat,
    inv_hat,
    eqax_to_rotation_matrix,
    rotation_matrix_to_eqax,
    eqax_to_mrp,
    mrp_to_eqax,
    # EquivalentAxis,
    rotation_matrix_to_quat,
    quat_to_rotation_matrix,
    rotation_matrix_to_euler,
    euler_to_rotation_matrix,
    twist_coords_to_g
)

class TestHat(unittest.TestCase):
    """Test functions to convert between rotation representation"""

    def test_hat(self):
        """hat makes skew symmetric matrix from column vector: array of size (3,1)"""

        vec_in: npt.NDArray[np.float64] = np.array([[1], [2], [3]])
        answer_mat: npt.NDArray[np.float64] = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])

        np.testing.assert_array_equal(answer_mat, hat(vec_in))

    def test_hat_bad_inputs(self):
        """hat raises error when inputs are not a column vector: (3,1) array"""

        test_cases = [
            {
                "input": np.array([1, 2, 3]),
                "error": AssertionError
            },
            {
                "input": np.array([[1], [2]]),
                "error": AssertionError
            },
            {
                "input": np.array([[1], [2], [3], [4]]),
                "error": AssertionError
            },
            {
                "input": 1,
                "error": AttributeError
            },
            {
                "input": "One",
                "error": AttributeError
            },
            {
                "input": [1, 2, 3],
                "error": AttributeError
            }
        ]

        for tc in test_cases:
            with self.assertRaises(tc["error"]):
                hat(tc["input"])

    def test_inv_hat(self):
        """inv_hat returns correct column vector from matrix input"""

        mat_in: npt.NDArray[np.float64] = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
        answer_vec: npt.NDArray[np.float64] = np.array([[1], [2], [3]])

        np.testing.assert_array_equal(answer_vec, inv_hat(mat_in))

    def test_inv_hat_bad_inputs(self):
        """inv_hat raises error when input is not a skew-symmetric matrix: (3,3) array"""

        test_cases = [
            {
                "input": np.array([[0, -3, 2], [3, 0, -1]]),
                "error": AssertionError
            },
            {
                "input": np.array([[0, -3], [3, 0], [-2, 1]]),
                "error": AssertionError
            },
            {
                "input": np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
                "error": AssertionError
            },
            {
                "input": 1,
                "error": AttributeError
            },
            {
                "input": "One",
                "error": AttributeError
            },
            {
                "input": [1, 2, 3],
                "error": AttributeError
            }
        ]

        for tc in test_cases:
            with self.assertRaises(tc["error"]):
                inv_hat(tc["input"])

    def test_mrp_to_eqax(self):
        """mrp_to_eqax converts MRP to equivalent axis representation"""

        # Calculate equivalent axis representation from MRP
        # TODO@dpwiese - is this array a valid MRP?
        eq_ax = mrp_to_eqax(np.array([[1], [2], [3]]))

        # TODO@dpwiese - make more and better tests here, with values from analytical calculations

        # Answer
        omega_ans = np.array([[0.267261], [0.534522], [0.801784]])
        theta_ans = 5.238555663567489

        # Check both omega and theta are correct
        # pylint: disable=C0301
        np.testing.assert_allclose(omega_ans, eq_ax["omega"], rtol=1e-05, atol=1e-08, equal_nan=False)
        np.testing.assert_equal(theta_ans, eq_ax["theta"])

    def test_mrp_to_eqax_bad_inputs(self):
        """mrp_to_eqax raises error when inputs are not a column vector: (3,1) array"""

        test_cases = [
            {
                "input": np.array([1, 2, 3]),
                "error": AssertionError
            },
            {
                "input": np.array([[1], [2]]),
                "error": AssertionError
            },
            {
                "input": np.array([[1], [2], [3], [4]]),
                "error": AssertionError
            },
            {
                "input": 1,
                "error": AttributeError
            },
            {
                "input": "One",
                "error": AttributeError
            },
            {
                "input": [1, 2, 3],
                "error": AttributeError
            }
        ]

        for tc in test_cases:
            with self.assertRaises(tc["error"]):
                mrp_to_eqax(tc["input"])

    def test_eqax_to_mrp(self):
        """Test stuff"""

        # Define input equivalent axis representation
        omega = np.array([[-0.26726124], [-0.53452248], [-0.80178373]])
        theta = 2.541527920405645

        eq_ax = {"omega": omega, "theta": theta}
        mrp = np.array([[-0.19707587], [-0.39415173], [-0.5912276]])

        np.testing.assert_allclose(mrp, eqax_to_mrp(eq_ax), rtol=1e-05, atol=1e-08, equal_nan=False)

    def test_eqax_to_mrp_bad_inputs(self):
        """Test stuff"""

        test_cases = [
            {
                "input": np.array([1, 2, 3]),
                "error": IndexError
            },
            {
                "input": np.array([[1], [2]]),
                "error": IndexError
            },
            {
                "input": np.array([[1], [2], [3], [4]]),
                "error": IndexError
            },
            {
                "input": {
                    "omega": np.array([1, 2, 3]),
                    "theta": 1
                },
                "error": AssertionError
            },
            {
                "input": {
                    "omega": np.array([[1], [2]]),
                    "theta": 1
                },
                "error": AssertionError
            },
            {
                "input": {
                    "omega": np.array([[1], [2], [3], [4]]),
                    "theta": 1
                },
                "error": AssertionError
            },
            {
                "input": {
                    "omega": 1,
                    "theta": 1
                },
                "error": AttributeError
            },
            {
                "input": {
                    "omega": "One",
                    "theta": 1
                },
                "error": AttributeError
            },
            {
                "input": {
                    "omega": [1, 2, 3],
                    "theta": 1
                },
                "error": AttributeError
            },
            {
                "input": {
                    "omega": np.array([[1], [2], [3]]),
                    "theta": "One"
                },
                "error": TypeError
            },
            {
                "input": {
                    "omega": np.array([[1], [2], [3]]),
                    "theta": [1]
                },
                "error": TypeError
            }
        ]

        for tc in test_cases:
            with self.assertRaises(tc["error"]):
                eqax_to_mrp(tc["input"])

    def test_eqax_to_rotation_matrix(self):
        """eqax_to_rotation_matrix correctly converts equivalent axis to rotation matrix"""

        # Test cases
        test_cases = [
            {
                "input": {
                    "omega": np.array([[0], [0], [0]]),
                    "theta": 0
                },
                "answer": eye(3, dtype=float)
            },
            {
                "input": {
                    # Regardless of omega, if theta is zero, there is no rotation
                    "omega": np.array([[1], [2], [3]]),
                    "theta": 0
                },
                "answer": eye(3, dtype=float)
            },
            {
                "input": {
                    # Regardless of theta, if omega is zero, there is no rotation
                    "omega": np.array([[0], [0], [0]]),
                    "theta": 1
                },
                "answer": eye(3, dtype=float)
            },
            {
                "input": {
                    # TODO@dpwiese - make this test case based on analytic inputs and outputs
                    "omega": np.array([[-0.26726124], [-0.53452248], [-0.80178373]]),
                    "theta": 1.0446296436120972
                },
                "answer": np.array([
                    [ 0.537778,  0.764444, -0.355556],
                    [-0.622222,  0.644444,  0.444444],
                    [ 0.568889, -0.017778,  0.822222]
                ])
            },
            {
                # A rotation about the x-axis is pure roll
                "input": {
                    "omega": np.array([[1], [0], [0]]),
                    "theta": 1
                },
                "answer": np.array([
                    [ 1,    0,          0         ],
                    [ 0,    np.cos(1), -np.sin(1) ],
                    [ 0,    np.sin(1),  np.cos(1) ]
                ])
            },
            {
                # A rotation about the y-axis is pure pitch
                "input": {
                    "omega": np.array([[0], [1], [0]]),
                    "theta": 1
                },
                "answer": np.array([
                    [ np.cos(1),    0,  np.sin(1) ],
                    [ 0,            1,  0         ],
                    [ -np.sin(1),   0,  np.cos(1) ]
                ])
            },
            {
                # A rotation about the z-axis is pure yaw
                "input": {
                    "omega": np.array([[0], [0], [1]]),
                    "theta": 1
                },
                "answer": np.array([
                    [ np.cos(1),   -np.sin(1),  0 ],
                    [ np.sin(1),    np.cos(1),  0 ],
                    [ 0,            0,          1 ]
                ])
            }
        ]

        # Test
        for tc in test_cases:
            ans = tc["answer"]
            rot_mat = eqax_to_rotation_matrix(tc["input"])
            np.testing.assert_allclose(ans, rot_mat, rtol=1e-04, atol=1e-08, equal_nan=False)

    def test_eqax_to_rotation_matrix_bad_inputs(self):
        """eqax_to_rotation_matrix raises errors when input is bad"""

        test_cases = [
            {
                "input": np.array([1, 2, 3]),
                "error": IndexError
            },
            {
                "input": np.array([[1], [2]]),
                "error": IndexError
            },
            {
                "input": np.array([[1], [2], [3], [4]]),
                "error": IndexError
            },
            {
                "input": {
                    "omega": np.array([1, 2, 3]),
                    "theta": 1
                },
                "error": AssertionError
            },
            {
                "input": {
                    "omega": np.array([[1], [2]]),
                    "theta": 1
                },
                "error": AssertionError
            },
            {
                "input": {
                    "omega": np.array([[1], [2], [3], [4]]),
                    "theta": 1
                },
                "error": AssertionError
            },
            {
                "input": {
                    "omega": 1,
                    "theta": 1
                },
                "error": AttributeError
            },
            {
                "input": {
                    "omega": "One",
                    "theta": 1
                },
                "error": AttributeError
            },
            {
                "input": {
                    "omega": [1, 2, 3],
                    "theta": 1
                },
                "error": AttributeError
            },
            {
                "input": {
                    "omega": np.array([[1], [2], [3]]),
                    "theta": "One"
                },
                "error": TypeError
            },
            {
                "input": {
                    "omega": np.array([[1], [2], [3]]),
                    "theta": [1]
                },
                "error": TypeError
            },
            {
                "input": {
                    "omega": np.array([[1], [2], [3]]),
                    "theta": np.array([[1], [2], [3]])
                },
                "error": ValueError
            }
        ]

        for tc in test_cases:
            with self.assertRaises(tc["error"]):
                eqax_to_rotation_matrix(tc["input"])

    def test_rotation_matrix_to_eqax(self):
        """Test stuff"""

        # Equivalent axes representation - omega and time
        # Identity rotation matrix is zero rotation (omega) for zero time (theta)
        omega_ans = np.array([[0], [0], [0]])
        eq_ax = rotation_matrix_to_eqax(eye(3, dtype=float))

        # TODO@dpwiese - more testing here
        np.testing.assert_array_equal(omega_ans, eq_ax["omega"])
        np.testing.assert_equal(0, eq_ax["theta"])

    def test_rotation_matrix_to_eqax_bad_inputs(self):
        """rotation_matrix_to_eqax raises error if input is not an array (3,3)"""

        test_cases = [
            {
                "input": np.array([1, 2, 3]),
                "error": AssertionError
            },
            {
                "input": np.array([[1], [2]]),
                "error": AssertionError
            },
            {
                "input": np.array([[1], [2], [3], [4]]),
                "error": AssertionError
            },
            {
                "input": 1,
                "error": AttributeError
            },
            {
                "input": "One",
                "error": AttributeError
            },
            {
                "input": [1, 2, 3],
                "error": AttributeError
            }
        ]

        for tc in test_cases:
            with self.assertRaises(tc["error"]):
                rotation_matrix_to_eqax(tc["input"])

    # TODO@dpwiese - test bad inputs
    def test_rotation_matrix_to_quat(self):
        """rotation_matrix_to_quat converts rotation matrix to quaternion correctly"""

        # Define arbitrary rotation matrix
        rot_mat = np.array([
            [ 0.53777778,   0.76444444, -0.35555556 ],
            [-0.62222222,   0.64444444,  0.44444444 ],
            [ 0.56888889,  -0.01777778,  0.82222222 ]
        ])

        q_ans = np.array([-0.13333333, -0.26666667, -0.4, 0.86666667])

        # Determine quaternion
        q_out = rotation_matrix_to_quat(rot_mat)

        # Another "answer" from comparing against scipy.spatial.transform
        q_ans_2 = Rotation.from_matrix(rot_mat).as_quat()

        # Check
        np.testing.assert_allclose(q_ans, q_out, rtol=1e-05, atol=1e-08, equal_nan=False)
        np.testing.assert_allclose(q_ans_2, q_out, rtol=1e-05, atol=1e-08, equal_nan=False)

    # TODO@dpwiese - test bad inputs
    def test_quat_to_rotation_matrix(self):
        """quat_to_rotation_matrix converts rotation matrix to quaternion"""

        # Define arbitrary quaternion (scalar-last format)
        quat = np.array([
            [ 0.13333333 ],
            [ 0.26666667 ],
            [ 0.4        ],
            [-0.86666667 ],
        ])

        rot_mat_ans = np.array([
            [ 0.53777778,   0.76444444, -0.35555556 ],
            [-0.62222222,   0.64444444,  0.44444444 ],
            [ 0.56888889,  -0.01777778,  0.82222222 ]
        ])

        # Determine rotation matrix
        rot_mat = quat_to_rotation_matrix(quat)

        # Another "answer" from comparing against scipy.spatial.transform
        rot_mat_ans_2 = Rotation.from_quat(quat.reshape(4,)).as_matrix()

        # Check
        np.testing.assert_allclose(rot_mat_ans, rot_mat, rtol=1e-05, atol=1e-08, equal_nan=False)
        np.testing.assert_allclose(rot_mat_ans_2, rot_mat, rtol=1e-05, atol=1e-08, equal_nan=False)

    # TODO@dpwiese - test bad inputs
    def test_rotation_matrix_to_euler(self):
        """rotation_matrix_to_euler converts rotation matrix to Euler angles"""

        # TODO@dpwiese - build this matrix analytically e.g. from Euler angles
        test_cases = [
            {
                # Define arbitrary rotation matrix
                "input": np.array([
                    [ 0.53777778,   0.76444444, -0.35555556 ],
                    [-0.62222222,   0.64444444,  0.44444444 ],
                    [ 0.56888889,  -0.01777778,  0.82222222 ]
                ]),
                "answer": {
                    "psi": -0.858067,
                    "theta": -0.605154,
                    "phi": -0.0216182
                }
            }
        ]

        for tc in test_cases:
            # Calculate Euler angles
            euler = rotation_matrix_to_euler(tc["input"])

            # Another "answer" from comparing against scipy.spatial.transform
            euler_ans_2 = Rotation.from_matrix(tc["input"]).as_euler("ZYX")

            # TODO@dpwiese - test all elements of dict at same time
            # pylint: disable=C0301
            # np.testing.assert_allclose(tc["answer"], euler, rtol=1e-05, atol=1e-08, equal_nan=False)
            # pylint: disable=C0301
            # np.testing.assert_allclose(euler_ans_2, euler, rtol=1e-05, atol=1e-08, equal_nan=False)

            # Check
            np.testing.assert_allclose(tc["answer"]["psi"], euler["psi"],
                rtol=1e-05, atol=1e-08, equal_nan=False)
            np.testing.assert_allclose(tc["answer"]["theta"], euler["theta"],
                rtol=1e-05, atol=1e-08, equal_nan=False)
            np.testing.assert_allclose(tc["answer"]["phi"], euler["phi"],
                rtol=1e-05, atol=1e-08, equal_nan=False)

            # Check against
            np.testing.assert_allclose(euler_ans_2[0], euler["psi"],
                rtol=1e-05, atol=1e-08, equal_nan=False)
            np.testing.assert_allclose(euler_ans_2[1], euler["theta"],
                rtol=1e-05, atol=1e-08, equal_nan=False)
            np.testing.assert_allclose(euler_ans_2[2], euler["phi"],
                rtol=1e-05, atol=1e-08, equal_nan=False)

    # TODO@dpwiese - test bad inputs
    def test_euler_to_rotation_matrix(self):
        """euler_to_rotation_matrix converts Euler angles to rotation matrix"""

        # Define some arbitrary Euler angles
        my_eulers = np.array([0.1, 0.2, 0.3])

        rot_mat_ans = np.array([
            [ 0.97517033, -0.03695701,  0.21835066],
            [ 0.0978434,   0.95642509, -0.27509585],
            [-0.19866933,  0.28962948,  0.93629336]
        ])

        # Convert Euler angles to rotation matrix
        rot_mat = euler_to_rotation_matrix({"psi": 0.1, "theta": 0.2, "phi": 0.3})

        # Another "answer" from comparing against scipy.spatial.transform
        rot_mat_ans_2 = Rotation.from_euler("ZYX", my_eulers).as_matrix()

        # Check
        np.testing.assert_allclose(rot_mat_ans, rot_mat, rtol=1e-05, atol=1e-08, equal_nan=False)
        np.testing.assert_allclose(rot_mat_ans_2, rot_mat, rtol=1e-05, atol=1e-08, equal_nan=False)

    # TODO@dpwiese - test bad inputs
    def test_twist_coords_to_g(self):
        """twist_coords_to_g converts twist triplet into g matrix"""

        test_cases = [
            {
                "twist": {
                    # No angular or linear velocity then g is just identity
                    "omega": np.array([[0], [0], [0]]),
                    "velocity": np.array([[0], [0], [0]]),
                    "theta": 1
                },
                "g_matrix_ans": eye(4, dtype=float)
            },
            {
                "twist": {
                    # No angular velocity then g is just identity with vt in upper right block
                    "omega": np.array([[0], [0], [0]]),
                    "velocity": np.array([[1], [2], [3]]),
                    "theta": 1
                },
                "g_matrix_ans": np.array([
                    [   1,  0,  0,  1 ],
                    [   0,  1,  0,  2 ],
                    [   0,  0,  1,  3 ],
                    [   0,  0,  0,  1 ]
                ])
            },
            {
                "twist": {
                    "omega": np.array([[1], [2], [3]]),
                    "velocity": np.array([[3], [4], [3]]),
                    "theta": 1
                },
                "g_matrix_ans": np.array([
                    [ 0.573138, -0.609007,  0.548292,  2.013727],
                    [ 0.740349,  0.671645, -0.027879,  4.55598 ],
                    [-0.351279,  0.421906,  0.835822,  2.958104],
                    [ 0.      ,  0.      ,  0.      ,  1.      ]
                ])
            }
        ]

        for tc in test_cases:
            # Calculate g from twist
            g_matrix = twist_coords_to_g(tc["twist"])

            # Check
            np.testing.assert_allclose(tc["g_matrix_ans"], g_matrix, rtol=1e-05, atol=1e-08, equal_nan=False)

if __name__ == '__main__':
    unittest.main()
