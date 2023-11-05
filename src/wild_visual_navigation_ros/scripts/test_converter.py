import unittest
import torch
from ros_converter import*  # Replace 'main_module' with the actual name of your module
from geometry_msgs.msg import Point
from ros_converter import torch_tensor_to_geometry_msgs_PointArray

class TestRosTfToTorch(unittest.TestCase):
    def test_valid_input(self):
        """Test the function with valid input."""
        tf_pose = ([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])  # A simple translation with no rotation
        success, torch_matrix = ros_tf_to_torch(tf_pose, device="cpu")
        expected_matrix = torch.eye(4)  # Expected transformation matrix
        expected_matrix[0, 3] = 1.0
        expected_matrix[1, 3] = 2.0
        expected_matrix[2, 3] = 3.0
        
        self.assertTrue(success)
        self.assertTrue(torch.allclose(torch_matrix, expected_matrix))

    def test_none_input(self):
        """Test the function with None as input."""
        tf_pose = (None, None)
        success, torch_matrix = ros_tf_to_torch(tf_pose, device="cpu")
        self.assertFalse(success)
        self.assertIsNone(torch_matrix)

    def test_invalid_input_type(self):
        """Test the function with an invalid type of input."""
        tf_pose = "invalid input"
        with self.assertRaises(AssertionError):
            ros_tf_to_torch(tf_pose, device="cpu")

    # Add more test cases as necessary for different scenarios


class TestConversionTorch(unittest.TestCase):
    def test_tensor_to_point_array(self):
        # Create a test tensor
        test_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        # Expected list of Points
        expected_points = [
            Point(x=1.0, y=2.0, z=3.0),
            Point(x=4.0, y=5.0, z=6.0),
            Point(x=7.0, y=8.0, z=9.0),
        ]

        # Perform the conversion
        converted_points = torch_tensor_to_geometry_msgs_PointArray(test_tensor)

        # Check that the lengths are the same
        self.assertEqual(len(converted_points), len(expected_points))

        # Check that each Point is converted correctly
        for converted_point, expected_point in zip(converted_points, expected_points):
            self.assertAlmostEqual(converted_point.x, expected_point.x, places=5)
            self.assertAlmostEqual(converted_point.y, expected_point.y, places=5)
            self.assertAlmostEqual(converted_point.z, expected_point.z, places=5)

class TestRosTfToNumpy(unittest.TestCase):
    
    def test_conversion(self):
        # Test data
        translation = [1.0, 2.0, 3.0]
        quaternion = [0.0, 0.0, 0.0, 1.0]  # No rotation (identity quaternion)
        
        # Expected output
        expected_matrix = np.array([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Perform the conversion
        success, result_matrix = ros_tf_to_numpy((translation, quaternion))
        
        # Check the results
        self.assertTrue(success)
        np.testing.assert_array_almost_equal(result_matrix, expected_matrix, decimal=6)

    def test_invalid_input(self):
        # Test data with invalid input (None translation)
        tf_pose = (None, [0.0, 0.0, 0.0, 1.0])
        
        # Expected output is False, None
        expected_output = (False, None)
        
        # Perform the conversion
        output = ros_tf_to_numpy(tf_pose)
        
        # Check the results
        self.assertEqual(output, expected_output)
class TestConversionNumpy(unittest.TestCase):
    def test_tensor_to_point_array(self):
        # Create a test tensor
        test_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        # Expected list of Points
        expected_points = [
            Point(x=1.0, y=2.0, z=3.0),
            Point(x=4.0, y=5.0, z=6.0),
            Point(x=7.0, y=8.0, z=9.0),
        ]

        # Perform the conversion
        converted_points = np_to_geometry_msgs_PointArray(test_tensor)

        # Check that the lengths are the same
        self.assertEqual(len(converted_points), len(expected_points))

        # Check that each Point is converted correctly
        for converted_point, expected_point in zip(converted_points, expected_points):
            self.assertAlmostEqual(converted_point.x, expected_point.x, places=5)
            self.assertAlmostEqual(converted_point.y, expected_point.y, places=5)
            self.assertAlmostEqual(converted_point.z, expected_point.z, places=5)
if __name__ == '__main__':
    unittest.main()
