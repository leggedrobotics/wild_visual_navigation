import unittest
import numpy as np
from .meshes import make_ellipsoid,make_polygon_from_points,lerp

class TestSuperquadricCircle(unittest.TestCase):
    def test_circle_generation(self):
        # Circle parameters
        radius = 1.0
        A = B = 2*radius
        C = 0  # Very small value to approximate 0
        
        grid_size = 100  # The resolution of the circle

        # Generate the circle
        circle_points = make_ellipsoid(A, B, C,  grid_size=grid_size)

        # Check if the number of points is correct
        self.assertEqual(circle_points.shape[0], grid_size, "The number of generated points is incorrect.")

        # Check if all points are in the x-y plane and have the correct radius within a tolerance
        for point in circle_points:
            # Check if z-coordinate is approximately 0
            self.assertAlmostEqual(point[2].item(), 0, places=4, msg="Point is not in the x-y plane.")
            
            # Check the radius is approximately equal to the specified radius within some tolerance
            distance_from_origin = np.sqrt(point[0]**2 + point[1]**2)
            self.assertAlmostEqual(distance_from_origin.item(), radius, places=4, msg="Point is not at the correct radius.")


class TestMakePolygonFromPoints(unittest.TestCase):

    def test_square_interpolation(self):
        # Define a square with points
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        
        # The expected number of points for a grid size of 10 (excluding the last point of each edge)
        grid_size = 10
        expected_num_points = 4 * (grid_size )  
        
        # Call the function
        finer_points = make_polygon_from_points(square, grid_size=grid_size)
        
        # Check the number of points
        self.assertEqual(finer_points.shape[0], expected_num_points)
        
        # Check the first and last points
        self.assertTrue(np.allclose(finer_points[0], square[0]))
        self.assertTrue(np.allclose(finer_points[-1], square[0]))
        
        # Check that all points are within the expected range
        for point in finer_points:
            self.assertTrue(0 <= point[0] <= 1 and 0 <= point[1] <= 1)
        

# This allows the test to be run from the command line
if __name__ == '__main__':
    unittest.main()
