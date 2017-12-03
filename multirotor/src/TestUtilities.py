import unittest
from Common import *


class TestUtilities(unittest.TestCase):
    df = pd.read_csv('unit_test.csv')

    def test_transform_to_body(self):
        pass

    def test_transform_to_earth(self):
        pass

    def test_transform_angular_rates_to_body(self):
        pass

    def test_transform_angular_rates_to_earth(self):
        pass

    def test_wrap_around_pi(self):
        pass

    def test_get_average_velocities(self):
        pass

    def test_down_sample(self):
        pass

    def test_toEulerianAngle(self):
        pass

    def test_euler_to_quaternion(self):
        pass

    def test_integrateTrajectoryAccelerationBody(self):
        pass


if __name__ == '__main__':
    unittest.main()
