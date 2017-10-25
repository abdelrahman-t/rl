import unittest
from Utilities import *
from Common import *


class TestUtilities(unittest.TestCase):
    df = pd.read_csv('unit_test.csv')

    def test_rotation(self):
        pass

    def test_quaternionConversion(self):
        pass

    def test_integrateOrientation(self):
        pass

    def test_integratePosition(self):
        pass

    def test_integrateTrajectoryVelocityBody(self):
        pass

    def test_integrateTrajectoryAccelerationBody(self):
        pass


if __name__ == '__main__':
    unittest.main()
