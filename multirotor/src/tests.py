import unittest
from Common import *


class TestUtilities(unittest.TestCase):
    df = pd.read_csv('unit_test.csv')

    def test_transformToBodyFrame(self):
        pass

    def test_transformToEarthFrame(self):
        pass

    def test_transformAngularRateToBody(self):
        pass

    def test_transformAngularRateToEarth(self):
        pass

    def test_wrapAroundPi(self):
        pass

    def test_getAveragesBody(self):
        pass

    def test_downSample(self):
        pass

    def test_toEulerianAngle(self):
        pass

    def test_eulerToQuaternion(self):
        pass

    def test_integrateTrajectoryAccelerationBody(self):
        pass


if __name__ == '__main__':
    unittest.main()
