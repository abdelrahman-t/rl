import unittest
from Utilities import *
from Common import *

import pandas as pd


class TestUtilities(unittest.TestCase):
    @staticmethod
    def addSuffix(l, suffix):
        return [i + suffix for i in l]

    def test_quaternionConversion(self):
        df = pd.read_csv('unit_test2.csv')
        euler = ['psi', 'theta', 'phi']
        quaternion = ['scalar', 'i', 'j', 'k']

        eulerAngles = np.concatenate((df[self.addSuffix(euler, '0')].values, df[self.addSuffix(euler, '1')].values))
        quaternions = np.concatenate((df[self.addSuffix(quaternion, '0')].values, df[self.addSuffix(quaternion, '1')].values))

        for e, q in zip(eulerAngles, quaternions):
            self.assertTrue(np.allclose(toEulerianAngle(eulerToQuaternion(*e)), e))

    def test_rotation(self):
        df = pd.read_csv('unit_test2.csv')

        quaternion = ['scalar', 'i', 'j', 'k']
        v = ['dX', 'dY', 'dZ']

        vEarth = np.concatenate((df[self.addSuffix(v, 'E0')].values, df[self.addSuffix(v, 'E1')].values))
        quaternions = np.concatenate((df[self.addSuffix(quaternion, '0')].values, df[self.addSuffix(quaternion, '1')].values))

        for vE, q in zip(vEarth, quaternions):
            q_ = Quaternion(q)
            vBody = transformToBodyFrame(vE, q_)
            self.assertTrue(np.allclose(transformToEarthFrame(vBody, q_), vE))

    def test_integrateOrientation(self):
        df = pd.read_csv('unit_test2.csv')

        orientation = ['psi', 'theta', 'phi']
        p0 = df[self.addSuffix(orientation, '0')].values
        p1 = df[self.addSuffix(orientation, '1')].values
        f = df['f'].values

        for nextPosition, initialPosition, frequency in zip(p1, p0, f):
            averageVelocity = getAverageAngularVelocity(nextPosition, initialPosition, frequency)
            self.assertTrue(np.allclose(nextPosition, integrateOrientation(initialPosition, averageVelocity, frequency)))

    def test_integratePosition(self):
        df = pd.read_csv('unit_test2.csv')

        position = ['x', 'y', 'z']
        p0 = df[self.addSuffix(position, '0')].values
        p1 = df[self.addSuffix(position, '1')].values
        f = df['f'].values

        for nextPosition, initialPosition, frequency in zip(p1, p0, f):
            averageVelocity = getAverageLinearVelocity(nextPosition, initialPosition, frequency)
            self.assertTrue(np.allclose(nextPosition, integratePosition(initialPosition, averageVelocity, frequency)))

    def test_integrateTrajectoryWithVelocity(self):
        df = pd.read_csv('unit_test2.csv')

        f = df['f'].values
        position = ['x', 'y', 'z']
        orientation = ['psi', 'theta', 'phi']
        quaternion = ['scalar', 'i', 'j', 'k']

        initialPositions = df[self.addSuffix(position, '0')].values
        nextPositions = df[self.addSuffix(position, '1')].values
        quaternions = df[self.addSuffix(quaternion, '1')].values

        initialOrientations = df[self.addSuffix(orientation, '0')].values
        nextOrientations = df[self.addSuffix(orientation, '1')].values

        vEarth_avg = np.array(list(starmap(getAverageLinearVelocity, zip(nextPositions, initialPositions, f))))
        vBody_avg = np.array(list(starmap(transformToBodyFrame, zip(vEarth_avg, quaternions))))
        angularVelocities_avg = np.array(list(starmap(getAverageAngularVelocity, zip(nextOrientations, initialOrientations, f))))

        gen = integrateTrajectory(initialPositions[0], initialOrientations[0], vBody_avg, angularVelocities_avg, f)
        for i in range(df.shape[0]):
            j = next(gen)
            self.assertTrue(np.allclose(j, nextPositions[i]), (j - nextPositions[i], i))


if __name__ == '__main__':
    unittest.main()
