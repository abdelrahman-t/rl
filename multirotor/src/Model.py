from Utilities import *


class AccelerationModel:
    def __init__(self, regressionModel, frequency):
        self.model = regressionModel
        self.frequency = frequency

        self.actionNames = ['moveForward', 'yawCCW', 'yawCW', 'hover']

    def initialize(self, initialState):
        pass

    def updateState(self, state, action):
        pass


class VelocityModel:
    def __init__(self, regressionModel, frequency):
        self.model = regressionModel
        self.frequency = frequency

        self.actionNames = ['moveForward', 'yawCCW', 'yawCW', 'hover']

    def initialize(self, initialState):
        orientation = Quaternion(eulerToQuaternion(0, 0, 0))
        linearVelocity = np.array([0, 0, 0])
        angularVelocity = np.array([0, 0, 0])
        position = np.array([0.0, 0.0, 0.0])
        frequency = self.frequency

        return StateT(update=False, position=position, orientation=orientation,
                      linearVelocity=linearVelocity, angularVelocity=angularVelocity)

    def updateState(self, state, action):
        selectedAction = np.array([0 if action != a else 1 for a in self.actionNames])
        position, orientation, frequency = state.position, state.orientation, self.frequency
        linearVelocity, angularVelocity = state.linearVelocity, state.angularVelocity

        roll, pitch, yaw = toEulerianAngle(orientation)

        s0 = np.concatenate((linearVelocity, angularVelocity, [roll, pitch], selectedAction))
        s1 = self.model.predict(s0.reshape(1, -1)).reshape(6, )

        orientation, position, linearVelocity, angularVelocity = \
            next(integrateTrajectoryVelocityBody(position=position, orientation=orientation,
                                                 frequency=[frequency],
                                                 linearVelocities=[s1[[0, 1, 2]]],
                                                 angularVelocities=[s1[[3, 4, 5]]]))

        return StateT(update=False, position=position, orientation=orientation,
                      linearVelocity=linearVelocity, angularVelocity=angularVelocity)
