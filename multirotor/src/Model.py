from Utilities import *


class AccelerationModel:
    def __init__(self, regressionModel, frequency):
        self.model = regressionModel
        self.frequency = frequency

        self.actionNames = ['moveForward', 'yawCCW', 'yawCW', 'hover']

    def initialize(self, initialState):
        orientation = initialState.orientation
        euler = toEulerianAngle(orientation)
        linearVelocity = transformToBodyFrame(initialState.linearVelocity, initialState.orientation)

        return StateT(update=False, position=initialState.position, orientation=euler,
                      linearVelocityBody=linearVelocity, bodyRates=initialState.angularVelocity)

    def updateState(self, state, action):
        selectedAction = [0 if action != a else 1 for a in self.actionNames]
        roll, pitch = state.orientation[0], state.orientation[1]

        x = np.concatenate((state.linearVelocityBody, state.bodyRates, [roll, pitch], selectedAction))
        y = self.model.predict(x.reshape(1, -1))[0]

        gen = integrateTrajectoryAccelerationBody(state.position, state.orientation,
                                                  state.linearVelocityBody, state.bodyRates,
                                                  [y[[0, 1, 2]]], [y[[3, 4, 5]]], [self.frequency])

        position, orientation, linearVelocityBody, bodyRates = next(gen)

        return StateT(update=False, position=position, orientation=orientation, linearVelocityBody=linearVelocityBody,
                      bodyRates=bodyRates)
