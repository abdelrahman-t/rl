from Utilities import *


class VelocityModel:
    def __init__(self, model, frequency):
        self.model = model
        self.frequency = frequency
        self.inverseKeymap = {'moveForward': 0, 'yawCCW': 0, 'yawCW': 0, 'hover': 0}

    def initialize(self, initialState):
        s1, s0 = initialState, initialState.prevState

        orientation1, orientation0 = getRollPitchYaw(s1.orientation), getRollPitchYaw(s0.orientation)
        angularVelocity = getAverageAngularVelocity(orientation1, orientation0, self.frequency)
        linearVelocity = transformToBodyFrame(getAverageLinearVelocity(s1.position, s0.position, self.frequency), s1.orientation)

        return State(update=False, position=s1.position, orientation=orientation1, linearVelocity=linearVelocity,
                     angularVelocity=angularVelocity)

    def updateState(self, state, action):
        x = np.concatenate(
            (state.linearVelocity, state.angularVelocity, state.orientation[:-1],
             [0 if k != action else 1 for k, v in self.inverseKeymap.items()]))
        y = self.model.predict(x.reshape(1, -1))[0]

        linearVelocity = y[[0, 1, 2]]
        angularVelocity = y[[3, 4, 5]]

        orientation = integrateOrientation(state.orientation, angularVelocity, self.frequency)
        position = integratePosition(state.position, transformToEarthFrame(linearVelocity, eulerToQuaternion(*orientation)), self.frequency)

        return State(update=False, position=position, orientation=orientation, linearVelocity=linearVelocity,
                     angularVelocity=angularVelocity)


class AccelerationModel:
    def __init__(self, model, frequency):
        self.model = model
        self.frequency = frequency
        self.inverseKeymap = {'moveForward': 8, 'yawCCW': 4, 'yawCW': 6, 'hover': 5}

    def initialize(self, initialState):
        pass

    def updateState(self, state, action):
        pass