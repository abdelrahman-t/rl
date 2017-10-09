from Common import *


class KeyT:
    def __init__(self):
        self.value = ''

    def update(self, value):
        self.value = str(value).replace('\'', '')

    def clear(self):
        self.value = ''


def getDateTime():
    return datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


def createConsoleLogger(name, level=None):
    level = globalLoggingLevel if level is None else level

    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(funcName)s @%(name)s : %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False

    return logger


def truncateFloat(number, decimalPlaces):
    return ("{0:.%sf}" % decimalPlaces).format(number)


def toEulerianAngle(q):
    ysqr = q[2] * q[2]

    t0 = +2.0 * (q[0] * q[1] + q[2] * q[3])
    t1 = +1.0 - 2.0 * (q[1] * q[1] + ysqr)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (q[0] * q[2] - q[3] * q[1])
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)

    t3 = +2.0 * (q[0] * q[3] + q[1] * q[2])
    t4 = +1.0 - 2.0 * (ysqr + q[3] * q[3])
    yaw = np.arctan2(t3, t4)

    return np.array([roll, pitch, yaw])


def getRollPitchYaw(q):
    return toEulerianAngle(q)


def eulerToQuaternion(roll, pitch, yaw):
    quaternion = [0] * 4
    cosPhi_2 = np.cos(roll / 2)
    sinPhi_2 = np.sin(roll / 2)
    cosTheta_2 = np.cos(pitch / 2)
    sinTheta_2 = np.sin(pitch / 2)
    cosPsi_2 = np.cos(yaw / 2)
    sinPsi_2 = np.sin(yaw / 2)

    quaternion[0] = (cosPhi_2 * cosTheta_2 * cosPsi_2 +
                     sinPhi_2 * sinTheta_2 * sinPsi_2)
    quaternion[1] = (sinPhi_2 * cosTheta_2 * cosPsi_2 -
                     cosPhi_2 * sinTheta_2 * sinPsi_2)
    quaternion[2] = (cosPhi_2 * sinTheta_2 * cosPsi_2 +
                     sinPhi_2 * cosTheta_2 * sinPsi_2)
    quaternion[3] = (cosPhi_2 * cosTheta_2 * sinPsi_2 -
                     sinPhi_2 * sinTheta_2 * cosPsi_2)

    return Quaternion(quaternion)


def transformToEarthFrame(vector, q):
    q_ = Quaternion(q)
    return q_.inverse.rotate(vector)


def transformToBodyFrame(vector, q):
    q_ = Quaternion(q)
    return q_.rotate(vector)


def generateGaussianNoise(std, mean=0):
    return np.random.normal(scale=std, loc=mean)


def wrapAroundPi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def integrateOrientation(euler, angularVelocity, f):
    return np.array(list(map(wrapAroundPi, euler + angularVelocity / f)))


def integratePosition(p0, v, f):
    return p0 + v / f


def getAverageAngularVelocity(p1, p0, f):
    return np.array(list(map(wrapAroundPi, p1 - p0))) * f


def getAverageLinearVelocity(p1, p0, f):
    return np.array((p1 - p0) * f)


def integrateTrajectory(initialPosition, initialOrientation, linearVelocitiesBody, angularVelocities, frequency):
    for v, w, f in zip(linearVelocitiesBody, angularVelocities, frequency):
        initialOrientation = integrateOrientation(initialOrientation, w, f)
        initialPosition = integratePosition(initialPosition,
                                            transformToEarthFrame(v, eulerToQuaternion(*initialOrientation)), f)
        yield initialPosition


class State:
    # if update is set to True, the environment will be able to update state whenever is needed.
    def __init__(self, update, **kwargs):
        self.prevState = None
        self.callbacks = OrderedDict()

        if not kwargs:
            pass
        elif 'state' in kwargs:
            for key, value in kwargs['state'].getKeyValuePairs.items():
                setattr(self, key, value)
        else:
            for key in kwargs:
                self.callbacks[key] = kwargs[key]
                setattr(self, key, None if update is True else kwargs[key])

    def areEqual(self, state, margins):
        for key in state.callbacks:
            diff = abs(getattr(self, key) - getattr(state, key))
            if (np.greater(diff, getattr(margins, key))).any():
                return False
        return True

    # States are immutable by design, each update creates a new State object
    def updateState(self, agent):
        temp = State(update=True)
        temp.callbacks = self.callbacks
        # keep track of previous state
        temp.prevState = self

        i, keys = 0, [i for i in temp.callbacks]

        while i < len(keys):
            key = keys[i]
            try:
                setattr(temp, key, self.callbacks[key](agent=agent, partialUpdate=temp))
                i += 1
            except Exception as e:  # sanity check
                agent.logger.critical(key + " " + str(e))

        # signal garbage collector
        temp.prevState.prevState = None

        # add a time stamp to state to indicate freshness
        temp.lastUpdate = time.time()

        return temp

    def getKeys(self):
        return [i for i in self.callbacks]

    def getKeyValuePairs(self):
        return {key: getattr(self, key) for key in self.getKeys()}
