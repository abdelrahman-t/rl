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
    return q_.rotate(vector)


def transformAngularRateToEarth(rates, q):
    roll, pitch, yaw = toEulerianAngle(Quaternion(q))
    roll_rate_body, pitch_rate_body, yaw_rate_body = rates
    roll_rate_earth = roll_rate_body + pitch_rate_body * np.sin(roll) * np.tan(pitch) + yaw_rate_body * np.cos(
        roll) * np.tan(pitch)
    pitch_rate_earth = pitch_rate_body * np.cos(roll) - yaw_rate_body * np.sin(roll)
    yaw_rate_earth = pitch_rate_body * np.sin(roll) / np.cos(pitch) + yaw_rate_body * np.cos(roll) / np.cos(pitch)

    return np.array([roll_rate_earth, pitch_rate_earth, yaw_rate_earth])


def transformAngularRateToBody(rates, q):
    roll, pitch, yaw = toEulerianAngle(Quaternion(q))
    roll_rate_earth, pitch_rate_earth, yaw_rate_earth = rates
    roll_rate_body = roll_rate_earth - yaw_rate_earth * np.sin(pitch)
    pitch_rate_body = pitch_rate_earth * np.cos(roll) + yaw_rate_earth * np.sin(roll) * np.cos(pitch)
    yaw_rate_body = pitch_rate_earth * -np.sin(roll) + yaw_rate_earth * np.cos(roll) * np.cos(pitch)

    return np.array([roll_rate_body, pitch_rate_body, yaw_rate_body])


def transformToBodyFrame(vector, q):
    q_ = Quaternion(q)
    return q_.inverse.rotate(vector)


def getGravityVector(roll, pitch):
    x = np.cos(pitch) * np.sin(roll)
    y = -np.sin(pitch)
    z = -np.cos(pitch) * np.cos(roll)

    return np.array([x, y, z])


def wrapAroundPi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def integrateOrientation(euler, angularVelocity, f):
    return np.array(list(map(wrapAroundPi, euler + angularVelocity / f)))


def integratePosition(p0, v, f):
    return p0 + v / f


def integrateAngularVelocity(w, alpha, f):
    return np.array(list(map(wrapAroundPi, w + alpha / f)))


def integrateLinearVelocity(v, a, f):
    return v + a / f


def downSample(originalFrequency, sampleFrequency, df):
    g = df.groupby(df.index // int(originalFrequency / sampleFrequency))

    d = (g[[i for i in df.columns if i.startswith('d')]].sum() / originalFrequency) * sampleFrequency
    p = g[[i for i in df.columns if i.startswith('d') is False
           and i not in {'t', 'f', 's', 'aIndex', 'aName'}]].last()

    a = g['aIndex'].agg(lambda x: x.value_counts().index[0])

    downSampled = reduce(lambda df1, df2: pd.merge(df1, df2, right_index=True,
                                                   left_index=True), [p, d, a.to_frame()])
    downSampled['f'] = sampleFrequency
    return downSampled


def getAverageAngularVelocity(p1, p0, f):
    return np.array(list(map(wrapAroundPi, p1 - p0))) * f


def getAverageLinearVelocity(p1, p0, f):
    return np.array((p1 - p0) * f)


def getAverageLinearAcceleration(v1, v0, f):
    return np.array((v1 - v0) * f)


def getAverageAngularAcceleration(v1, v0, f):
    return np.array((v1 - v0) * f)


def getAveragesBody(df, limit, frequency):
    shape = (limit, 3)
    linearVelocities, linearAccelerations = np.zeros(shape), np.zeros(shape)
    angularVelocities, angularAccelerations = np.zeros(shape), np.zeros(shape)

    for i in range(1, limit):
        linearVelocities[i] = getAverageLinearVelocity(df.loc[i, ['x', 'y', 'z']].values,
                                                       df.loc[i - 1, ['x', 'y', 'z']].values,
                                                       frequency)

        angularVelocities[i] = getAverageAngularVelocity(df.loc[i, ['roll', 'pitch', 'yaw']].values,
                                                         df.loc[i - 1, ['roll', 'pitch', 'yaw']].values,
                                                         frequency)

    for i in range(1, limit - 1):
        linearAccelerations[i] = \
            getAverageLinearAcceleration(linearVelocities[i + 1], linearVelocities[i], frequency)

        angularAccelerations[i] = \
            getAverageAngularAcceleration(angularVelocities[i + 1], angularVelocities[i], frequency)

    for i in range(1, limit):
        q = df.loc[i, ['scalar', 'i', 'j', 'k']].values

        linearVelocities[i] = \
            transformToBodyFrame(linearVelocities[i], q)

        linearAccelerations[i] = \
            transformToBodyFrame(linearAccelerations[i], q)

        angularVelocities[i] = \
            transformAngularRateToBody(angularVelocities[i], q)

        angularAccelerations[i] = \
            transformAngularRateToBody(angularAccelerations[i], q)

    return linearVelocities[1:], angularVelocities[1:], linearAccelerations[1:], angularAccelerations[1:]


def getInputOutputAccelerationModel(df, frequency, limit=500):
    v, w, a, alpha = getAveragesBody(df, limit, frequency=frequency)
    X, y = np.zeros((limit, 12)), np.zeros((limit, 6))

    inverseKeymap = ['moveForward', 'yawCCW', 'yawCW', 'hover']

    for i in range(limit - 2):
        rowi, action = df.iloc[i + 1], df.loc[i + 2, 'aName']
        X[i] = np.concatenate(
            (v[i], w[i], [rowi['roll']], [rowi['pitch']], [0 if a != action else 1 for a in inverseKeymap]))
        y[i] = np.concatenate((a[i], alpha[i]))

    xColumns = ['dXB', 'dYB', 'dZB', 'dRoll', 'dPitch', 'dYaw', 'roll', 'pitch'] + [i for i in inverseKeymap]
    yColumns = ['d2XB', 'd2YB', 'd2ZB', 'd2Roll', 'd2Pitch', 'd2Yaw']
    X, y = pd.DataFrame(X, columns=xColumns), \
           pd.DataFrame(y, columns=yColumns)

    return X, y


def integrateTrajectoryAccelerationBody(initialPosition, initialOrientation, initialLinearVelocityBody,
                                        initialAngularVelocityBody, linearAccelerationsBody, angularAccelerationsBody,
                                        frequency):
    for a, alpha, f in zip(linearAccelerationsBody, angularAccelerationsBody, frequency):
        initialAngularVelocityBody = integrateAngularVelocity(initialAngularVelocityBody, alpha, f)
        initialLinearVelocityBody = integrateLinearVelocity(initialLinearVelocityBody, a, f)

        qInitial = eulerToQuaternion(*initialOrientation)
        nextOrientation = integrateOrientation(initialOrientation,
                                               transformAngularRateToEarth(initialAngularVelocityBody, qInitial), f)

        qNext = eulerToQuaternion(*nextOrientation)
        initialPosition = integratePosition(initialPosition,
                                            transformToEarthFrame(initialLinearVelocityBody, qInitial), f)

        initialAngularVelocityBody = \
            transformAngularRateToBody(transformAngularRateToEarth(initialAngularVelocityBody, qInitial), qNext)

        initialLinearVelocityBody = (qNext.inverse * qInitial).rotate(initialLinearVelocityBody)
        initialOrientation = nextOrientation

        yield initialPosition, initialOrientation


class StateT:
    # if update is set to True, the environment will be able to update state whenever is needed.
    def __init__(self, update, **kwargs):
        self.prevState = None

        if 'callbacks' in kwargs:
            self.callbacks = kwargs['callbacks']
        elif 'state' in kwargs:
            for key, value in kwargs['state'].getKeyValuePairs.items():
                setattr(self, key, value)
        else:
            self.callbacks = OrderedDict()
            for key in kwargs:
                self.callbacks[key] = kwargs[key]
                setattr(self, key, None if update is True else kwargs[key])

    def areEqual(self, state, margins):
        for key in state.callbacks:
            diff = abs(getattr(self, key) - getattr(state, key))
            if (np.greater(diff, getattr(margins, key))).any():
                return False
        return True

    # States are immutable by design, each update creates a new StateT object
    def updateState(self, agent):
        temp = StateT(update=True, callbacks=self.callbacks)
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
