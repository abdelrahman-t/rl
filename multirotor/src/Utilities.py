from Common import *
from VectorMath import *


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
    return ('{0:.%sf}' % decimalPlaces).format(number)


def getAverageRatesBody(df, limit, frequency):
    shape = (limit, 3)
    linearVelocities, linearAccelerations = np.zeros(shape), np.zeros(shape)
    angularVelocities, angularAccelerations = np.zeros(shape), np.zeros(shape)

    # get average linear velocity (EARTH)
    for i in range(1, limit):
        linearVelocities[i] = getAverageLinearVelocity(df.loc[i, ['x', 'y', 'z']].values,
                                                       df.loc[i - 1, ['x', 'y', 'z']].values,
                                                       frequency)

        angularVelocities[i] = getAverageAngularVelocity(df.loc[i, ['roll', 'pitch', 'yaw']].values,
                                                         df.loc[i - 1, ['roll', 'pitch', 'yaw']].values,
                                                         frequency)
    # get average acceleration (EARTH)
    for i in range(1, limit - 1):
        linearAccelerations[i] = \
            getAverageLinearAcceleration(linearVelocities[i + 1], linearVelocities[i], frequency)

        angularAccelerations[i] = \
            getAverageAngularAcceleration(angularVelocities[i + 1], angularVelocities[i], frequency)

    # transform rates to be in BODY FRAME
    for i in range(1, limit):
        q = df.loc[i, ['scalar', 'i', 'j', 'k']].values

        linearVelocities[i] = transformToBodyFrame(linearVelocities[i], q)
        linearAccelerations[i] = transformToBodyFrame(linearAccelerations[i], q)

        angularVelocities[i] = transformEulerRatesToBody(angularVelocities[i], q)
        angularAccelerations[i] = transformEulerRatesToBody(angularAccelerations[i], q)

    return linearVelocities[1:], angularVelocities[1:], linearAccelerations[1:], angularAccelerations[1:]


def getXyAccelerationModel(df, frequency, limit=500):
    v, w, a, alpha = getAverageRatesBody(df, limit, frequency=frequency)
    X, y = np.zeros((limit, 12)), np.zeros((limit, 6))

    actionNames = ['moveForward', 'yawCCW', 'yawCW', 'hover']
    for i, j, k in zip(range(limit - 2), range(1, limit - 2), range(2, limit - 2)):
        roll, pitch = df.loc[j, ['roll', 'pitch']].values
        selectedAction = [0 if a != df.loc[k, 'aName'] else 1 for a in actionNames]

        X[i] = np.concatenate((v[i], w[i], [roll, pitch], selectedAction))
        y[i] = np.concatenate((a[i], alpha[i]))

    xColumns = ['dXB', 'dYB', 'dZB', 'dRoll', 'dPitch', 'dYaw', 'sin(roll)', 'cos(roll)', 'sin(pitch)', 'cos(pitch)']\
             + [i for i in actionNames]
        
    yColumns = ['d2XB', 'd2YB', 'd2ZB', 'd2Roll', 'd2Pitch', 'd2Yaw']
    return pd.DataFrame(X, columns=xColumns), pd.DataFrame(y, columns=yColumns)


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
