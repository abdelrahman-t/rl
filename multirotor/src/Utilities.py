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
    linearVelocities,  angularVelocities = np.zeros(shape), np.zeros(shape)

    # get average linear and angular velocities (EARTH)
    for i in range(1, limit):
        p1 = df.loc[i-1, ['x', 'y', 'z']].values.astype(np.float64)
        p2 = df.loc[i, ['x', 'y', 'z']].values.astype(np.float64)
        
        q1 = Quaternion(df.loc[i-1, ['scalar', 'i', 'j', 'k']].values)
        q2 = Quaternion(df.loc[i, ['scalar', 'i', 'j', 'k']].values)
        
        v = getAverageLinearVelocity(p2, p1, frequency)
        linearVelocities[i] = v
        
        axis, angle = getAverageAngularVelocity(q2, q1, frequency)
        angularVelocities[i] = axis * angle
        
        # sanity check
        #print(np.rad2deg(toEulerianAngle(integrateOrientation(q1, axis*angle, frequency))), np.rad2deg(toEulerianAngle(q2)))
        #assert integrateOrientation(q1, axis*angle, frequency) == q2
        #assert np.allclose(integratePosition(p1, v, frequency), p2)
        
    return linearVelocities, angularVelocities


def getXyVelocityModel(df, frequency, limit=500):
    v, w, = getAverageRatesBody(df, limit, frequency=frequency)
    
    input_shape = (limit, 12)
    output_shape = (limit, 6)
    
    X, y = np.zeros(input_shape), np.zeros(output_shape)
    actionNames = ['moveForward', 'yawCCW', 'yawCW', 'hover']
    
    limit = limit - 2
    for t0, t1 in zip(range(limit), range(1, limit)):
        selectedAction = [0 if a != df.loc[t1, 'aName'] else 1 for a in actionNames]
        
        q = Quaternion(df.loc[t0, ['scalar', 'i', 'j', 'k']].values)
        roll, pitch, yaw = toEulerianAngle(q)
        
        X[t0] = np.concatenate((transformToBodyFrame(v[t0], q), transformEulerRatesToBody(w[t0], q), [roll, pitch], selectedAction))
        y[t0] = np.concatenate((transformToBodyFrame(v[t1], q), transformEulerRatesToBody(w[t1], q)))

    xColumns = ['dXB', 'dYB', 'dZB', 'dRoll', 'dPitch', 'dYaw', 'roll', 'pitch']\
             + [i for i in actionNames]
        
    yColumns = ['dXB', 'dYB', 'dZB', 'dRoll', 'dPitch', 'dYaw']
    
    return pd.DataFrame(X[:-5], columns=xColumns), pd.DataFrame(y[:-5], columns=yColumns)


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
