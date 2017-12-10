from Common import *


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


def transformToBodyFrame(vector, q):
    q_ = Quaternion(q)
    return q_.inverse.rotate(vector)


def transformBodyRatesToEarth(rates, euler):
    roll, pitch, yaw = euler
    roll_rate_body, pitch_rate_body, yaw_rate_body = rates

    roll_rate_earth = roll_rate_body + pitch_rate_body * np.sin(roll) * np.tan(pitch) + yaw_rate_body * np.cos(
        roll) * np.tan(pitch)
    pitch_rate_earth = pitch_rate_body * np.cos(roll) - yaw_rate_body * np.sin(roll)
    yaw_rate_earth = pitch_rate_body * np.sin(roll) / np.cos(pitch) + yaw_rate_body * np.cos(roll) / np.cos(pitch)

    return np.array([roll_rate_earth, pitch_rate_earth, yaw_rate_earth])


def transformEulerRatesToBody(rates, euler):
    roll, pitch, yaw = euler
    roll_rate_earth, pitch_rate_earth, yaw_rate_earth = rates

    roll_rate_body = roll_rate_earth - yaw_rate_earth * np.sin(pitch)
    pitch_rate_body = pitch_rate_earth * np.cos(roll) + yaw_rate_earth * np.sin(roll) * np.cos(pitch)
    yaw_rate_body = pitch_rate_earth * -np.sin(roll) + yaw_rate_earth * np.cos(roll) * np.cos(pitch)

    return np.array([roll_rate_body, pitch_rate_body, yaw_rate_body])


def getGravityVector(roll, pitch):
    x = np.cos(pitch) * np.sin(roll)
    y = -np.sin(pitch)
    z = -np.cos(pitch) * np.cos(roll)

    return np.array([x, y, z])


def wrapAngleAroundPi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def getAverage(m1, m0, frequency, wrap=lambda _: _):
    average = list(map(wrap, (m1 - m0)))
    return np.array(average) * frequency


def getAverageLinearVelocity(position1, position0, frequency):
    return getAverage(m1=position1, m0=position0, frequency=frequency)


def getAverageLinearAcceleration(linearVelocity1, linearVelocity0, frequency):
    return getAverage(m1=linearVelocity1, m0=linearVelocity0, frequency=frequency)


def getAverageAngularAcceleration(angularVelocity1, angularVelocity0, frequency):
    return getAverage(m1=angularVelocity1, m0=angularVelocity0, frequency=frequency)


def getAverageAngularVelocity(angularPosition1, angularPosition0, frequency):
    return getAverage(m1=angularPosition1, m0=angularPosition0, frequency=frequency, wrap=wrapAngleAroundPi)


def integrate(initial, rate, frequency, wrap=lambda _: _):
    integral = list(map(wrap, initial + rate / frequency))
    return np.array(integral)


def integrateAngularVelocity(initialAngularVelocity, angularAcceleration, frequency):
    return integrate(initial=initialAngularVelocity, rate=angularAcceleration, frequency=frequency)


def integrateLinearVelocity(initialLinearVelocity, linearAcceleration, frequency):
    return integrate(initial=initialLinearVelocity, rate=linearAcceleration, frequency=frequency)


def integratePosition(initialPosition, linearVelocityEarth, frequency):
    return integrate(initial=initialPosition, rate=linearVelocityEarth, frequency=frequency)


def integrateOrientation(initialOrientation, eulerRates, frequency):
    return integrate(initial=initialOrientation, rate=eulerRates, frequency=frequency, wrap=wrapAngleAroundPi)


def integrateTrajectoryAccelerationBody(initialPosition, initialOrientation, initialLinearVelocityBody,
                                        initialBodyRates, linearAccelerationsBody, angularAccelerationsBody,
                                        frequency):

    for aBody, alphaBody, f in zip(linearAccelerationsBody, angularAccelerationsBody, frequency):
        eulerRates = transformBodyRatesToEarth(
            integrateAngularVelocity(initialBodyRates, alphaBody, f), initialOrientation)

        nextOrientation = integrateOrientation(initialOrientation, eulerRates, f)

        initialQ, nextQ = eulerToQuaternion(*initialOrientation), eulerToQuaternion(*nextOrientation)

        linearVelocityEarth = transformToEarthFrame(integrateLinearVelocity(initialLinearVelocityBody, aBody, f),
                                                    initialQ)

        initialLinearVelocityBody = transformToBodyFrame(linearVelocityEarth, nextQ)
        initialBodyRates = transformEulerRatesToBody(eulerRates, nextOrientation)

        initialPosition = integratePosition(initialPosition, linearVelocityEarth, f)
        initialOrientation = nextOrientation

        yield initialPosition, initialOrientation, initialLinearVelocityBody, initialBodyRates
