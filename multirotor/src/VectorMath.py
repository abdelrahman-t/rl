from Common import *


# -------------
# euler quaternion conversions
# -------------

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


# -------------
# inertial, non-inertial frame transformations
# -------------

def transformToEarthFrame(vector, q):
    _q = Quaternion(q)
    return _q.rotate(vector)


def transformToBodyFrame(vector, q):
    _q = Quaternion(q)
    return _q.inverse.rotate(vector)


def transformBodyRatesToEarth(rates, q):
    roll, pitch, yaw = toEulerianAngle(q)
    roll_rate_body, pitch_rate_body, yaw_rate_body = rates

    roll_rate_earth = roll_rate_body + pitch_rate_body * np.sin(roll) * np.tan(pitch) + yaw_rate_body * np.cos(
        roll) * np.tan(pitch)
    pitch_rate_earth = pitch_rate_body * np.cos(roll) - yaw_rate_body * np.sin(roll)
    yaw_rate_earth = pitch_rate_body * np.sin(roll) / np.cos(pitch) + yaw_rate_body * np.cos(roll) / np.cos(pitch)

    return np.array([roll_rate_earth, pitch_rate_earth, yaw_rate_earth])


def transformEulerRatesToBody(rates, q):
    roll, pitch, yaw = toEulerianAngle(q)
    roll_rate_earth, pitch_rate_earth, yaw_rate_earth = rates

    roll_rate_body = roll_rate_earth - yaw_rate_earth * np.sin(pitch)
    pitch_rate_body = pitch_rate_earth * np.cos(roll) + yaw_rate_earth * np.sin(roll) * np.cos(pitch)
    yaw_rate_body = pitch_rate_earth * -np.sin(roll) + yaw_rate_earth * np.cos(roll) * np.cos(pitch)

    return np.array([roll_rate_body, pitch_rate_body, yaw_rate_body])


# -------------
# get average rates
# -------------

def getAverage(m1, m0, frequency, wrap=lambda _: _):
    average = list(map(wrap, (m1 - m0)))
    return np.array(average) * frequency


def getAverageLinearVelocity(position1, position0, frequency):
    return getAverage(m1=position1, m0=position0, frequency=frequency)


def getAverageAngularVelocity(q2, q1, frequency):
    qDiff = q1.inverse * q2
    axis, angle = qDiff.axis, qDiff.angle
    return axis, angle * frequency


# -------------
# integration
# -------------

def integrate(initial, rate, frequency, wrap=lambda _: _):
    integral = map(wrap, np.array(initial + rate / frequency))
    return np.array(list(integral))


def getAxisAngle(vector):
    x, y, z = vector
    magnitude = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    return vector / magnitude, magnitude


def integrateOrientation(q1, angularVelocity, frequency):
    q2 = Quaternion(q1)
    q2.integrate(angularVelocity, 1 / frequency)

    return q2


def integratePosition(initialPosition, linearVelocityEarth, frequency):
    return integrate(initial=initialPosition, rate=linearVelocityEarth, frequency=frequency)


# -------------
# preprocessing
# -------------

def wrapAngleAroundPi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def preprocessAngles(angle):
    return np.array([np.sin(angle), np.cos(angle)])


# -------------
# integrate trajectory from accelerations
# -------------
# position is (x, y, z)
# orientation is quaternion (w, x, y, z)

# linear velocity is [Vx, Vy, Vz] in body
# linear acceleration is [Ax, Ay, Az] in body

# angular velocity is [Wx, Wy Wz]
# angular acceleration is [ALPHAx, ALPHAy, ALPHAz]
# -------------

def integrateTrajectoryVelocityBody(position, orientation, linearVelocities, angularVelocities, frequency):
    for v, w, f in zip(linearVelocities, angularVelocities, frequency):
        eulerRates = transformBodyRatesToEarth(w, orientation)
        linearVelocityEarth = transformToEarthFrame(v, orientation)

        newOrientation = integrateOrientation(orientation, eulerRates, f)
        position = integratePosition(position, linearVelocityEarth, f)

        yield (newOrientation,
               position,
               transformToBodyFrame(linearVelocityEarth, newOrientation),
               transformEulerRatesToBody(eulerRates, newOrientation))

        orientation = newOrientation
