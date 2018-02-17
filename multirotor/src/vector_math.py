import numpy
from pyquaternion import Quaternion


# -------------
# euler quaternion conversions
# -------------

def to_euler_angles(q):
    ysqr = q[2] * q[2]

    t0 = +2.0 * (q[0] * q[1] + q[2] * q[3])
    t1 = +1.0 - 2.0 * (q[1] * q[1] + ysqr)
    roll = numpy.arctan2(t0, t1)

    t2 = +2.0 * (q[0] * q[2] - q[3] * q[1])
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = numpy.arcsin(t2)

    t3 = +2.0 * (q[0] * q[3] + q[1] * q[2])
    t4 = +1.0 - 2.0 * (ysqr + q[3] * q[3])
    yaw = numpy.arctan2(t3, t4)

    return numpy.array([roll, pitch, yaw])


def euler_to_quaternion(roll, pitch, yaw):
    quaternion = [0] * 4
    cosPhi_2 = numpy.cos(roll / 2)
    sinPhi_2 = numpy.sin(roll / 2)
    cosTheta_2 = numpy.cos(pitch / 2)
    sinTheta_2 = numpy.sin(pitch / 2)
    cosPsi_2 = numpy.cos(yaw / 2)
    sinPsi_2 = numpy.sin(yaw / 2)

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

def transform_to_earth_frame(vector, q):
    _q = Quaternion(q)
    return _q.rotate(vector)


def transform_to_body_frame(vector, q):
    _q = Quaternion(q)
    return _q.inverse.rotate(vector)


def transform_body_rates_to_earth(rates, q):
    roll, pitch, yaw = to_euler_angles(q)
    roll_rate_body, pitch_rate_body, yaw_rate_body = rates

    roll_rate_earth = roll_rate_body + pitch_rate_body * numpy.sin(roll) * numpy.tan(pitch) + yaw_rate_body * numpy.cos(
        roll) * numpy.tan(pitch)
    pitch_rate_earth = pitch_rate_body * numpy.cos(roll) - yaw_rate_body * numpy.sin(roll)
    yaw_rate_earth = pitch_rate_body * numpy.sin(roll) / numpy.cos(pitch) + yaw_rate_body * numpy.cos(roll) / numpy.cos(
        pitch)

    return numpy.array([roll_rate_earth, pitch_rate_earth, yaw_rate_earth])


def transform_euler_rates_to_body(rates, q):
    roll, pitch, yaw = to_euler_angles(q)
    roll_rate_earth, pitch_rate_earth, yaw_rate_earth = rates

    roll_rate_body = roll_rate_earth - yaw_rate_earth * numpy.sin(pitch)
    pitch_rate_body = pitch_rate_earth * numpy.cos(roll) + yaw_rate_earth * numpy.sin(roll) * numpy.cos(pitch)
    yaw_rate_body = pitch_rate_earth * -numpy.sin(roll) + yaw_rate_earth * numpy.cos(roll) * numpy.cos(pitch)

    return numpy.array([roll_rate_body, pitch_rate_body, yaw_rate_body])


# -------------
# get average rates
# -------------

def get_average(m1, m0, frequency, wrap=lambda _: _):
    average = list(map(wrap, (m1 - m0)))
    return numpy.array(average) * frequency


def get_average_linear_velocity(position1, position0, frequency):
    return get_average(m1=position1, m0=position0, frequency=frequency)


def get_average_angular_velocity(q2, q1, frequency):
    qDiff = q1.inverse * q2
    axis, angle = qDiff.axis, qDiff.angle
    return axis, angle * frequency


# -------------
# integration
# -------------

def integrate(initial, rate, frequency, wrap=lambda _: _):
    integral = map(wrap, numpy.array(initial + rate / frequency))
    return numpy.array(list(integral))


def get_axis_angle(vector):
    x, y, z = vector
    magnitude = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    return vector / magnitude, magnitude


def integrate_orientation(q1, angular_velocity, frequency):
    q2 = Quaternion(q1)
    q2.integrate(angular_velocity, 1 / frequency)

    return q2


def integrate_position(initial_position, linear_velocity_earth, frequency):
    return integrate(initial=initial_position, rate=linear_velocity_earth, frequency=frequency)


# -------------
# preprocessing
# -------------

def wrap_around_pi(angle):
    return numpy.arctan2(numpy.sin(angle), numpy.cos(angle))


def preprocess_angles(angle):
    return numpy.array([numpy.sin(angle), numpy.cos(angle)])


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

def integrate_trajectory_velocity_body(position, orientation, linear_velocities, angular_velocities, frequency):
    for v, w, f in zip(linear_velocities, angular_velocities, frequency):
        euler_rates = transform_body_rates_to_earth(w, orientation)
        linear_velocity_earth = transform_to_earth_frame(v, orientation)

        new_orientation = integrate_orientation(orientation, euler_rates, f)
        position = integrate_position(position, linear_velocity_earth, f)

        yield (new_orientation,
               position,
               transform_to_body_frame(linear_velocity_earth, new_orientation),
               transform_euler_rates_to_body(euler_rates, new_orientation))

        orientation = new_orientation
