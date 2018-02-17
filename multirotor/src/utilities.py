from vector_math import *
from collections import OrderedDict
from pyquaternion import Quaternion
from datetime import datetime

import numpy
import pandas as pd
import warnings
import logging
import time

warnings.filterwarnings("ignore")
GLOBAL_LOGGING_LEVEL = logging.INFO


class KeyT:
    def __init__(self):
        self.value = ''

    def on_press(self, key):
        self.value = str(key).replace('\'', '')

    def on_release(self, _):
        self.value = ''


def get_date_time():
    return datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


def create_console_logger(name, level=None):
    level = GLOBAL_LOGGING_LEVEL if level is None else level

    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(funcName)s @%(name)s : %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False

    return logger


def truncate_float(number, decimal_places):
    return ('{0:.%sf}' % decimal_places).format(number)


def get_average_rates_body(df, limit, frequency):
    shape = (limit, 3)
    linear_velocities, angular_velocities = numpy.zeros(shape), numpy.zeros(shape)

    # get average linear and angular velocities (EARTH)
    for i in range(1, limit):
        p1 = df.loc[i - 1, ['x', 'y', 'z']].values.astype(numpy.float64)
        p2 = df.loc[i, ['x', 'y', 'z']].values.astype(numpy.float64)

        q1 = Quaternion(df.loc[i - 1, ['scalar', 'i', 'j', 'k']].values)
        q2 = Quaternion(df.loc[i, ['scalar', 'i', 'j', 'k']].values)

        linear_velocities[i] = get_average_linear_velocity(p2, p1, frequency)

        axis, angle = get_average_angular_velocity(q2, q1, frequency)
        angular_velocities[i] = axis * angle

    return linear_velocities, angular_velocities


def get_xy_velocity_model(df, frequency, limit=500):
    v, w, = get_average_rates_body(df, limit, frequency=frequency)

    input_shape = (limit, 15)
    output_shape = (limit, 6)

    X, y = numpy.zeros(input_shape), numpy.zeros(output_shape)
    action_names = ['move_forward', 'yaw_ccw', 'yaw_cw', 'hover', 'move_backward', 'move_left', 'move_right']

    limit = limit - 2
    for t0, t1 in zip(range(limit), range(1, limit)):
        selected_action = [0 if a != df.loc[t1, 'aName'] else 1 for a in action_names]

        q = Quaternion(df.loc[t0, ['scalar', 'i', 'j', 'k']].values)
        roll, pitch, yaw = to_euler_angles(q)

        X[t0] = numpy.concatenate((transform_to_body_frame(v[t0], q),
                                   transform_euler_rates_to_body(w[t0], q),
                                   [roll, pitch], selected_action))

        y[t0] = numpy.concatenate((transform_to_body_frame(v[t1], q),
                                   transform_euler_rates_to_body(w[t1], q)))

    x_columns = ['dXB', 'dYB', 'dZB', 'dRoll', 'dPitch', 'dYaw', 'roll', 'pitch'] + [i for i in action_names]
    y_columns = ['dXB', 'dYB', 'dZB', 'dRoll', 'dPitch', 'dYaw']

    return pd.DataFrame(X[:-5], columns=x_columns), pd.DataFrame(y[:-5], columns=y_columns)


class StateT:
    # if update is set to True, the environment will be able to update state whenever is needed.
    def __init__(self, update, **kwargs):
        self.prevState = None

        if 'callbacks' in kwargs:
            self.callbacks = kwargs['callbacks']
        else:
            self.callbacks = OrderedDict()
            for key in kwargs:
                self.callbacks[key] = kwargs[key]
                setattr(self, key, None if update is True else kwargs[key])

    def are_equal(self, state, margins):
        for key in state.callbacks:
            diff = abs(getattr(self, key) - getattr(state, key))
            if (numpy.greater(diff, getattr(margins, key))).any():
                return False
        return True

    # States are immutable by design, each update creates a new StateT object
    def update_state(self, agent):
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
                raise Exception(key + " " + str(e))

        # signal garbage collector
        temp.prevState.prevState = None

        # add a time stamp to state to indicate freshness
        temp.lastUpdate = time.time()

        return temp

    def get_keys(self):
        return [i for i in self.callbacks]

    def get_key_value_pairs(self):
        return {key: getattr(self, key) for key in self.get_keys()}

    def __hash__(self):
        return hash(tuple(tuple(getattr(self, c)) for c in self.get_keys()))
