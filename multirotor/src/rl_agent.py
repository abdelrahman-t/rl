from abc import ABCMeta
from functools import partial
import pickle

from utilities import *
from airsim import *
from vector_math import transform_to_earth_frame
from pynput import keyboard


class RLAgent():
    __metaclass__ = ABCMeta

    def __init__(self, name, server_ip_address='127.0.0.1', default_speed=3, default_altitude=1.5,
                 yaw_rate=70, decision_frequency=10, crash_recovery_period=16):
        self._logger = create_console_logger(name)
        self._server_ip_address = server_ip_address
        self._yaw_rate, self._default_speed, self._default_altitude = yaw_rate, default_speed, default_altitude
        self._crash_recovery_period = crash_recovery_period

        self._decision_frequency = decision_frequency
        self._key_pressed = KeyT()
        self.keyboard_listener = keyboard.Listener(on_press=self.key_pressed.on_press,
                                                   on_release=self.key_pressed.on_release)
        self.keyboard_listener.start()

    def initialize(self):
        try:
            client = MultirotorClient()
            client.confirmConnection()
            self._client = client

        except Exception as e:
            self._logger.critical(e)
            sys.exit(1)

        self.actions = {'move_forward': self._move_forward, 'yaw_cw': partial(self._yaw, self._yaw_rate),
                        'yaw_ccw': partial(self._yaw, -self._yaw_rate), 'hover': self._move_by_velocity_z,
                        'move_right': self._move_right, 'move_left': self._move_left,
                        'move_backward': self._move_backward}

        time.sleep(self._crash_recovery_period)
        self.perform_action(partial(self._client.enableApiControl, True))
        self.perform_action(partial(self._client.armDisarm, True))
        self.perform_action(self._client.takeoff)
        self.update_state()

        # enable programmatic control of the multi-rotor and reset collision flag
        self._logger.info("Ready")

    def define_state(self, **kwargs):
        self._state = StateT(update=True, **kwargs)

    def define_goal(self, **kwargs):
        self._goal = StateT(update=False, **kwargs)

    @property
    def state(self):
        return self._state

    def update_state(self):
        s = self._state.update_state(self)

        # HACK!
        #################
        s.position = s.k.position.toNumpyArray()
        s.angular_velocity = s.k.angular_velocity.toNumpyArray()
        s.linear_velocity = s.k.linear_velocity.toNumpyArray()
        s.orientation = s.k.orientation.toNumpyArray()
        ###############

        self._state = s

    @property
    def actions(self):
        return self._actions

    @actions.setter
    def actions(self, a):
        self._actions = a

    def perform_action(self, action):
        try:
            action()
        except Exception as e:
            self._logger.critical(e)

    def _move_by_velocity_z(self, vx=0, vy=0, yaw_mode=YawMode(True, 0), drivetrain=DrivetrainType.MaxDegreeOfFreedom,
                            duration=10.0):

        self._client.moveByVelocityZ(vx=vx, vy=vy, z=-self._default_altitude, yaw_mode=yaw_mode, duration=duration,
                                     drivetrain=drivetrain)

    def _hover(self):
        self.perform_action(self._move_by_velocity_z())

    def _default_action(self):
        pass

    def _move_forward(self):
        velocity_body = np.array([self._default_speed, 0.0, 0.0])
        velocity_earth = transform_to_earth_frame(velocity_body, self.state.orientation)
        self.perform_action(partial(self._move_by_velocity_z, vx=velocity_earth[0], vy=velocity_earth[1]))

    def _move_backward(self):
        velocity_body = np.array([-self._default_speed, 0.0, 0.0])
        velocity_earth = transform_to_earth_frame(velocity_body, self.state.orientation)
        self.perform_action(partial(self._move_by_velocity_z, vx=velocity_earth[0], vy=velocity_earth[1]))

    def _move_right(self):
        velocity_body = np.array([0.0, self._default_speed, 0.0])
        velocity_earth = transform_to_earth_frame(velocity_body, self.state.orientation)
        self.perform_action(partial(self._move_by_velocity_z, vx=velocity_earth[0], vy=velocity_earth[1]))

    def _move_left(self):
        velocity_body = np.array([0.0, -self._default_speed, 0.0])
        velocity_earth = transform_to_earth_frame(velocity_body, self.state.orientation)
        self.perform_action(partial(self._move_by_velocity_z, vx=velocity_earth[0], vy=velocity_earth[1]))

    def _yaw(self, rate):
        self.perform_action(partial(self.actions['hover'], vx=0.0, vy=0.0, yaw_mode=YawMode(True, rate)))

    def set_rl(self, callback):
        f = partial(callback, agent=self)
        self._rl = f

    def set_reward(self, callback):
        f = partial(callback, agent=self)
        self._reward = f

    def set_terminal(self, f):
        self._is_terminal = f

    @property
    def key_pressed(self):
        return self._key_pressed

    def run(self):
        self.initialize()
        callback = self._rl()

        period = 1 / self._decision_frequency

        while True:
            start = time.time()

            # give turn to the agent
            a = next(callback)
            # perform action selected by the agent
            self.perform_action(self.actions[a])

            # delay to match agent's decision freq.
            while time.time() - start < period:
                continue

            self.update_state()
            s, r, is_terminal = self.state, self._reward(), self._is_terminal()

            # send agent the the transition reward, new state and isTerminal and wait until the agent yields (OK signal)
            next(callback)
            callback.send((r, s, is_terminal))

    def save_progress(self, progress, file_name, append=False):
        try:
            with open(file_name, 'a+b' if append else 'wb') as f:
                pickle.dump(progress, f, protocol=pickle.HIGHEST_PROTOCOL)

        except Exception as e:
            self._logger.critical(e)

        return time.time()

    def load_progress(self, file_name):
        last_check_point = None
        try:
            with open(file_name, 'rb') as handle:
                last_check_point = pickle.load(handle)
        except Exception as e:
            self._logger.critical(e)

        return last_check_point

    def _rl(self, *args, **kwargs):
        raise NotImplementedError

    def _reward(self, *args, **kwargs):
        raise NotImplementedError

    def _is_terminal(self):
        raise NotImplementedError
