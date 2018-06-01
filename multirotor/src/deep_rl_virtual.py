import json
import time
from operator import itemgetter
from itertools import groupby
from typing import Tuple, NamedTuple, List

import numpy
import numpy as np

from pyquaternion import Quaternion
from sklearn.externals import joblib

from tensorforce.agents import Agent
from tensorforce.execution import Runner

from utilities import StateT
from vector_math import transform_to_body_frame, to_euler_angles, distance, delta_heading_2d, unit, \
    integrate_trajectory_velocity_body, euler_to_quaternion, generate_random_point

# obstacle type
ObstacleT = NamedTuple('Obstacle', [('position', np.ndarray), ('distance', float), ('angle', float)])


class Simulator:
    ACTIONS = np.array([[1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]])

    def __init__(self, safe_distance: float, goal_margin: float, obstacles_view_radius: float, max_episode_length: int,
                 model, environment_size: float, number_obstacles: int, default_speed: float = 3.0, frequency: float = 10.0):
        """
        creates a simulator instance.
        :param safe_distance: minimum safe distance to obstacles.
        :param goal_margin: specifies margin within which agent is considered to have reached the goal.
        :param obstacles_view_radius: specifies max. distance to obstacles to be detected.
        :param max_episode_length: max episode length.
        :param model: model of environment.
        :param environment_size: radius of the environment.
        :param number_obstacles: number of obstacles to be generated.
        :param default_speed: multirotor default speed (depends on the given model).
        :param frequency: model frequency (depends on the given model).
        """
        self.frequency = frequency
        self.safe_distance = safe_distance
        self.goal_margin = goal_margin
        self.default_speed = default_speed
        self.obstacles_view_radius = obstacles_view_radius
        self.model = model
        self.environment_size = environment_size
        self.number_obstacles = number_obstacles

        self.max_episode_length = max_episode_length

        self.time_step = 0
        self.rays = np.linspace(-np.pi, np.pi, 13)
        self.goal, self.obstacles = None, None
        self.state = None
        self.episode = 0
        self.episode_total_reward = 0.0

    def generate_goal(self) -> numpy.ndarray:
        """
        generates goal.
        :return: new goal position np.array([x, y, z]).
        """
        return generate_random_point(radius=self.environment_size)

    def generate_obstacles(self, min_distance: float, number_obstacles: int) -> List[numpy.ndarray]:
        """
        generates number_obstacles obstacles that are min_distance apart from each other, obstacles are never at agent
        starting position.
        :param min_distance: minimum distance between any two obstacles.
        :param number_obstacles: number of obstacles to generate.
        :return:
        """
        obstacles = [np.array([float('inf'), float('inf'), float('inf')])]
        initial_position = np.array([0.0, 0.0, -1.0])

        for i in range(number_obstacles):
            # obstacle must be more than min_distance away from starting position AND
            # obstacle must be more than min_distance away from goal AND
            # obstacle must be more than min_distance away from any other obstacle.
            point = initial_position
            while distance(point, initial_position) < min_distance or \
                    distance(point, self.goal) < min_distance or \
                    min([*map(lambda obs: distance(point, obs), obstacles), float('inf')]) < min_distance:
                point = generate_random_point(radius=self.environment_size)

            obstacles.append(point)

        return obstacles

    def vectorize_state(self, state: StateT) -> np.ndarray:
        """state (22, ):
            inertial:
                linear velocity: in body-fixed frame, meter/second [0-2]
                angular velocity: in body-fixed frame, radians/second [3-5]

                roll: expressed in radians [6]
                pitch: expressed in radians [7]

            goal:
                delta heading: difference between agent's current heading and goal heading. expressed in radians [8].
                distance to goal: distance from current position to goal position, expressed in meters [9].

            obstacles_view:
                distance to obstacles_view: expressed in meters [10-21]
        """
        inertial = np.concatenate((state.linear_velocity_body,
                                   state.angular_velocity_body, to_euler_angles(state.orientation)[:2]))
        goal = [delta_heading_2d(state.position, state.orientation, self.goal), distance(state.position, self.goal)]

        return np.concatenate((inertial, goal, self.obstacles_view(state)))

    def reset(self) -> np.ndarray:
        """
        resets environment to initial state, generates a new goal position, and obstacles if specified in the config.
        :return:
        """
        try:
            terminal_state = self.vectorize_state(self.state)
            template = 'episode {ep} summary: heading error: {head}, remaining distance to goal: {dist}, ' \
                       'total reward: {r}\nobstacles view: {obs}\n----------\n'

            print(template.format(head=int(np.rad2deg(terminal_state[8])),
                                  dist=int(distance(self.state.position, self.goal)),
                                  r=int(self.episode_total_reward), ep=self.episode,
                                  obs=np.round(self.obstacles_view(self.state))))

        except AttributeError:
            pass

        # generate initial state
        initial_state = StateT(update=False, orientation=Quaternion(euler_to_quaternion(roll=0.0, pitch=0.0, yaw=0.0)),
                               linear_velocity_body=[0.0, 0.0, 0.0], angular_velocity_body=[0.0, 0.0, 0.0],
                               position=[0.0, 0.0, -1.0])

        # update current state
        self._update_state(initial_state)

        # generate goal
        self.goal = self.generate_goal()

        # generate obstacles
        # goal must be generated first!
        self.obstacles = self.generate_obstacles(min_distance=21.0, number_obstacles=self.number_obstacles)

        # increment episode
        # reset total reward for the new episode
        self.episode += 1
        self.episode_total_reward = 0.0

        return self.vectorize_state(initial_state)

    def obstacles_view(self, state) -> np.ndarray:
        """
        :return array of distances for nearest obstacles in view for each of the sensors.
        """
        num_bins = len(self.rays)
        angle_dist = dict(zip(range(num_bins), [self.obstacles_view_radius] * num_bins))

        # get obstacles in radius
        in_radius = filter(lambda obs: distance(obs, state.position) <= self.obstacles_view_radius,
                           self.obstacles)

        # get angle between heading and obstacles in radius.
        obs_dist_angle = \
            [*map(lambda obs: ObstacleT(position=obs, distance=distance(obs, state.position), angle=int(np.digitize(
                delta_heading_2d(state.position, state.orientation, obs), self.rays, right=True))), in_radius)]

        # sort and group by angle, using minimum distance if more than one obstacles lie on the same line.
        obs_dist_angle.sort(key=lambda x: x.angle)
        for key, group in groupby(obs_dist_angle, key=lambda x: x.angle):
            angle_dist[key] = min(angle_dist[key], min(map(itemgetter(1), group)))

        view = [angle_dist[i] for i in range(num_bins)]

        # merge pi and -pi into the same slot
        view[0] = min(view[0], view[-1])

        return view[:-1]

    def reward(self, state) -> float:
        """
        reward for a given state
        :param state: current state
        :return: reward is equal to the dot product of velocity and unit vector in direction of the goal
        (expressed in Body-fixed frame) and then divided by max velocity to normalize reward to be in the range [-1.0, 1.0],
        then r is clipped to be in [0.0, 1] and then translated to finally be in the range [-1, 0.0]

        if safe distance away from obstacles, then
        r = (unit_vector(R[Earth->Body].(Position_goal[Earth] - position)) . velocity[Body]) / max_velocity
        r = r.clip(0.0, 1.0) - 1.0

        else

        r = -max_episode_length
        """
        goal_body = transform_to_body_frame(self.goal - state.position, state.orientation)
        unit_displacement = np.dot(unit(goal_body), state.linear_velocity_body)

        if min(self.obstacles_view(state)) < self.safe_distance:
            return -2.0 * self.max_episode_length

        elif distance(self.goal, state.position) < self.goal_margin:
            return 0.0

        else:
            return np.clip(unit_displacement / self.default_speed, a_min=0.0, a_max=1.0) - 1.0

    def is_terminal(self, state) -> bool:
        """
        is_terminal
        :param state: current state
        :return: a boolean indicating whether given state is terminal or not
        """
        return min(self.obstacles_view(state)) < self.safe_distance or distance(self.goal, state.position) < self.goal_margin

    def generate_next_state(self, action) -> StateT:
        """
        generates next state given some action
        :param action: action to be applied to the current state
        :return: new state after applying given action
        """
        roll, pitch, _ = to_euler_angles(self.state.orientation)

        initial_state = np.concatenate((self.state.linear_velocity_body,
                                        self.state.angular_velocity_body, [roll, pitch],
                                        self.ACTIONS[action]))

        next_state = self.model.predict(initial_state.reshape(1, -1)).ravel()

        orientation, position, linear_velocity_body, angular_velocity_body = \
            next(integrate_trajectory_velocity_body(position=self.state.position,
                                                    orientation=self.state.orientation,
                                                    linear_velocities_body=[next_state[:3]],
                                                    angular_velocities_body=[next_state[3:6]],
                                                    frequency=[self.frequency]))

        return StateT(update=False, orientation=orientation, position=position, linear_velocity_body=linear_velocity_body,
                      angular_velocity_body=angular_velocity_body)

    def _update_state(self, state: StateT):
        self.state = state

    def execute(self, actions) -> Tuple[np.ndarray, float, bool]:
        """
        executes an action in the environment.
        :param actions: action to be applied.
        :return: an ordered tuple containing next_state, is_terminal, reward.
        """
        self.time_step += 1

        # perform action
        next_state = self.generate_next_state(actions)
        state_vector = self.vectorize_state(next_state)

        # update current_state
        self._update_state(next_state)

        # generate reward and is_terminal
        r = self.reward(next_state)
        self.episode_total_reward += r
        terminal = self.is_terminal(next_state)

        return state_vector, terminal, r

    @property
    def states_dim(self):
        return dict(shape=(22,), type='float')

    @property
    def actions_dim(self):
        return dict(type='int', num_actions=4)

    def close(self):
        print('session is closed!')


def main():
    environment = Simulator(model=joblib.load('models/nn-m.model'), frequency=10.0, safe_distance=8.0, goal_margin=8.0,
                            default_speed=3.0, obstacles_view_radius=30.0, max_episode_length=1200, environment_size=70.0,
                            number_obstacles=20)

    with open('network.json', 'r') as fp:
        network_spec = json.load(fp=fp)

    with open('config.json', 'r') as fp:
        config = json.load(fp=fp)

    agent = Agent.from_spec(
        spec=config,
        kwargs=dict(
            states=environment.states_dim,
            actions=environment.actions_dim,
            network=network_spec)
    )

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )

    def episode_finished(r):
        sps = r.timestep / (time.time() - r.start_time)
        print("Finished episode {ep} steps per second {sps}".format(ep=r.episode, sps=int(sps)))
        return True

    runner.run(
        timesteps=100000 * environment.max_episode_length,
        episodes=50000,
        max_episode_timesteps=environment.max_episode_length,
        deterministic=False,
        episode_finished=episode_finished
    )

    # agent.restore_model('saved/progress')

    terminal, state = False, environment.reset()
    while not terminal:
        action = agent.act(state)
        state, terminal, reward = environment.execute(action)

    runner.close()


if __name__ == '__main__':
    main()
