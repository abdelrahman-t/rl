import json
import time
from operator import itemgetter
from itertools import groupby
from typing import Tuple, NamedTuple, List

import numpy
import numpy as np

from pyquaternion import Quaternion
from sklearn.externals import joblib

# loading model
from tensorforce.agents import Agent
from tensorforce.execution import Runner

from utilities import StateT
from vector_math import transform_to_body_frame, to_euler_angles, distance, delta_heading_2d, unit, \
    integrate_trajectory_velocity_body, euler_to_quaternion

# obstacle type
ObstacleT = NamedTuple('Obstacle', [('position', np.ndarray), ('distance', float), ('angle', float)])


class Simulator:
    ACTIONS = np.array([[1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]])

    def __init__(self, safe_distance: float, goal_margin: float, obstacles_view_radius: float, max_episode_length: int, model,
                 environment_size: float, number_obstacles: int, default_speed: float = 3.0, frequency: float = 10.0):
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
        self.bins = np.linspace(-np.pi, np.pi, 9)
        self.goal, self.obstacles = None, None
        self.state = None
        self.episode = 0
        self.episode_total_reward = 0.0

    def generate_random_point(self, radius: float):
        # from https://programming.guide/random-point-within-circle.html
        """
        generate random point.
        :param radius: generate a 3d point in a given radius.
        :return: returns a 3d point.
        """
        angle = np.random.rand() * 2 * np.pi
        r = radius * np.random.rand() ** 0.5
        return np.array([r * np.cos(angle), r * np.sin(angle), -1.0])

    def generate_goal(self) -> numpy.ndarray:
        """
        generates goal.
        :return: new goal position np.array([x, y, z]).
        """
        return self.generate_random_point(radius=self.environment_size)

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
        safe = self.safe_distance + self.default_speed

        for i in range(number_obstacles):
            # obstacle must be more than safe_distance away from starting position OR
            # obstacle must be more than min_distance away from goal OR
            # obstacle must be more than min_distance away from any other obstacle.
            point = initial_position
            while distance(point, initial_position) < safe or \
                    distance(point, self.goal) < safe or \
                    min([*map(lambda obs: distance(point, obs), obstacles), float('inf')]) < min_distance:
                point = self.generate_random_point(radius=self.environment_size)

            obstacles.append(point)

        return obstacles

    def vectorize_state(self, state: StateT) -> np.ndarray:
        """state (26, ):
            inertial:
                position: position of multirotor. [0-2]
                linear velocity: in body-fixed frame, meter/second [3-5]
                angular velocity: in body-fixed frame, radians/second [6-8]

                roll: expressed in radians [9]
                pitch: expressed in radians [10]
                yaw: expressed in radians [11]

            goal:
                delta heading: difference between agent's current heading and goal heading. expressed in radians [12].
                distance to goal: distance from current position to goal position, expressed in meters [13].
                goal position: goal position [14-16].

            obstacles_view:
                distance to obstacles_view: expressed in meters [17-26]
        """
        inertial = np.concatenate((state.position,
                                   transform_to_body_frame(state.linear_velocity, state.orientation),
                                   state.angular_velocity, to_euler_angles(state.orientation)))
        goal = np.concatenate(
            ([delta_heading_2d(state.position, state.orientation, self.goal), distance(state.position, self.goal)],
             self.goal))

        return np.concatenate((inertial, goal))

    def reset(self) -> np.ndarray:
        """
        resets environment to inital state, generates a new goal position, and obstacles if specified in the config.
        :return:
        """
        try:
            terminal_state = self.vectorize_state(self.state)
            print('episode: {ep} summary: heading error: {h}, distance to goal: {d}, total reward: {r}\n'
                  .format(h=int(np.rad2deg(terminal_state[12])), d=int(terminal_state[13]),
                          r=int(self.episode_total_reward), ep=self.episode))

        except AttributeError:
            pass

        # generate initial state
        initial_state = StateT(update=False, orientation=Quaternion(euler_to_quaternion(roll=0.0, pitch=0.0, yaw=0.0)),
                               position=[0.0, 0.0, -1.0], linear_velocity=[0.0, 0.0, 0.0], angular_velocity=[0.0, 0.0, 0.0])

        # update current state
        self._update_state(initial_state)

        # generate goal
        self.goal = self.generate_goal()

        # generate obstacles
        # goal must be generated first!
        self.obstacles = self.generate_obstacles(min_distance=16.0, number_obstacles=self.number_obstacles)

        # increment episode
        # reset total reward for the new episode
        self.episode += 1
        self.episode_total_reward = 0.0

        return self.vectorize_state(initial_state)

    def obstacles_view(self, state) -> np.ndarray:
        """
        :return array of distances for nearest obstacles in view for each of the sensors.
        """
        num_bins = len(self.bins)
        angle_dist = dict(zip(range(num_bins), [self.obstacles_view_radius] * num_bins))

        # get obstacles in radius
        in_radius = filter(lambda obs: distance(obs, state.position) <= self.obstacles_view_radius,
                           self.obstacles)

        # get angle between heading and obstacles in radius.
        obs_dist_angle = \
            [*map(lambda obs: ObstacleT(position=obs, distance=distance(obs, state.position), angle=int(np.digitize(
                delta_heading_2d(state.position, state.orientation, obs), self.bins, right=True))), in_radius)]

        # sort and group by angle, using minimum distance if more than one obstacles lie on the same line.
        obs_dist_angle.sort(key=lambda x: x.angle)
        for key, group in groupby(obs_dist_angle, key=lambda x: x.angle):
            angle_dist[key] = int(min(angle_dist[key], min(map(itemgetter(1), group))))

        return [angle_dist[i] for i in range(num_bins)]

    def reward(self, state) -> float:
        """
        reward for a given state
        :param state: current state
        :return: reward is equal to the dot product of velocity and unit vector in direction of the goal
        (expressed in Body-fixed frame) and then divided by max velocity to normalize reward to be in the range [-1.0, 1.0],
        then r is clipped to be in [0.0, 1] and then translated to finally be in the range [-1, 0.0]

        if safe distance away from obstacles, then
        r = (unit_vector(R[Earth->Body].(Position_goal[E] - position)) . velocity) / max_velocity
        r = r.clip(0.0, 1) - 1

        else

        r = -max_episode_length
        """
        goal_body = transform_to_body_frame(self.goal - state.position, state.orientation)
        unit_displacement = np.dot(unit(goal_body), transform_to_body_frame(state.linear_velocity, state.orientation))

        if min(self.obstacles_view(state)) < self.safe_distance:
            return -self.max_episode_length

        elif distance(self.goal, state.position) <= self.goal_margin:
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
        initial_state = np.concatenate((self.state.linear_velocity,
                                        self.state.angular_velocity, [roll, pitch],
                                        self.ACTIONS[action]))

        next_state = self.model.predict(initial_state.reshape(1, -1)).ravel()

        orientation, position, linear_velocity, angular_velocity = \
            next(integrate_trajectory_velocity_body(position=self.state.position,
                                                    orientation=self.state.orientation,
                                                    linear_velocities=[next_state[:3]],
                                                    angular_velocities=[next_state[3:6]],
                                                    frequency=[self.frequency]))

        return StateT(update=False, orientation=orientation, position=position, linear_velocity=linear_velocity,
                      angular_velocity=angular_velocity)

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
        return dict(shape=(17,), type='float')

    @property
    def actions_dim(self):
        return dict(type='int', num_actions=4)

    def close(self):
        print('session is closed!')


def main():
    environment = Simulator(model=joblib.load('models/nn-m.model'), frequency=10.0, safe_distance=8.0, goal_margin=8.0,
                            default_speed=3.0, obstacles_view_radius=30.0, max_episode_length=1000, environment_size=70.0,
                            number_obstacles=0)

    with open('network.json', 'r') as fp:
        network_spec = json.load(fp=fp)

    with open('config.json', 'r') as fp:
        config = json.load(fp=fp)

    with open('reward_preprocessing.json', 'r') as fp:
        reward_preprocessing = json.load(fp=fp)

    agent = Agent.from_spec(
        spec=config,
        kwargs=dict(
            states=environment.states_dim,
            actions=environment.actions_dim,
            network=network_spec,
            reward_preprocessing=reward_preprocessing)
    )

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )

    def episode_finished(r):
        sps = r.timestep / (time.time() - r.start_time)
        print("Finished episode {ep} Steps Per Second {sps}".format(ep=r.episode, sps=sps))
        print("Episode reward: {}".format(r.episode_rewards[-1]))
        print("Average of last 500 rewards: {}".format(sum(r.episode_rewards[-500:]) / 500))
        print("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
        print("----------\n")
        return True

    runner.run(
        timesteps=100000 * environment.max_episode_length,
        episodes=100000,
        max_episode_timesteps=environment.max_episode_length,
        deterministic=False,
        episode_finished=episode_finished
    )

    terminal, state = False, environment.reset()
    while not terminal:
        action = agent.act(state)
        state, terminal, reward = environment.execute(action)

    runner.close()


if __name__ == '__main__':
    main()
