import json
import time
from _operator import itemgetter
from itertools import groupby
from typing import Tuple, NamedTuple

import numpy as np
from keras import Sequential
from keras.layers import Flatten, Dense, Activation
from keras.optimizers import Adam
from pyquaternion import Quaternion
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from sklearn.externals import joblib

# loading model
from tensorforce.agents import Agent
from tensorforce.execution import Runner

from utilities import StateT
from vector_math import transform_to_body_frame, to_euler_angles, distance, delta_heading_2d, unit, \
    integrate_trajectory_velocity_body, euler_to_quaternion

# obstacle type
Obstacle = NamedTuple('Obstacle', [('position', np.ndarray), ('distance', float), ('angle', float)])


class Simulator:
    ACTIONS = np.array([[1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]])

    def __init__(self, frequency: float, safe_distance: float, goal_margin: float, default_speed: float, radius: float,
                 episode_length: int, model, environment_size: float):
        self.frequency = frequency
        self.safe_distance = safe_distance
        self.goal_margin = goal_margin
        self.default_speed = default_speed
        self.radius = radius
        self.model = model
        self.environment_size = environment_size

        self.episode_length = episode_length

        self.time_step = 0
        self.bins = np.linspace(-np.pi, np.pi, 9)
        self.goal, self.obstacles = None, None
        self.state = None
        self.episode = 0
        self.episode_total_reward = 0.0

    def generate_random_point(self, radius: float):
        # from https://programming.guide/random-point-within-circle.html
        angle = np.random.rand() * 2 * np.pi
        r = radius * np.random.rand() ** 0.5
        return np.array([r * np.cos(angle), r * np.sin(angle), -1.0])

    def generate_goal(self):
        return self.generate_random_point(radius=self.environment_size)

    def generate_obstacles(self, min_distance: float, number_obstacles: int):
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
        try:
            terminal_state = self.vectorize_state(self.state)
            print('episode: {ep}, heading: {h}, goal: {d}, rewards: {r}\n'
                  .format(h=int(np.rad2deg(terminal_state[12])), d=int(terminal_state[13]),
                          r=int(self.episode_total_reward), ep=self.episode))

        except AttributeError:
            pass

        # generate initial state
        initial_state = StateT(update=False, orientation=Quaternion(euler_to_quaternion(roll=0.0, pitch=0.0, yaw=0.0)),
                               position=[0.0, 0.0, -1.0], linear_velocity=[0.0, 0.0, 0.0], angular_velocity=[0.0, 0.0, 0.0])

        # update current state
        self.update_state(initial_state)

        # generate goal
        self.goal = self.generate_goal()

        # generate obstacles
        # goal must be generated first!
        self.obstacles = self.generate_obstacles(min_distance=16.0, number_obstacles=0)
        self.episode += 1
        self.episode_total_reward = 0.0

        return self.vectorize_state(initial_state)

    def obstacles_view(self, state) -> np.ndarray:
        """
        :return array of distances for nearest obstacles in view for each of the sensors.
        """
        num_bins = len(self.bins)
        angle_dist = dict(zip(range(num_bins), [self.radius] * num_bins))

        # get obstacles in radius
        in_radius = filter(lambda obs: distance(obs, state.position) <= self.radius,
                           self.obstacles)

        # get angle between heading and obstacles in radius.
        obs_dist_angle = \
            [*map(lambda obs: Obstacle(position=obs, distance=distance(obs, state.position), angle=int(np.digitize(
                delta_heading_2d(state.position, state.orientation, obs), self.bins, right=True))), in_radius)]

        # sort and group by angle, using minimum distance if more than one obstacles lie on the same line.
        obs_dist_angle.sort(key=lambda x: x.angle)
        for key, group in groupby(obs_dist_angle, key=lambda x: x.angle):
            angle_dist[key] = int(min(angle_dist[key], min(map(itemgetter(1), group))))

        return [angle_dist[i] for i in range(num_bins)]

    def reward(self, state) -> float:
        goal_body = transform_to_body_frame(self.goal - state.position, state.orientation)
        unit_displacement = np.dot(unit(goal_body), transform_to_body_frame(state.linear_velocity, state.orientation))

        if min(self.obstacles_view(state)) < self.safe_distance:
            return -self.episode_length

        elif distance(self.goal, state.position) <= self.goal_margin:
            return 0.0

        else:
            return np.clip(unit_displacement / self.default_speed, a_min=0.0, a_max=1.0) - 1.0

    def is_terminal(self, state) -> bool:
        return min(self.obstacles_view(state)) < self.safe_distance or distance(self.goal, state.position) < self.goal_margin

    def generate_next_state(self, action):
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

    def update_state(self, state: StateT):
        self.state = state

    def execute(self, actions) -> Tuple[np.ndarray, float, bool]:
        self.time_step += 1

        # perform action
        next_state = self.generate_next_state(actions)
        state_vector = self.vectorize_state(next_state)

        # update current_state
        self.update_state(next_state)

        # generate reward and is_terminal
        r = self.reward(next_state)
        self.episode_total_reward += r
        terminal = self.is_terminal(next_state)

        return state_vector, terminal, r

    def step(self, actions):
        return self.execute(actions) + ({},)

    @property
    def states_dim(self):
        return dict(shape=(17,), type='float')

    @property
    def actions_dim(self):
        return dict(type='int', num_actions=4)

    def close(self):
        print('session is closed!')

    def render(self, mode='human', close=False):
        raise NotImplementedError()


def main():
    environment = Simulator(model=joblib.load('models/nn-m.model'), frequency=10.0, safe_distance=8.0, goal_margin=8.0,
                            default_speed=3.0, radius=30.0, episode_length=1000, environment_size=70.0)

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
        timesteps=100000 * environment.episode_length,
        episodes=100000,
        max_episode_timesteps=environment.episode_length,
        deterministic=False,
        episode_finished=episode_finished
    )

    terminal, state = False, environment.reset()
    while not terminal:
        action = agent.act(state)
        state, terminal, reward = environment.execute(action)

    runner.close()


def keras_rl_impl():
    environment = Simulator(model=joblib.load('models/nn-m.model'), frequency=10.0, safe_distance=8.0, goal_margin=8.0,
                            default_speed=3.0, radius=30.0, episode_length=1000, environment_size=70.0)

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + environment.states_dim['shape']))

    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dense(environment.actions_dim['num_actions']))
    model.add(Activation('linear'))
    print(model.summary())

    memory = SequentialMemory(limit=9000, window_length=1)
    policy = BoltzmannQPolicy()

    dqn = DQNAgent(model=model, nb_actions=environment.actions_dim['num_actions'], memory=memory, nb_steps_warmup=10,
                   enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)

    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn.fit(environment, nb_steps=50000000, visualize=False, verbose=0, nb_max_episode_steps=1000)
    dqn.save_weights('dqn_{}_weights.h5f'.format(time.time()), overwrite=True)


if __name__ == '__main__':
    main()
