import time
import json
from collections import defaultdict
from typing import Tuple, NamedTuple
from itertools import groupby
from operator import itemgetter

import numpy as np
from tensorforce.execution import Runner
from tensorforce.agents import Agent

from agent_helpers import *
from rl_agent import RLAgent
from vector_math import to_euler_angles, transform_to_body_frame, delta_heading_2d, distance, unit
from utilities import StateT

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

Obstacle = NamedTuple('Obstacle', [('position_body', np.ndarray), ('distance', float), ('angle', float)])

# CONFIG #
# START = np.array([4180.0, -4270.0, 0.0])
START = np.array([2440.0, -6210.0, 0.0])
GOAL = (np.array([6290.0, -9490.0, 0.0]) - START) / 100.0
SAFE = 7.6
DONE = 7.5

OBSTACLES = (np.array([  # [2440.0, -6210.0, 0.0],
    [4380.0, -6210.0, 0.0],
    [6240.0, -6210.0, 0.0],

    [3190.0, -7830.0, 0.0],
    [5240.0, -7830.0, 0.0],
    [7380.0, -7830.0, 0.0],

    [2630.0, -9490.0, 0.0],
    [4390.0, -9490.0, 0.0],
    [6290.0, -9490.0, 0.0],

    [5920.0, -11140.0, 0.0],
]) - START) / 100.0


# OBSTACLES += [float('inf'), float('inf'), float('inf')]


class Simulator:
    def __init__(self):
        self.agent = RLAgent('agent', default_speed=4.0, default_altitude=1, yaw_rate=60, decision_frequency=10.0)
        self.agent.define_state(k=get_true_kinematic_state)

        self.agent.initialize()
        self.agent.update_state()

        self.period = 1 / self.agent.decision_frequency
        self.time_step = 0
        self.last_call = time.time()
        self.bins = np.linspace(-np.pi, np.pi, 9)

        self.actions_map = {0: self.agent.actions['move_forward'], 1: self.agent.actions['hover'],
                            2: self.agent.actions['yaw_cw'], 3: self.agent.actions['yaw_ccw']}

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

            obstacles:
                distance to obstacles: expressed in meters [17-26]
        """
        inertial = np.concatenate((state.position,
                                   transform_to_body_frame(state.linear_velocity, state.orientation),
                                   state.angular_velocity, to_euler_angles(state.orientation)))
        goal = np.concatenate(
            ([delta_heading_2d(state.position, state.orientation, GOAL), distance(state.position, GOAL)], GOAL))
        obs = self.obstacles(state)

        return np.concatenate((inertial, goal, obs))

    def reset(self) -> np.ndarray:
        return self.vectorize_state(self.agent.reset())

    def obstacles(self, state) -> np.ndarray:
        max_radius = self.agent.default_speed * 5
        num_bins = len(self.bins)
        angle_dist = dict(zip(range(num_bins), [max_radius] * num_bins))

        in_radius = filter(lambda obs: distance(obs, state.position) <= max_radius,
                           OBSTACLES)

        obs_dist_angle = [*map(lambda obs: Obstacle(position_body=obs,
                                                    distance=distance(obs, state.position),
                                                    angle=int(np.digitize(
                                                        delta_heading_2d(state.position, state.orientation, obs),
                                                        self.bins, right=True))),
                               in_radius)]

        obs_dist_angle.sort(key=lambda x: x.angle)
        for key, group in groupby(obs_dist_angle, key=lambda x: x.angle):
            angle_dist[key] = int(min(angle_dist[key], min(map(itemgetter(1), group))))

        return [angle_dist[i] for i in range(num_bins)]

    def reward(self, state) -> float:
        goal_body = transform_to_body_frame(GOAL - state.position, state.orientation)
        unit_displacement = np.dot(unit(goal_body), transform_to_body_frame(state.linear_velocity, state.orientation))
        distance_obs = min(self.obstacles(state))

        if distance_obs < SAFE:
            return -1 * 800.0

        return np.clip(unit_displacement / self.agent.default_speed, a_min=-1.0, a_max=1.0)

    def is_terminal(self, state) -> bool:
        return min(self.obstacles(state)) < SAFE or distance(GOAL, state.position) < DONE

    def execute(self, actions) -> Tuple[np.ndarray, float, bool]:
        start = time.time()
        self.time_step += 1

        self.agent.perform_action(self.actions_map[actions])
        while time.time() - start < self.period:
            continue
        self.agent.update_state()

        next_state = self.agent.state
        r = self.reward(next_state)
        terminal = self.is_terminal(next_state)

        state_vector = self.vectorize_state(next_state)
        if self.time_step % self.agent.decision_frequency == 0:
            print('reward: {}, heading error: {}, distance to goal: {}, obstacles'
                  .format(np.round(r, 3), int(np.rad2deg(state_vector[12])), int(state_vector[13])), state_vector[17:])

        # format (observation, reward, done) keras
        # return state_vector, r, terminal

        # format (observation, done, reward) tensorforce
        return state_vector, terminal, r

    def step(self, actions):
        return self.execute(actions) + ({},)

    @property
    def states_dim(self):
        return dict(shape=(26,), type='float')

    @property
    def actions_dim(self):
        return dict(type='int', num_actions=4)

    def __str__(self):
        pass

    def close(self):
        print('\nclosed\n')

    def render(self, mode='human', close=False):
        pass


def keras_rl_impl():
    environment = Simulator()

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + environment.states_dim['shape']))

    model.add(Dense(128))
    model.add(Activation('relu'))

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
    # dqn.load_weights('latest.h5f')
    dqn.fit(environment, nb_steps=500000, visualize=True, verbose=2, nb_max_episode_steps=1200)
    dqn.save_weights('dqn_{}_weights.h5f'.format(time.time()), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    # dqn.test(environment, nb_episodes=5, visualize=True)


def test():
    bins = np.linspace(-np.pi, np.pi, 9)

    def reward(state):
        goal_body = transform_to_body_frame(GOAL - state.position, state.orientation)
        displacement = np.dot(goal_body / np.linalg.norm(goal_body),
                              transform_to_body_frame(state.linear_velocity, state.orientation))
        return displacement / agent.default_speed

    def obstacles(state) -> np.ndarray:
        max_radius = 15.0
        num_bins = len(bins)
        angle_dist = dict(zip(range(num_bins), [max_radius] * num_bins))

        in_radius = filter(lambda obs: distance(obs, state.position) <= max_radius,  # and
                           # -np.pi / 2 <= delta_heading_2d(state.position, state.orientation, obs) <= np.pi / 2,
                           OBSTACLES)

        obs_dist_angle = [*map(lambda obs: Obstacle(position_body=obs,
                                                    distance=distance(obs, state.position),
                                                    angle=int(np.digitize(
                                                        delta_heading_2d(state.position, state.orientation, obs),
                                                        bins, right=True))),
                               in_radius)]

        obs_dist_angle.sort(key=lambda x: x.angle)
        for key, group in groupby(obs_dist_angle, key=lambda x: x.angle):
            angle_dist[key] = int(min(angle_dist[key], min(map(itemgetter(1), group))))

        return [angle_dist[i] for i in range(num_bins)]

    def rl(agent: RLAgent):
        keymap = defaultdict(lambda: 'hover')
        keymap.update([('Key.up', 'move_forward'), ('Key.left', 'yaw_ccw'), ('Key.right', 'yaw_cw'), ('Key.down', 'hover'),
                       ('d', 'move_right'), ('a', 'move_left'), ('s', 'move_backward')])

        while True:
            yield keymap[agent.key_pressed.value]
            r, next_state, is_terminal = (yield)
            yield

            print('reward {}, deviation: {}, distance: {} obs: {}'
                  .format(np.round(reward(next_state), 1),
                          int(np.rad2deg(delta_heading_2d(next_state.position, next_state.orientation, GOAL))),
                          int(np.linalg.norm(GOAL - next_state.position)),
                          obstacles(next_state)))

    agent = RLAgent('agent', default_speed=3, default_altitude=1.0, yaw_rate=60, decision_frequency=10.0)
    agent.define_state(k=get_true_kinematic_state)

    agent.set_rl(rl)
    agent.set_terminal(lambda *args, **kwargs: False)
    agent.set_reward(lambda *args, **kwargs: None)
    agent.run()


def tensorforce_impl():
    environment = Simulator()

    with open('network.json', 'r') as fp:
        network_spec = json.load(fp=fp)

    with open('config.json', 'r') as fp:
        config = json.load(fp=fp)

    agent = Agent.from_spec(
        spec=config,
        kwargs=dict(
            states=environment.states_dim,
            actions=environment.actions_dim,
            network=network_spec,
        )
    )

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )

    logger = environment.agent.logger

    def episode_finished(r):
        sps = r.timestep / (time.time() - r.start_time)
        logger.info("Finished episode {ep} after {ts} timesteps. Steps Per Second {sps}".
                    format(ep=r.episode, ts=r.timestep, sps=sps))
        logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
        logger.info("Episode timesteps: {}".format(r.episode_timestep))
        logger.info("Average of last 500 rewards: {}".format(sum(r.episode_rewards[-500:]) / 500))
        logger.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
        return True

    runner.run(
        timesteps=6000000,
        episodes=1000,
        max_episode_timesteps=800,
        deterministic=False,
        episode_finished=episode_finished
    )

    terminal, state = False, environment.reset()
    while not terminal:
        action = agent.act(state)
        state, terminal, reward = environment.execute(action)

    runner.close()


if __name__ == '__main__':
    tensorforce_impl()
