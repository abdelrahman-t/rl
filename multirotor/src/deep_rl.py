import time
import json
from collections import defaultdict
from functools import partial
from itertools import starmap
from typing import List, Union

import numpy as np
from tensorforce.execution import Runner
from tensorforce.agents import Agent

from agent_helpers import *
from rl_agent import RLAgent
from vector_math import to_euler_angles, transform_to_body_frame, wrap_around_pi
from tensorforce.environments import Environment

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from scipy.interpolate import interp1d

# CONFIG #
START = np.array([4180.0, -4270.0, 0.0])
GOAL = (np.array([6290.0, -9490.0, 0.0]) - START) / 100.0
SAFE = 8.0
DONE = 7.5

OBSTACLES = (np.array([[2620.0, -6210.0, 0.0],
                       [4380.0, -6210.0, 0.0],
                       [6280.0, -6210.0, 0.0],

                       [2630.0, -9490.0, 0.0],
                       [4390.0, -9490.0, 0.0],
                       [6290.0, -9490.0, 0.0],

                       [3480.0, -7830.0, 0.0],
                       [5240.0, -7830.0, 0.0],
                       [7140.0, -7830.0, 0.0],

                       [5920.0, -11140.0, 0.0],
                       ]) - START) / 100.0


class Simulator(Environment):
    ACTIONS = ['move_forward', 'hover', 'yaw_cw', 'yaw_ccw']

    def __init__(self):
        self.agent = RLAgent('agent', default_speed=4.0, default_altitude=1, yaw_rate=60, decision_frequency=10.0)
        self.agent.define_state(k=get_true_kinematic_state)

        self.agent.initialize()
        self.agent.update_state()

        self.period = 1 / self.agent.decision_frequency
        self.scalar = interp1d([-1.0, 1.0], [-1.0, 0.0], bounds_error=False, fill_value='extrapolate')

        self.time_step = 0
        self.last_call = time.time()
        self.bins = np.linspace(-np.pi / 2, np.pi / 2, 5)

    def distance_goal(self, state):
        return np.linalg.norm(state.position - GOAL)

    def vectorized_state(self, state):
        nearest_obs = self.nearest_obstacle(state)
        return np.concatenate((state.angular_velocity, transform_to_body_frame(state.linear_velocity, state.orientation),
                               [self.distance_goal(state), self.desired_heading(state),
                                self.obstacle_direction(state, nearest_obs), self.distance(nearest_obs, state.position)],
                               to_euler_angles(state.orientation)[:2]))

    def reset(self):
        return self.vectorized_state(self.agent.reset())

    def distance(self, p1: Union[List[float], np.ndarray], p2: Union[List[float], np.ndarray]) -> float:
        return np.linalg.norm(p2 - p1)

    def obstacles(self, state) -> np.ndarray:
        obs = [*filter(lambda x: x < 15.0, sorted(OBSTACLES, key=lambda o: -self.distance(state.position, o)))]
        dist_ang = np.array(
            [*filter(lambda x: abs(x[1]) <= np.pi / 2, map(lambda y: (y, self.obstacle_direction(state, y)), obs))])

        np.digitize(dist_ang[])

    def reward(self, state):
        goal_body = transform_to_body_frame(GOAL - state.position, state.orientation)
        displacement = np.dot(goal_body / np.linalg.norm(goal_body),
                              transform_to_body_frame(state.linear_velocity, state.orientation))
        return displacement / self.agent.default_speed

    def desired_heading(self, state):
        goal_body = transform_to_body_frame(GOAL - state.position, state.orientation)
        return wrap_around_pi(np.arctan2(*goal_body[:2]) - np.pi / 2)

    def obstacle_direction(self, state, obstacle: np.ndarray) -> float:
        obstacle_body = transform_to_body_frame(obstacle - state.position, state.orientation)
        return wrap_around_pi(np.arctan2(*obstacle_body[:2]) - np.pi / 2)

    def is_terminal(self, state) -> bool:
        if self.time_step % (self.agent.decision_frequency * 2) == 0:
            print('frequency: ', round(1 / (time.time() - self.last_call), 2))
        self.last_call = time.time()
        return self.distance(GOAL, state.position) <= DONE or \
               self.distance(self.nearest_obstacle(state), state.position) <= SAFE

    def execute(self, actions):
        results = []

        start = time.time()

        for action in (actions if isinstance(actions, list) else [actions]):
            self.time_step += 1
            self.agent.perform_action(self.agent.actions[self.ACTIONS[action]])

            while time.time() - start < self.period:
                continue

            self.agent.update_state()
            state = self.agent.state

            safe = self.distance(self.nearest_obstacle(state), state.position) < SAFE
            results.append(
                (self.vectorized_state(self.agent.state), self.scalar(self.reward(state)) + int(safe) * -1000,
                 self.is_terminal(state)))

        if self.time_step % (2 * self.agent.decision_frequency) == 0:
            print('heading error: ', round(np.rad2deg(results[0][0][7])), 'reward: ', np.round(results[0][1], 2),
                  'velocity: ', np.round(self.reward(state), 2),
                  'obstacle angle: ', np.round(np.rad2deg(results[0][0][-4]), 2),
                  'obstacle distance: ', np.round(results[0][0][-3], 2), '\n')

        return results if len(results) > 1 else results[0]

    def step(self, actions):
        return self.execute(actions) + ({},)

    @property
    def states(self):
        return dict(shape=(12,), type='float')

    @property
    def actions(self):
        return dict(type='int', num_actions=4)

    def __str__(self):
        pass

    def close(self):
        print('\nclosed\n')

    def render(self, mode='human', close=False):
        pass


def tensorforce_impl():
    environment = Simulator()

    with open('network.json', 'r') as fp:
        network_spec = json.load(fp=fp)

    with open('config.json', 'r') as fp:
        config = json.load(fp=fp)

    agent = Agent.from_spec(
        spec=config,
        kwargs=dict(
            states=environment.states,
            actions=environment.actions,
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
        max_episode_timesteps=900,
        deterministic=False,
        episode_finished=episode_finished
    )

    terminal, state = False, environment.reset()
    while not terminal:
        action = agent.act(state)
        state, terminal, reward = environment.execute(action)

    runner.close()


def keras_rl_impl():
    environment = Simulator()

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + environment.states['shape']))

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(environment.actions['num_actions']))
    model.add(Activation('linear'))
    print(model.summary())

    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()

    # model.load_weights('dqn_1525215916.541092_weights.h5f')
    dqn = DQNAgent(model=model, nb_actions=environment.actions['num_actions'], memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)

    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn.fit(environment, nb_steps=500000, visualize=True, verbose=2, nb_max_episode_steps=1000)
    dqn.save_weights('dqn_{}_weights.h5f'.format(time.time()), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    # dqn.test(environment, nb_episodes=5, visualize=True)


def test():
    scalar = interp1d([-1.0, 1.0], [-1.0, 0.0])

    def reward(state):
        goal_body = transform_to_body_frame(GOAL - state.position, state.orientation)
        displacement = np.dot(goal_body / np.linalg.norm(goal_body),
                              transform_to_body_frame(state.linear_velocity, state.orientation))
        return scalar(displacement / agent.default_speed)

    def desired_heading(state):
        goal_body = transform_to_body_frame(GOAL - state.position, state.orientation)
        return wrap_around_pi(np.arctan2(*goal_body[:2]) - np.pi / 2)

    def rl(agent: RLAgent):
        keymap = defaultdict(lambda: 'hover')
        keymap.update([('Key.up', 'move_forward'), ('Key.left', 'yaw_ccw'), ('Key.right', 'yaw_cw'), ('Key.down', 'hover'),
                       ('d', 'move_right'), ('a', 'move_left'), ('s', 'move_backward')])

        while True:
            yield keymap[agent.key_pressed.value]
            r, next_state, is_terminal = (yield)
            yield

            displacement = np.dot(GOAL / np.linalg.norm(GOAL), next_state.linear_velocity)

            print('reward {}, deviation: {}, distance: {}'
                  .format(reward(next_state),
                          round(np.rad2deg(desired_heading(next_state)), 2),
                          round(np.linalg.norm(GOAL - next_state.position), 2)))

    agent = RLAgent('agent', default_speed=3, default_altitude=10.0, yaw_rate=60, decision_frequency=20.0)
    agent.define_state(k=get_true_kinematic_state)

    agent.set_rl(rl)
    agent.set_terminal(lambda *args, **kwargs: False)
    agent.set_reward(lambda *args, **kwargs: None)
    agent.run()


if __name__ == '__main__':
    keras_rl_impl()
