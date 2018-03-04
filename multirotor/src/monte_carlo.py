from rl_agent import *
from vector_math import *
from sklearn.externals import joblib
from agent_helpers import *

import mcts
from mcts_utils import GameState
from tree_policy import UCB1
from backups import *
from typing import List, NewType
from numpy import ndarray

A = NewType('action', int)

# CONFIG #
START = numpy.array([4180.0, -4270.0])
GOAL = (numpy.array([5920.0, -12560.0]) - START) / 100.0

ACTION = numpy.array([[1., 0., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]])

OBSTACLE = (numpy.array([[2620.0, -6210.0],
                         [4380.0, -6210.0],
                         [6280.0, -6210.0],

                         [2630.0, -9490.0],
                         [4390.0, -9490.0],
                         [6290.0, -9490.0],

                         [3480.0, -7830.0],
                         [5240.0, -7830.0],
                         [7140.0, -7830.0]
                         ]) - START) / 100.0

MIN_OBS = 5.0
MIN_GOAL = 10.0
ACTION_NAMES = ['move_forward', 'yaw_ccw', 'yaw_cw', 'hover']

FREQUENCY = 10.5


# CONFIG #


def get_horizontal_distance(p1: List[float], p2: List[float]) -> float:
    return ((p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2) ** 0.5


class State(GameState[int]):
    w_o_bins: ndarray = numpy.linspace(start=0.0, stop=numpy.deg2rad(180.0), num=90 + 1)  # 90
    v_bins: ndarray = numpy.linspace(start=0.0, stop=5.0, num=200 + 1)  # 200
    p_bins: ndarray = numpy.linspace(start=0.0, stop=200.0, num=1600 + 1)  # 1600

    def perform(self, action: A) -> GameState:
        roll, pitch, yaw = to_euler_angles(self.orientation)
        s0 = np.concatenate((self.linear_velocity, self.angular_velocity, [roll, pitch], ACTION[action]))
        s1 = self.model.predict(s0.reshape(1, -1))[0]

        orientation, position, linear_velocity, angular_velocity = \
            next(integrate_trajectory_velocity_body(position=self.position, orientation=self.orientation,
                                                    linear_velocities=[s1[:3]], angular_velocities=[s1[3:6]],
                                                    frequency=[self.frequency]))

        disc = numpy.array([numpy.digitize(numpy.abs(linear_velocity), State.v_bins),
                            numpy.digitize(numpy.abs(angular_velocity), State.w_o_bins),
                            numpy.digitize(numpy.abs(position), State.p_bins),
                            numpy.digitize(numpy.abs(to_euler_angles(orientation)), State.w_o_bins)])

        return State(position=position, orientation=orientation, linear_velocity=linear_velocity,
                     angular_velocity=angular_velocity, frequency=self.frequency, model=self.model, disc=disc)

    def distance_goal(self) -> float:
        distance: float = (1 * (self.position[0] - GOAL[0]) ** 2 +
                           (1 * (self.position[1] - GOAL[1]) ** 2)) ** 0.5

        return distance

    def distance_to_obstacle(self):
        return min([get_horizontal_distance(self.position, o) for o in OBSTACLE])

    def reward(self):
        dist_obs: float = self.distance_to_obstacle()
        dist_goal: float = self.distance_goal()

        if dist_goal > MIN_GOAL:
            return -dist_goal + 50 * numpy.exp(-MIN_OBS / dist_obs)
        else:
            return 1e3

    @property
    def actions(self) -> List[A]:
        return [0, 1, 2, 3]

    def is_terminal(self) -> bool:
        return self.distance_to_obstacle() < MIN_OBS or \
               self.distance_goal() < MIN_GOAL

    def __eq__(self, other):
        return numpy.array_equal(self.disc, other.disc)

    def __hash__(self):
        return hash(self.disc.tostring())


def rl(agent):
    model = joblib.load('models/nn-m.model')

    m = mcts.MCTSRootParallel(number_of_processes=4, tree_policy=UCB1(4.5), default_policy='random-k', k=20,
                              backup=monte_carlo, time_limit=1 / FREQUENCY, persist_tree=True, refit=True, cache=True)

    while True:
        s = agent.state
        args = State(position=s.position, orientation=s.orientation,
                     linear_velocity=transform_to_body_frame(s.linear_velocity, s.orientation),
                     angular_velocity=s.angular_velocity, frequency=FREQUENCY, model=model)

        yield ACTION_NAMES[m.run(state=args)]
        r, next_state, is_terminal = (yield)

        f = 1 / (next_state.lastUpdate - s.lastUpdate)
        print(f, args.distance_to_obstacle(), args.distance_goal())

        yield


def main():
    agent = RLAgent('agent', default_speed=4.0, default_altitude=1, yaw_rate=60, decision_frequency=FREQUENCY)
    agent.define_state(k=get_true_kinematic_state)
    agent.define_goal(position=GOAL)

    agent.set_rl(rl)
    agent.set_terminal(lambda: False)
    agent.set_reward(lambda *args, **kwargs: None)
    agent.run()


if __name__ == '__main__':
    main()
