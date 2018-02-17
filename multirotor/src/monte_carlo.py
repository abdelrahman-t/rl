from rl_agent import *
from vector_math import *
from sklearn.externals import joblib
from agent_helpers import *

from mcts import mcts
from mcts.mcts_utils import GameState
from mcts.tree_policy import UCB1
from mcts.backups import monte_carlo
from typing import List, NewType

A = NewType('action', int)

# CONFIG #
START = numpy.array([4180.0, -1830.0])
GOAL = (numpy.array([5920.0, -11140.0]) - START) / 100.0
ACTION = numpy.array([[1., 0., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]])

OBSTACLE = (numpy.array([[2620.0, -6210.0],
                         #[3430.0, -6210.0],
                         [4380.0, -6210.0],
                         #[5330.0, -6210.0],
                         [4390.0, -9270],
                         [5350.0, -9270],
                         [6270.0, -9270],
                         [6280.0, -6210.0]]) - START) / 100.0

MIN_OBS = 9.0
ACTION_NAMES = ['move_forward', 'yaw_ccw', 'yaw_cw', 'hover']

FREQUENCY = 5
# CONFIG #


def get_horizontal_distance(p1: List[float], p2: List[float]) -> float:
    return ((p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2) ** 0.5


class State(GameState[int]):
    def perform(self, action: A) -> GameState:
        roll, pitch, yaw = to_euler_angles(self.orientation)
        s0 = np.concatenate((self.linear_velocity, self.angular_velocity, [roll, pitch], ACTION[action]))
        s1 = self.model.predict(s0.reshape(1, -1))[0]

        orientation, position, linear_velocity, angular_velocity = \
            next(integrate_trajectory_velocity_body(position=self.position, orientation=self.orientation,
                                                    linear_velocities=[s1[:3]], angular_velocities=[s1[3:6]],
                                                    frequency=[self.frequency]))

        return State(position=position, orientation=orientation, linear_velocity=linear_velocity,
                     angular_velocity=angular_velocity, frequency=self.frequency, model=self.model)

    def reward(self) -> float:
        r1: float = -(1 * (self.position[0] - GOAL[0]) ** 2 +
                      (1 * (self.position[1] - GOAL[1]) ** 2))

        obs = self.distance_to_obstacle()
        if self.is_terminal():
            if obs < MIN_OBS:
                return -1.0e6

            else:
                return 1.0e6

        else:
            return r1

    @property
    def actions(self) -> List[A]:
        return [0, 1, 2, 3]

    def distance_to_obstacle(self):
        _ = min([get_horizontal_distance(self.position, o) for o in OBSTACLE])
        return _

    def is_terminal(self) -> bool:
        return self.distance_to_obstacle() < MIN_OBS or get_horizontal_distance(GOAL, self.position) < 5.0


def rl(agent):
    model = joblib.load('models/nn-m.model')

    m = mcts.MCTSRootParallel(number_of_processes=6, tree_policy=UCB1(3.0), default_policy='random-k', k=20,
                              backup=monte_carlo, time_limit=1 / FREQUENCY)

    while True:
        s = agent.state
        args = State(position=s.position, orientation=s.orientation,
                     linear_velocity=transform_to_body_frame(s.linear_velocity, s.orientation),
                     angular_velocity=s.angular_velocity, frequency=FREQUENCY, model=model)

        best_action = m.run(state=args)

        yield ACTION_NAMES[best_action]
        r, next_state, is_terminal = (yield)

        f = 1 / (next_state.lastUpdate - s.lastUpdate)
        print((f, get_horizontal_distance(GOAL, next_state.position)))

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
