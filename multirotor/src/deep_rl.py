import numpy
from rl_agent import RLAgent
from typing import List, Tuple, Union

# CONFIG #
START = numpy.array([4180.0, -4270.0])
GOAL = (numpy.array([5920.0, -12560.0]) - START) / 100.0
SAFE = 9

OBSTACLE = (numpy.array([[2620.0, -6210.0],
                         [4380.0, -6210.0],
                         [6280.0, -6210.0],

                         [2630.0, -9490.0],
                         [4390.0, -9490.0],
                         [6290.0, -9490.0],

                         [3480.0, -7830.0],
                         [5240.0, -7830.0],
                         [7140.0, -7830.0],

                         [5920.0, -11140.0],
                         ]) - START) / 100.0


def get_horizontal_distance(p1: Union[List[float], numpy.ndarray], p2: Union[List[float], numpy.ndarray]) -> float:
    return ((p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2) ** 0.5


def get_nearest_obstacle(agent: RLAgent) -> Tuple[float, Union[List[float], numpy.ndarray]]:
    state = agent.state
    nearest_obs = min(OBSTACLE, key=lambda o: get_horizontal_distance(state.position, o))
    cos_angle: float = numpy.dot(nearest_obs, state.position[:2]) / (
            numpy.linalg.norm(nearest_obs) * numpy.linalg.norm(state.position))

    return cos_angle, nearest_obs


def get_distance_goal(agent: RLAgent) -> float:
    return get_horizontal_distance(agent.state.position[:2], GOAL)


def reward(agent: RLAgent) -> float:
    cos_angle, nearest_obs = get_nearest_obstacle(agent)
    return -get_distance_goal(agent) if get_horizontal_distance(nearest_obs, agent.state.position) >= SAFE else -1e6


def trpo(agent: RLAgent, num_episodes: int, checkpoints_path: str):
    pass


def dqn(agent: RLAgent, num_episodes: int, checkpoints_path: str):
    pass
