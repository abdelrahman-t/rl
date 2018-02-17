from collections import defaultdict
from itertools import count
from rl_agent import *
from utilities import *
from agent_helpers import *

import csv


def flight_logger(agent, dataset=None, base_frequency=10):
    keymap = defaultdict(lambda: 'hover')
    keymap.update([('Key.up', 'move_forward'), ('Key.left', 'yaw_ccw'), ('Key.right', 'yaw_cw'), ('Key.down', 'hover'),
                   ('d', 'move_right'), ('a', 'move_left'), ('s', 'move_backward')])
    inverse_keymap = {'move_forward': 8, 'yaw_ccw': 4, 'yaw_cw': 6, 'hover': 5, 'move_backward': -8, 'move_right': 9,
                      'move_left': 7}

    counter = count(1)
    writer, time_step = None, next(counter)
    if dataset:
        df = pd.read_csv(dataset)
        a_indices, a_names = df['aIndex'].values, df['aName'].values
        indices = np.repeat(df.index.values, agent._decision_frequency // base_frequency)

    with open('datasets/' + get_date_time().strip() + '.csv', 'w') as csv_file:
        while True:
            initial_state, a = agent.state, agent.key_pressed.value

            try:
                selected_action = a_names[indices[time_step]] if dataset else keymap[a]
            except Exception as e:
                print('data collection is complete!')
                sys.exit(0)

            yield selected_action
            r, next_state, is_terminal = (yield)

            f = 1 / (next_state.lastUpdate - initial_state.lastUpdate)
            records = {
                # [time-step, frequency, numerical value of selected action, action description]
                'timestep': time_step, 'frequency': f, 'period': time_step // agent._decision_frequency,
                'aIndex': inverse_keymap[selected_action], 'aName': selected_action,

                # -----------------
                # absolute next position
                'x': next_state.position[0], 'y': next_state.position[1], 'z': next_state.position[2],

                # ----------------
                # next Orientation [Quaternion]
                'scalar': next_state.orientation[0], 'i': next_state.orientation[1], 'j': next_state.orientation[2],
                'k': next_state.orientation[3],

                # -----------------
                # next Linear Velocities in Earth [Instantaneous]
                'dXB': next_state.linear_velocity[0], 'dYB': next_state.linear_velocity[1],
                'dZB': next_state.linear_velocity[2],

                # ----------------
                # next Angular Velocities Body [instantaneous]
                'dRoll': next_state.angular_velocity[0], 'dPitch': next_state.angular_velocity[1],
                'dYaw': next_state.angular_velocity[2],
                # ----------------
            }

            if time_step == 1:
                writer = csv.DictWriter(csv_file, fieldnames=[i for i in records])
                writer.writeheader()

            if time_step % agent._decision_frequency == 0:
                stats = inverse_keymap[selected_action], selected_action, truncate_float(f, 3)
                print(stats)

            writer.writerow(records)
            time_step = next(counter)

            yield


def main():
    agent = RLAgent('agent', default_speed=3, default_altitude=10.0, yaw_rate=60, decision_frequency=20.0)

    agent.define_state(k=get_true_kinematic_state)

    agent.set_rl(partial(flight_logger, dataset='datasets/improved_action_set.csv'))
    agent.set_terminal(lambda: False)
    agent.set_reward(lambda *args, **kwargs: None)
    agent.run()


if __name__ == '__main__':
    main()
