from RLAgent import *
from Utilities import *


def flightLogger(agent, dataset=None, baseFrequency=10):
    keymap = defaultdict(lambda: 'hover')
    keymap.update([('Key.up', 'moveForward'), ('Key.left', 'yawCCW'), ('Key.right', 'yawCW'), ('Key.down', 'hover')])
    inverseKeymap = {'moveForward': 8, 'yawCCW': 4, 'yawCW': 6, 'hover': 5}

    counter = count(1)
    writer, timeStep = None, next(counter)
    if dataset:
        df = pd.read_csv(dataset)
        aIndices, aNames = df['aIndex'].values, df['aName'].values
        indices = np.repeat(df.index.values, agent.decisionFrequency // baseFrequency)

    with open('datasets/' + getDateTime().strip() + '.csv', 'w') as csvfile:
        while True:
            initialState, a = agent.getState(), agent.keyPressed.value

            try:
                selectedAction = aNames[indices[timeStep]] if dataset else keymap[a]
            except Exception as e:
                agent.logger.critical('data collection is complete!')
                sys.exit(0)

            yield selectedAction
            r, nextState, isTerminal = (yield)

            f = 1 / (nextState.lastUpdate - initialState.lastUpdate)
            records = {
                # [time-step, frequency, numerical value of selected action, action description]
                'timestep': timeStep, 'frequency': f, 'period': timeStep // agent.decisionFrequency,
                'aIndex': inverseKeymap[selectedAction], 'aName': selectedAction,

                # -----------------
                # absolute next position
                'x': nextState.position[0], 'y': nextState.position[1], 'z': nextState.position[2],

                # ----------------
                # next Orientation [Quaternion]
                'scalar': nextState.orientation[0], 'i': nextState.orientation[1], 'j': nextState.orientation[2],
                'k': nextState.orientation[3],

                # -----------------
                # next Linear Velocities in Earth [Instantaneous]
                'dXB': nextState.linearVelocity[0], 'dYB': nextState.linearVelocity[1],
                'dZB': nextState.linearVelocity[2],

                # -----------------
                # next Linear Accelerations in Earth [Instantaneous]
                'd2XB': nextState.linearAcceleration[0], 'd2YB': nextState.linearAcceleration[1],
                'd2ZB': nextState.linearAcceleration[2],

                # ----------------
                # next Angular Velocities Body [instantaneous]
                'dRoll': nextState.angularVelocity[0], 'dPitch': nextState.angularVelocity[1],
                'dYaw': nextState.angularVelocity[2],
                # ----------------

                # ----------------
                # next Angular Accelerations Body [instantaneous]
                'd2Roll': nextState.angularAcceleration[0], 'd2Pitch': nextState.angularAcceleration[1],
                'd2Yaw': nextState.angularAcceleration[2]
                # ----------------
            }

            if timeStep == 1:
                writer = csv.DictWriter(csvfile, fieldnames=[i for i in records])
                writer.writeheader()

            if timeStep % agent.decisionFrequency == 0:
                stats = inverseKeymap[selectedAction], selectedAction, truncateFloat(f, 3)
                agent.logger.info(stats)

            writer.writerow(records)
            timeStep = next(counter)
            yield


def main():
    agent = RLAgent('agent', decisionFrequency=10.0, defaultSpeed=4, defaultAltitude=6, yawRate=60)

    agent.defineState(orientation=getOrientation, position=getPosition,
                      angularVelocity=getAngularVelocity, linearVelocity=getVelocity,
                      linearAcceleration=getLinearAcceleration, angularAcceleration=getAngularAcceleration)

    agent.setRl(partial(flightLogger, dataset='datasets/' + 'replay.csv'))
    agent.start()
    agent.join()


if __name__ == '__main__':
    main()
