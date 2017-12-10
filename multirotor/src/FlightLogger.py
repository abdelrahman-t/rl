from RLAgent import *
from Utilities import *


def flightLogger(agent, dataset=None, baseFrequency=10):
    keymap = defaultdict(lambda: 'hover')
    keymap.update([('Key.up', 'moveForward'), ('Key.left', 'yawCCW'), ('Key.right', 'yawCW'), ('Key.down', 'hover')])
    inverseKeymap = {'moveForward': 8, 'yawCCW': 4, 'yawCW': 6, 'hover': 5}

    try:
        df = pd.read_csv(dataset)
        aIndices, aNames = df['aIndex'].values, df['aName'].values
        indices = np.repeat(df.index.values, agent.decisionFrequency // baseFrequency)

    except Exception as e:
        pass

    writer, timeStep = None, 1
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

            linearVelocityBody = transformToBodyFrame(nextState.linearVelocity, nextState.orientation)
            orientationEuler = toEulerianAngle(nextState.orientation)

            linearAccelerationBody = transformToBodyFrame(nextState.linearAcceleration, nextState.orientation)

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

                # ----------------
                # next Orientation [(roll, pitch, yaw) or (x, y, z)]
                'roll': orientationEuler[0], 'pitch': orientationEuler[1], 'yaw': orientationEuler[2],

                # -----------------
                # next Linear Velocities in Body [Instantaneous]
                'dXB': linearVelocityBody[0], 'dYB': linearVelocityBody[1], 'dZB': linearVelocityBody[2],

                # -----------------
                # next Linear Accelerations in Body [Instantaneous]
                'd2XB': linearAccelerationBody[0], 'd2YB': linearAccelerationBody[1],
                'd2ZB': linearAccelerationBody[2],

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
            timeStep += 1
            yield


def main():
    agent = RLAgent('agent', decisionFrequency=30.0, defaultSpeed=4, defaultAltitude=6, yawRate=60)

    agent.defineState(orientation=getOrientation, position=getPosition,
                      angularVelocity=getAngularVelocity, linearVelocity=getVelocity,
                      linearAcceleration=getLinearAcceleration,
                      angularAcceleration=getAngularAcceleration)

    agent.setRl(partial(flightLogger, dataset='datasets/' + 'replay.csv'))
    agent.start()
    agent.join()


if __name__ == '__main__':
    main()
