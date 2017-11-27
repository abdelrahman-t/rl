import csv
import pandas as pd

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

            linearVelocityEarth = nextState.linearVelocity
            linearVelocityBody = nextState.orientation.rotate(linearVelocityEarth)

            linearAccelerationEarth = nextState.linearAcceleration
            linearAccelerationBody = nextState.orientation.rotate(linearAccelerationEarth)

            orientationEuler = getRollPitchYaw(nextState.orientation)

            records = {
                # [time-step, frequency, numerical value of selected action, action description]
                't': timeStep, 'f': f, 's': timeStep // agent.decisionFrequency, 'aIndex': inverseKeymap[selectedAction],
                'aName': selectedAction,
                # -----------------
                # absolute next position [Unreal Engine]
                'x': nextState.position[0], 'y': nextState.position[1], 'z': nextState.position[2],

                # -----------------
                # next Linear Velocities in Body, Earth [Instantaneous]
                'dXB': linearVelocityBody[0], 'dYB': linearVelocityBody[1], 'dZB': linearVelocityBody[2],
                'dXE': linearVelocityEarth[0], 'dYE': linearVelocityEarth[1], 'dZE': linearVelocityEarth[2],

                # -----------------
                # next Linear Accelerations in Body, Earth [Instantaneous]
                'd2XB': linearAccelerationBody[0], 'd2YB': linearAccelerationBody[1], 'd2ZB': linearAccelerationBody[2],
                'd2XE': linearAccelerationEarth[0], 'd2YE': linearAccelerationEarth[1], 'd2ZE': linearAccelerationEarth[2],

                # ----------------
                # next Orientation [(roll, pitch, yaw) or (x, y, z)]
                'psi': orientationEuler[0], 'theta': orientationEuler[1], 'phi': orientationEuler[2],

                # ----------------
                # next Orientation [Quaternion]
                'scalar': nextState.orientation[0], 'i': nextState.orientation[1], 'j': nextState.orientation[2],
                'k': nextState.orientation[3],

                # ----------------
                # next Angular Velocities [instantaneous]
                'dPsi': nextState.angularVelocity[0], 'dTheta': nextState.angularVelocity[1], 'dPhi': nextState.angularVelocity[2],
                # ----------------

                # ----------------
                # next Angular Accelerations [instantaneous]
                'd2Psi': nextState.angularAcceleration[0], 'd2Theta': nextState.angularAcceleration[1],
                'd2Phi': nextState.angularAcceleration[2]
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
    agent = RLAgent('agent', decisionFrequency=50.0, defaultSpeed=4, defaultAltitude=6, yawRate=70)

    # callbacks will be called in the order they were specified, beware of order of execution (if any state parameter is dependant on
    # another)
    # state is lazily updated by the environment as the agent needs it , agent always get the freshest estimate of the state
    # state updates are done by the environment in a rate that corresponds to agent decision making freq.
    agent.defineState(orientation=RLAgent.getOrientation, angularVelocity=RLAgent.getAngularVelocity, linearVelocity=RLAgent.getVelocity,
                      position=RLAgent.getPosition, linearAcceleration=RLAgent.getLinearAcceleration,
                      angularAcceleration=RLAgent.getAngularAcceleration)

    # agent.setRl(partial(flightLogger, dataset='datasets/' + 'replay.csv'))
    agent.setRl(flightLogger)
    agent.start()
    agent.join()


if __name__ == '__main__':
    main()
