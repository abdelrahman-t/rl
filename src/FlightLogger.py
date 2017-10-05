import csv
import pandas as pd

from RLAgent import *
from Utilities import *


def flightLogger(agent, dataset=None, baseFrequency=10):
    keymap = defaultdict(lambda: 'hover')
    keymap.update([('Key.up', 'moveForward'), ('Key.left', 'yawCCW'), ('Key.right', 'yawCW'), ('Key.down', 'hover')])
    inverseKeymap = {'moveForward': 8, 'yawCCW': 4, 'yawCW': 6, 'hover': 5}
    timeStep = 0

    try:
        df = pd.read_csv(dataset)
        aIndices, aNames = df['aIndex'].values, df['aName'].values
        indices = np.repeat(df.index.values, agent.decisionFrequency // baseFrequency)

    except Exception as e:
        agent.logger.critical(e)
        sys.exit(1)

    writer = None
    with open(getDateTime().strip() + '.csv', 'w') as csvfile:
        while True:
            start = time.time()
            initialState, a = agent.getState(), agent.keyPressed.value

            try:
                selectedAction = aNames[indices[timeStep]] if dataset else keymap[a]
            except Exception as e:
                agent.logger.info('data collection is complete!')
                sys.exit(0)

            yield selectedAction
            r, nextState, isTerminal = (yield)

            f = 1 / (time.time() - start)

            linearVelocityE0, linearVelocityE1 = initialState.linearVelocity, nextState.linearVelocity
            linearVelocityB0, linearVelocityB1 = initialState.orientation.rotate(linearVelocityE0), nextState.orientation.rotate(
                linearVelocityE1)

            euler0, euler1 = getRollPitchYaw(initialState.orientation), getRollPitchYaw(nextState.orientation)

            records = {
                # [time-step, frequency, numerical value of selected action, action description]
                't': timeStep, 'f': f, 'aIndex': inverseKeymap[selectedAction], 'aName': selectedAction,

                # -----------------
                # absolute initial position [Unreal Engine]
                'x0': initialState.position[0], 'y0': initialState.position[1], 'z0': initialState.position[2],

                # -----------------
                # initial Linear Velocities in Body, Earth [Instantaneous]
                'dXB0': linearVelocityB0[0], 'dYB0': linearVelocityB0[1], 'dZB0': linearVelocityB0[2],
                'dXE0': linearVelocityE0[0], 'dYE0': linearVelocityE0[1], 'dZE0': linearVelocityE0[2],

                # ----------------
                # initial Orientation [(roll, pitch, yaw) or (x, y, z)]
                'psi0': euler0[0], 'theta0': euler0[1], 'phi0': euler0[2],

                # ----------------
                # initial Orientation [Quaternion]
                'scalar0': initialState.orientation[0], 'i0': initialState.orientation[1], 'j0': initialState.orientation[2],
                'k0': initialState.orientation[3],

                # ----------------
                # initial Angular Velocities [Instantaneous]
                'dPsi0': initialState.angularVelocity[0], 'dTheta0': initialState.angularVelocity[1],
                'dPhi0': initialState.angularVelocity[2],

                # -----------------
                # absolute next position [Unreal Engine]
                'x1': nextState.position[0], 'y1': nextState.position[1], 'z1': nextState.position[2],

                # -----------------
                # next Linear Velocities in Body, Earth [Instantaneous]
                'dXB1': linearVelocityB1[0], 'dYB1': linearVelocityB1[1], 'dZB1': linearVelocityB1[2],
                'dXE1': linearVelocityE1[0], 'dYE1': linearVelocityE1[1], 'dZE1': linearVelocityE1[2],

                # ----------------
                # next Orientation [(roll, pitch, yaw) or (x, y, z)]
                'psi1': euler1[0], 'theta1': euler1[1], 'phi1': euler1[2],

                # ----------------
                # next Orientation [Quaternion]
                'scalar1': nextState.orientation[0], 'i1': nextState.orientation[1], 'j1': nextState.orientation[2],
                'k1': nextState.orientation[3],

                # ----------------
                # next Angular Velocities [instantaneous]
                'dPsi1': nextState.angularVelocity[0], 'dTheta1': nextState.angularVelocity[1],
                'dPhi1': nextState.angularVelocity[2]
                # ----------------
            }

            if timeStep == 0:
                writer = csv.DictWriter(csvfile, fieldnames=[i for i in records])
                writer.writeheader()

            if timeStep % agent.decisionFrequency == 0:
                stats = inverseKeymap[selectedAction], selectedAction, truncateFloat(f, 3)
                agent.logger.info(stats)

            writer.writerow(records)
            timeStep += 1
            yield


def main():
    agent = RLAgent('agent', decisionFrequency=10.0, defaultSpeed=4, defaultAltitude=6, yawRate=70)

    # callbacks will be called in the order they were specified, beware of order of execution (if any state parameter is dependant on
    # another)
    # state is lazily updated by the environment as the agent needs it , agent always get the freshest estimate of the state
    # state updates are done by the environment in a rate that corresponds to agent decision making freq.
    agent.defineState(orientation=RLAgent.getOrientation, angularVelocity=RLAgent.getAngularVelocity, linearVelocity=RLAgent.getVelocity,
                      position=RLAgent.getPosition)

    agent.setRl(partial(flightLogger, dataset='C:/Users/talaa/Desktop/out.csv'))
    agent.start()


if __name__ == '__main__':
    main()
