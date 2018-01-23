from Utilities import *
from Common import *
from RLAgent import *
from Model import *

from multiprocess import Pool, Pipe


def monteCarloSearch(agent, pipe, shared):
    q, isTerminal = 0.0, False
    timestep = 0
    # print('{} started!!'.format(agent.name))
    # blocking call
    state, a = pipe.recv()
    actions = agent.getActions()

    while True:
        nextActions = actions[np.random.randint(4)]
        yield nextActions if timestep else a
        r, nextState, isTerminal = (yield)

        q += r

        if isTerminal:
            # branch is exhausted, return q
            pipe.send((a, q))
            timestep = -1

            # blocking call, wait for next instructions
            state, a = pipe.recv()

            # reset state to new state
            agent.initialize(state)

        timestep += 1

        # nothing to yield
        yield None


# has side effects!!
def setRandomActions(shared, actions, depth):
    sequence = np.random.randint(len(actions), size=depth - 1)
    ### removed!


def monteCarlo(agent, maxDepth=50, trials=1, frequency=10, **kwargs):
    model = VelocityModel(regressionModel=joblib.load('models/gradient-m.model'),
                          frequency=frequency)

    actions = np.array(agent.getActions())
    shared = []
    agents = []
    pipes = []
    kwargs = []

    for i in range(len(actions) * trials):
        virtualAgent = RLAgent('virtual agent ({})'.format(i),
                               alternativeModel=model, decisionFrequency=math.inf, maxDepth=maxDepth)

        virtualAgent.setReward(reward)
        virtualAgent.goal = agent.getGoal()
        virtualAgent.goalMargins = agent.getGoalMargins()

        parent, child = Pipe()
        kwargs.append({'pipe': child, 'shared': shared})

        virtualAgent.setRl(monteCarloSearch)
        virtualAgent.initialize(agent.getState())
        agents.append(virtualAgent)

        pipes.append(parent)

    pool = Pool()
    for index, agent_ in enumerate(agents):
        pool.imap(agent_.run, [kwargs[index]])

    time.sleep(10)
    tree = np.repeat(actions, trials)

    print('pool initialized!')

    while True:
        qs = defaultdict(int)
        initialState = agent.getState()
        # setRandomActions(shared, actions, maxDepth)

        for i, root in enumerate(tree):
            # send agents starting actions
            pipes[i].send((initialState, root))

        for pipe in pipes:
            a, q = pipe.recv()
            qs[a] += q

        yield actions[np.argmax([np.average(qs[a]) for a in actions])]
        r, nextState, isTerminal = (yield)

        f = 1 / (nextState.lastUpdate - initialState.lastUpdate)
        agent.logger.info((f, nextState.goal))

        #model.frequency = f
        yield


def main():
    agent = RLAgent('agent', decisionFrequency=10.0, defaultSpeed=4, defaultAltitude=6, yawRate=70)

    # callbacks will be called in the order they were specified, beware of order of execution (if any state parameter is
    #  dependant on another)
    # state is lazily updated by the environment as the agent needs it , agent always get the freshest estimate of the
    # state, state updates are done by the environment in a rate that corresponds to agent decision making freq.

    agent.defineState(orientation=getOrientation, angularVelocity=getAngularVelocity,
                      linearVelocity=getVelocity, position=getPosition, goal=getHorizontalDistanceGoal)

    agent.setRl(monteCarlo)
    agent.setReward(reward)
    agent.setGoal(position=np.array([-40, -50, 0]))
    agent.setGoalMargins(position=np.array([2.0, 2.0, math.inf]))
    agent.start()
    agent.join()


if __name__ == '__main__':
    main()
