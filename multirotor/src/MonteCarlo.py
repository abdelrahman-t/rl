from Utilities import *
from Common import *
from RLAgent import *
from Model import *
from itertools import count
from multiprocess import Pool

def reward(agent):
    terminalStateReward = (1.0e2, -1.0e2)
    wPositionX = wPositionY = 1
    g, s1 = agent.getGoal(), agent.getState()

    if agent.isTerminal():
        r = terminalStateReward[0] if isGoal(agent=agent) else terminalStateReward[1]
    else:
        r = -(wPositionX * (s1.position[0] - g.position[0]) ** 2 +
             (wPositionY * (s1.position[1] - g.position[1]) ** 2))

    return r


def monteCarloSearch(agent, actions):
    q, isTerminal = 0.0, False
    timestep = count()

    while not isTerminal:
        yield actions[next(timestep)]
        r, nextState, isTerminal = (yield)

        q += r

        if isTerminal:
            yield (actions[0], q)
        yield


def getRandomActions(start, actions, depth):
    sequence = actions[np.random.randint(len(actions), size=depth)]
    sequence[0] = start

    return sequence


# to do: use multiprocessing instead of threading as GIL renders the threaded approach useless!!
# thread-safety is not enforced, code is only thread-safe on CPython
def monteCarlo(agent, maxDepth=3, trials=12, frequency=10):
    PROCESSES = 4
    model = VelocityModel(regressionModel=joblib.load('models/gradient-m.model'), frequency=frequency)
    actions = np.array(agent.getActions())
    initialState, isTerminal = agent.getState(), 0

    jobs = [None] * len(actions) * trials
    while bool(isTerminal) is False:
        initialState = agent.getState()
        qs = {i: [] for i in actions}

        for index, a in enumerate(np.repeat(actions, trials)):
            virtualAgent, isTerminal = RLAgent('virtual', alternativeModel=model, decisionFrequency=math.inf,
                                               maxDepth=maxDepth, initialState=initialState), False
            virtualAgent.setReward(reward)
            virtualAgent.goal = agent.getGoal()
            virtualAgent.goalMargins = agent.getGoalMargins()

            virtualAgent.setRl(partial(monteCarloSearch,
                                       actions=getRandomActions(a, actions, maxDepth)))
            jobs[index] = virtualAgent

        pool = Pool(8)
        results = [pool.apply_async(job.run) for job in jobs]
        for result in results:
            action, score = result.get()
            qs[action].append(score)

        pool.close()
        pool.join()

        yield actions[np.argmax([np.average(qs[a]) for a in actions])]
        r, nextState, isTerminal = (yield)

        f = 1 / (nextState.lastUpdate - initialState.lastUpdate)
        # correct for deviations from desired freq.
        model.frequency = f

        agent.logger.info(f)

        yield


def main():
    agent = RLAgent('agent', decisionFrequency=10.0, defaultSpeed=4, defaultAltitude=20, yawRate=70)

    # callbacks will be called in the order they were specified, beware of order of execution (if any state parameter is
    #  dependant on another)
    # state is lazily updated by the environment as the agent needs it , agent always get the freshest estimate of the
    # state, state updates are done by the environment in a rate that corresponds to agent decision making freq.

    agent.defineState(orientation=getOrientation, angularVelocity=getAngularVelocity,
                      linearVelocity=getVelocity, position=getPosition)

    agent.setRl(monteCarlo)
    agent.setReward(reward)
    agent.setGoal(position=np.array([-40, -50, 0]))
    agent.setGoalMargins(position=np.array([0.5, 0.5, math.inf]))
    agent.start()
    agent.join()


if __name__ == '__main__':
    main()