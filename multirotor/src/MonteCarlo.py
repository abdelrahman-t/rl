from Utilities import *
from Common import *
from RLAgent import *
from Model import *
from itertools import count


def reward(agent):
    terminalStateReward = (1.0e2, -1.0e2)
    wPositionX = wPositionY = 1
    g, s1 = agent.getGoal(), agent.getState()

    if agent.isTerminal():
        r = terminalStateReward[0] if isGoal(agent=agent) else terminalStateReward[1]
    else:
        # −(αx(x − x∗) **2 + αy(y − y∗) **2)
        r = -(wPositionX * (s1.position[0] - g.position[0]) ** 2 + (wPositionY * (s1.position[1] - g.position[1]) ** 2))

    return r


def monteCarloSearch(agent, callback, actions):
    q, isTerminal = 0.0, False
    timestep = count()

    while not isTerminal:
        yield actions[next(timestep)]
        r, nextState, isTerminal = (yield)

        q += r

        if isTerminal:
            callback(q=q)
        yield


def getRandomActions(start, actions, depth):
    sequence = actions[np.random.randint(len(actions), size=depth)]
    sequence[0] = start

    return sequence


# to do: use multiprocessing instead of threading as GIL renders the threaded approach useless!!
# thread-safety is not enforced, code is only thread-safe on CPython
def monteCarlo(agent, maxDepth=10, trials=30, frequency=10):
    model = AccelerationModel(regressionModel=joblib.load('models/mlp.model'), frequency=frequency)
    actions = np.array(agent.getActions())

    while True:
        initialState, isTerminal = agent.getState(), False
        while isTerminal is False:
            queue_, qs = Queue(), {i: [] for i in actions}

            for a in np.repeat(actions, trials):
                virtualAgent, isTerminal = RLAgent('virtual', alternativeModel=model, decisionFrequency=math.inf,
                                                   maxDepth=maxDepth, initialState=initialState), False
                virtualAgent.setReward(reward)
                virtualAgent.goal = agent.getGoal()
                virtualAgent.goalMargins = agent.getGoalMargins()
                virtualAgent.setRl(
                    partial(monteCarloSearch, actions=getRandomActions(a, actions, maxDepth),
                            callback=lambda q: (qs[a].append(q), queue_.get(), queue_.task_done())))

                queue_.put(a)
                virtualAgent.start()

            queue_.join()

            yield actions[np.argmax([np.average(qs[a]) for a in actions])]
            r, nextState, isTerminal = (yield)

            f = 1 / (nextState.lastUpdate - initialState.lastUpdate)
            # correct for deviations from desired freq.
            model.frequency = f

            yield agent.logger.info((nextState.goal, int(f)))


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
    agent.setGoalMargins(position=np.array([0.5, 0.5, math.inf]))
    agent.start()
    agent.join()


if __name__ == '__main__':
    main()
