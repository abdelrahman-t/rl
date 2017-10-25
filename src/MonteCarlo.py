from Utilities import *
from RLAgent import *
from Model import *

from sklearn.externals import joblib

from queue import Queue


def horizontalDistanceGoal(**kwargs):
    return ((kwargs['agent'].getGoal().position[0] - kwargs['partialUpdate'].position[0]) ** 2 +
            (kwargs['agent'].getGoal().position[1] - kwargs['partialUpdate'].position[1]) ** 2) ** 0.5


def reward(agent):
    terminalStateReward = (1.0e2, -1.0e2)
    wPositionX = wPositionY = 1
    g, s1 = agent.getGoal(), agent.getState()

    if agent.isTerminal():
        r = terminalStateReward[0] if agent.isGoal(agent) else terminalStateReward[1]
    else:
        # −(αx(x − x∗) **2 + αy(y − y∗) **2)
        r = -(wPositionX * (s1.position[0] - g.position[0]) ** 2 + (wPositionY * (s1.position[1] - g.position[1]) ** 2))

    return r


def monteCarloSearch(agent, callback):
    policy, q = lambda: np.random.randint(0, 4), 0.0

    while True:
        yield agent.getActions()[policy()]
        r, nextState, isTerminal = (yield)
        q += r
        if isTerminal:
            callback(q=q)
        yield


# to do: use multiprocessing instead of threading as GIL renders the threaded approach useless!!
# thread-safety is not enforced, code is only thread-safe on CPython
def monteCarlo(agent, maxDepth=15, trials=10, frequency=10):
    model = VelocityModel(model=joblib.load('linearModel.model'), frequency=frequency)
    while True:
        initialState, isTerminal = agent.getState(), False
        while isTerminal is False:
            actions = agent.getActions()
            queue_, qs = Queue(), {i: [] for i in actions}

            for a in np.repeat(actions, trials):
                virtualAgent, isTerminal = RLAgent('virtual', model=model, decisionFrequency=math.inf,
                                                   maxDepth=maxDepth, initialState=initialState), False
                virtualAgent.setReward(reward)
                virtualAgent.goal = agent.getGoal()
                virtualAgent.goalMargins = agent.getGoalMargins()
                virtualAgent.setRl(partial(monteCarloSearch, callback=lambda q: (qs[a].append(q), queue_.get(), queue_.task_done())))

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

    # callbacks will be called in the order they were specified, beware of order of execution (if any state parameter is dependant on
    # another)
    # state is lazily updated by the environment as the agent needs it , agent always get the freshest estimate of the state
    # state updates are done by the environment in a rate that corresponds to agent decision making freq.
    agent.defineState(orientation=RLAgent.getOrientation, angularVelocity=RLAgent.getAngularVelocity, linearVelocity=RLAgent.getVelocity,
                      position=RLAgent.getPosition, goal=horizontalDistanceGoal)

    agent.setRl(monteCarlo)
    agent.setReward(reward)
    agent.setGoal(position=np.array([-40, -50, 0]))
    agent.setGoalMargins(position=np.array([0.5, 0.5, math.inf]))
    agent.start()
    agent.join()


if __name__ == '__main__':
    main()
