from RLAgent import *
from RLEnvironment import *
from Utilities import *


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


def Q(state, actionId):
    f = [e(state) for e in Q.features]
    return np.dot(Q.weights[actionId], f), f


# fix state aliasing caused by the depth sensor
def fixDepthAliasing(agent):
    pass


def eGreedy(state, actionsIds, epsilon=0.05):
    if np.random.rand() > epsilon:
        aMax = np.argmax([Q(state, action)[0] for action in actionsIds])
    else:
        aMax = np.random.randint(0, len(actionsIds) - 1)

    return aMax


# Pseudocode at : http://artint.info/html/ArtInt_272.html
def sarsa(agent, checkpointsIntervals=60 * 5):
    # setup features and initialize weights randomly , each action has a different set of weights for each feature
    f1 = lambda s: s.hDistanceNearestObs
    f2 = lambda s: 1 + s.hDistanceGoal
    features, actions = [f1, f2], agent.getActions(all=True)

    weights = np.random.random((len(actions), len(features)))
    Q.features, Q.weights = features, weights

    lastCheckPoint = time.time()
    while True:
        # observe current state s
        s, isTerminal = agent.getState(), False

        # select action a
        a = eGreedy(s, agent.getActions().keys())

        # Episode
        while isTerminal is False:

            # save progress every once in a while
            if time.time() - lastCheckPoint > checkpointsIntervals:
                lastCheckPoint = agent.saveProgress(weights, 'weights')

            # carry out action a
            yield actions[a]
            # observe reward r and state s1
            r, s1, isTerminal = yield  # yield to apply action and get feedback (feedback delay = 1/decision freq.)

            # select action a' (using a policy based on Qw)
            a1 = eGreedy(s1, agent.getActions().keys())

            # let δ= r+γQw(s',a')-Qw(s,a)
            q0, f0 = Q(s, a)
            q1, f1 = Q(s1, a1)
            td_error = r + (agent.discount * q1) - q0

            # update action a weights
            # for i=0 to n do
            # wi ←wi + ηδFi(s,a)
            for index, value in enumerate(weights[a]):
                weights[a][index] += agent.learningRate * td_error * f0[index]

            # s ←s'
            s = s1

            # a ←a'
            a = a1

            yield  # yield to inform the program that agent has received the most recent feedback
            # End of iteration
            # End of Episode


def main():
    agent = RLAgent('agent', decisionFrequency=15, defaultSpeed=5, discount=1, learningRate=0.05, crashRecoveryPeriod=14)

    # callbacks will be called in the order they were specified, beware of order of execution (if any state parameter is dependant on
    # another)
    # state is lazily updated by the environment as the agent needs it , agent always get the freshest estimate of the state
    # state updates are done by the environment in a rate that corresponds to agent decision making freq.
    agent.defineState(position=RLAgent.getPosition, rollPitchYaw=RLAgent.getRollPitchYaw, hDistanceGoal=hDistanceGoal)

    agent.setGoal(position=np.array([4, 5, 0]), rollPitchYaw=np.array([0, 0, 0]))
    agent.setGoalMargins(position=np.array([10, 10, math.inf]), rollPitchYaw=np.array([math.inf, math.inf, math.inf]))

    # states where agent perception is aliased or impaired should be terminal as agent should avoid these states at all costs
    agent.addTerminal(lambda a: True if a.getState().hDistanceNearestObs == 0 else False)
    agent.setRl(sarsa)
    agent.setReward(reward)
    agent.start()


if __name__ == '__main__':
    main()
