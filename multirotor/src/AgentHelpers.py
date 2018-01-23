from Common import *
from VectorMath import *


def getPosition(**kwargs):
    return kwargs['agent'].client.getPosition().toNumpyArray()


def getOrientation(**kwargs):
    return Quaternion(kwargs['agent'].client.getOrientation().toNumpyArray())


def getVelocity(**kwargs):
    return kwargs['agent'].client.getVelocity().toNumpyArray()


def getAngularVelocity(**kwargs):
    return kwargs['agent'].client.getAngularVelocity().toNumpyArray()


def getAngularAcceleration(**kwargs):
    return kwargs['agent'].client.getAngularAcceleration().toNumpyArray()


def getLinearAcceleration(**kwargs):
    return kwargs['agent'].client.getLinearAcceleration().toNumpyArray()


def isGoal(**kwargs):
    return kwargs['agent'].getState().areEqual(kwargs['agent'].goal, kwargs['agent'].goalMargins) if kwargs[
        'agent'].getGoal() else False


def getCollisionInfo(**kwargs):
    return kwargs['agent'].hitObstacleFlag


def getHorizontalDistance(p1, p2):
    return ((p1.x - p2.x) ** 2 +
            (p1.y - p2.y) ** 2) ** 0.5


def getHorizontalDistanceGoal(**kwargs):
    return ((kwargs['agent'].getGoal().position[0] - kwargs['partialUpdate'].position[0]) ** 2 +
            (kwargs['agent'].getGoal().position[1] - kwargs['partialUpdate'].position[1]) ** 2) ** 0.5


def onPress(key, token):
    token.update(key)


def onRelease(key, token):
    token.clear()


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
