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
