import pickle
from Common import *
from Utilities import *
from abc import ABCMeta, abstractmethod


class RLAgent(threading.Thread):
    __metaclass__ = ABCMeta

    def __init__(self, name, model=None, maxDepth=20, initialState=None, serverIpAddress='127.0.0.1', defaultSpeed=3, defaultAltitude=1.5,
                 yawRate=70, decisionFrequency=10, learningRate=0.01, discount=1, crashRecoveryPeriod=16, logFlight=False,
                 logFileName=getDateTime().strip()):

        threading.Thread.__init__(self, name=name)

        self.logger = createConsoleLogger(name)
        self.logFlight, self.logFileName = logFlight, logFileName

        self.shell, self.client, self.actions, self.serverIpAddress = None, None, None, serverIpAddress
        self.yawRate, self.defaultSpeed, self.defaultAltitude = yawRate, defaultSpeed, defaultAltitude

        self.decisionFrequency, self.learningRate, self.discount = decisionFrequency, learningRate, discount
        self.state = self.goal = self.goalMargins = None
        self.currentAction = None
        self.isTerminalConditions = [self.isGoal, self.getCollisionInfo]
        self.hitObstacleFlag = False
        self.crashRecoveryPeriod = crashRecoveryPeriod

        self.keyMap, self.keyPressed = None, KeyT()
        self.keyboardListener = keyboard.Listener(on_press=partial(self.onPress, token=self.keyPressed),
                                                  on_release=partial(self.onRelease, token=self.keyPressed))

        self.keyboardListener.start()
        self.model = model
        self.maxDepth, self.timeStep = maxDepth, 0
        self.initialState = initialState

        if self.model:
            self.performAction = lambda _: None
            self.reset = lambda: None
            self.isTerminal = lambda: self.timeStep >= self.maxDepth

    def initialize(self):
        if self.model:
            self.state = self.model.initialize(self.initialState)
            self.actions = {'moveForward': None, 'yawCW': None, 'yawCCW': None, 'hover': None}
        else:
            self.initializeConnection()
            moveByVelocityZ = partial(self.client.moveByVelocityZ, vx=0, vy=0, z=-self.defaultAltitude, yaw_mode=YawMode(True, 0),
                                      duration=10.0, drivetrain=DrivetrainType.MaxDegreeOfFreedom)

            self.actions = {'moveForward': self.moveForward, 'yawCW': partial(self.yaw, self.yawRate),
                            'yawCCW': partial(self.yaw, -self.yawRate), 'hover': moveByVelocityZ}

            self.initializeState()

    def initializeConnection(self):
        try:
            client = MultirotorClient()
            client.confirmConnection()

        except Exception as e:
            self.logger.critical(e)
            sys.exit(1)

        self.client = client

    def initializeState(self):
        time.sleep(self.crashRecoveryPeriod)
        self.performAction(partial(self.client.enableApiControl, True))
        self.performAction(partial(self.client.armDisarm, True))
        self.performAction(self.client.takeoff)

        # wait for steady-state hover and update state
        time.sleep(5)
        self.updateState()

        # enable programmatic control of the multirotor and reset collision flag
        self.hitObstacleFlag = False
        self.logger.info("Ready")

    def defineState(self, **kwargs):
        self.state = State(update=True, **kwargs)

    def setGoal(self, **kwargs):
        self.goal = State(update=False, **kwargs)

    def setGoalMargins(self, **kwargs):
        self.goalMargins = State(update=False, **kwargs)

    def getGoalMargins(self):
        return self.goalMargins

    @staticmethod
    def getPosition(**kwargs):
        return kwargs['agent'].client.getPosition().toNumpyArray()

    @staticmethod
    def getOrientation(**kwargs):
        return Quaternion(kwargs['agent'].client.getOrientation().toNumpyArray())

    @staticmethod
    def getVelocity(**kwargs):
        return kwargs['agent'].client.getVelocity().toNumpyArray()

    @staticmethod
    def getAngularVelocity(**kwargs):
        return kwargs['agent'].client.getAngularVelocity().toNumpyArray()

    @staticmethod
    def getAngularAcceleration(**kwargs):
        return kwargs['agent'].client.getAngularAcceleration().toNumpyArray()

    @staticmethod
    def getLinearAcceleration(**kwargs):
        return kwargs['agent'].client.getLinearAcceleration().toNumpyArray()

    @staticmethod
    def getCurrentAction(**kwargs):
        return kwargs['agent'].currentAction

    @staticmethod
    def isGoal(agent):
        return agent.getState().areEqual(agent.goal, agent.goalMargins) if agent.getGoal() else False

    @staticmethod
    def getCollisionInfo(agent):
        return agent.hitObstacleFlag

    @staticmethod
    def getHorizontalDistance(p1, p2):
        return ((p1.x - p2.x) ** 2 +
                (p1.y - p2.y) ** 2) ** 0.5

    @staticmethod
    def onPress(key, token):
        token.update(key)

    @staticmethod
    def onRelease(key, token):
        token.clear()

    def moveForward(self):
        # The attitude in the aeronautical frame (right-handed, Z-down, X-front, Y-right).
        velocityEarth = np.array([self.defaultSpeed, 0, 0])

        # temporary hack for performance
        velocityBody = transformToBodyFrame(velocityEarth, self.getState().orientation)
        self.performAction(partial(self.actions['hover'], vx=velocityBody[0], vy=velocityBody[1]))

    def yaw(self, rate):
        # temporary hack for performance
        velocityEarth = self.getState().linearVelocity
        self.performAction(partial(self.actions['hover'], vx=velocityEarth[0], vy=velocityEarth[1], yaw_mode=YawMode(True, rate)))

    def isTerminal(self):
        return len([True for i in self.isTerminalConditions if i(agent=self)])

    def addTerminal(self, condition):
        self.isTerminalConditions.append(condition)

    def getState(self):
        return self.state

    def getGoal(self):
        return self.goal

    def updateState(self):
        if self.model:
            self.state = self.model.updateState(state=self.getState(), action=self.currentAction)
        else:
            self.state = self.state.updateState(self)

    def getActions(self, all=False):
        # all parameter=true will return all actions the agents have regardless of context, as some actions might not be accessible to the
        # agent depending on state (feature to be completed)
        return list(self.actions.keys())

    def performAction(self, action):
        try:
            action()
        except Exception as e:
            self.logger.critical(e)

    def setRl(self, callback):
        # set rl function = client callback
        f = partial(callback, agent=self)
        # for the code to work rl callback has to return a generator
        assert isinstance(f(), types.GeneratorType)
        self.rl = f

    def setReward(self, callback):
        # set reward function = client callback
        f = partial(callback, agent=self)
        self.reward = f

    def reward(self):
        return 0

    def reset(self):
        self.logger.info("Resetting")
        # resetting environment , the old way
        # make sure simulator window is active
        self.shell.SendKeys('\b')

    def run(self):
        self.initialize()
        callback = self.rl()

        logFileStream = open('datasets/' + self.logFileName, 'wb') if self.logFlight else None
        logToFile = lambda data: pickle.dump(data, logFileStream, protocol=pickle.HIGHEST_PROTOCOL)
        flightLogger = logToFile if logFileStream else lambda _: False

        # write log file header
        stateKeys = self.getState().getKeys()
        flightLogger(stateKeys)

        self.timeStep, period, error = 0, 1 / self.decisionFrequency, 0
        while True:
            if self.model is None and self.timeStep == 0:
                self.updateState()
            start = time.time()
            # give turn to the agent
            a = callback.__next__()
            # perform action selected by the agent
            self.performAction(self.actions[a])
            self.currentAction = a

            # delay to match agent's decision freq.
            while time.time() - start < period - error:
                continue

            # state is lazily updated by the environment as the agent needs it , agent always get the freshest estimate of the state
            # state updates are done by the environment in a rate that corresponds to agent decision making freq.
            self.updateState()

            s, r, isTerminal = self.getState(), self.reward(), self.isTerminal()

            # send agent the the transition reward, new state and isTerminal and wait until the agent yields (OK signal)
            callback.__next__()
            callback.send((r, s, isTerminal))

            if self.timeStep % self.decisionFrequency == 0:
                self.logger.debug((['{}={}'.format(key, getattr(s, key)) for key in stateKeys], isTerminal))

            flightLogger(s)

            if self.isTerminal():
                if self.model:
                    break
                # disarm agent
                self.performAction(self.client.disarm)
                # reset environment
                self.reset()
                # get agent into initial state
                self.initializeState()

            self.timeStep += 1

    def saveProgress(self, progress, fileName, append=False):
        try:
            with open(fileName, 'a+b' if append else 'wb') as f:
                pickle.dump(progress, f, protocol=pickle.HIGHEST_PROTOCOL)

        except Exception as e:
            self.logger.critical(e)

        return time.time()

    def loadProgress(self, fileName):
        lastCheckPoint = None
        try:
            with open(fileName, 'rb') as handle:
                lastCheckPoint = pickle.load(handle)
        except Exception as e:
            self.logger.critical(e)

        return lastCheckPoint

    # Must be a generator!
    # Old implementation, currently unused
    def rl(self):
        pass
