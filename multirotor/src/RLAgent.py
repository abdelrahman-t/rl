from Common import *
from Utilities import *
from VectorMath import *
from AgentHelpers import *
from abc import ABCMeta


# TODO: refactor
class RLAgent(threading.Thread):
    __metaclass__ = ABCMeta

    def __init__(self, name, model=None, maxDepth=20, initialState=None, serverIpAddress='127.0.0.1', defaultSpeed=3,
                 defaultAltitude=1.5, yawRate=70, decisionFrequency=10, learningRate=0.01, discount=1,
                 crashRecoveryPeriod=16, logFlight=False, logFileName=getDateTime().strip()):

        threading.Thread.__init__(self, name=name)

        self.logger = createConsoleLogger(name)
        self.logFlight, self.logFileName = logFlight, logFileName

        self.serverIpAddress = serverIpAddress
        self.shell = self.client = self.actions = None

        self.yawRate, self.defaultSpeed, self.defaultAltitude = yawRate, defaultSpeed, defaultAltitude

        self.decisionFrequency, self.learningRate, self.discount = decisionFrequency, learningRate, discount
        self.state = self.currentAction = self.goal = self.goalMargins = None

        self.hitObstacleFlag, self.crashRecoveryPeriod = False, crashRecoveryPeriod
        self.isTerminalConditions = [isGoal, getCollisionInfo]

        self.keyMap, self.keyPressed = None, KeyT()
        self.keyboardListener = keyboard.Listener(on_press=partial(onPress, token=self.keyPressed),
                                                  on_release=partial(onRelease, token=self.keyPressed))
        self.keyboardListener.start()

        self.model, self.initialState, self.maxDepth, self.timeStep = model, initialState, maxDepth, 0
        self.reward = self.rl = lambda: 0

        if self.model:
            self.performAction = lambda _: None
            self.reset = lambda: None
            self.isTerminal = lambda: self.timeStep >= self.maxDepth

    def initialize(self):
        if self.model:
            self.state = self.model.initialize(self.initialState)
            actions = ['moveForward', 'yawCW', 'yawCCW', 'hover']
            self.actions = {a: lambda: self.setCurrentAction(action=a) for a in actions}
        else:
            self.initializeConnection()

            self.actions = {'moveForward': self.moveForward, 'yawCW': partial(self.yaw, self.yawRate),
                            'yawCCW': partial(self.yaw, -self.yawRate), 'hover': self.moveByVelocityZ}

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
        self.state = StateT(update=True, **kwargs)

    def setGoal(self, **kwargs):
        self.goal = StateT(update=False, **kwargs)

    def setGoalMargins(self, **kwargs):
        self.goalMargins = StateT(update=False, **kwargs)

    def getGoalMargins(self):
        return self.goalMargins

    def getCurrentAction(self):
        return self.currentAction

    def setCurrentAction(self, action):
        self.currentAction = action

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
            self.state = self.model.updateState(state=self.getState(), action=self.getCurrentAction())
        else:
            self.state = self.state.updateState(self)

    def getActions(self, all=False):
        # all parameter=true will return all actions the agents have regardless of context,
        # as some actions might not be accessible to the agent depending on state (feature to be completed)
        return list(self.actions.keys())

    def performAction(self, action):
        try:
            action()
        except Exception as e:
            self.logger.critical(e)

    def moveByVelocityZ(self):
        return partial(self.client.moveByVelocityZ, vx=0, vy=0, z=-self.defaultAltitude,
                       yaw_mode=YawMode(True, 0),
                       duration=10.0, drivetrain=DrivetrainType.MaxDegreeOfFreedom)

    def hover(self):
        self.performAction(self.moveByVelocityZ())

    def moveForward(self):
        velocityBody = np.array([self.defaultSpeed, 0, 0])
        # temporary hack for performance
        velocityEarth = transformToEarthFrame(velocityBody, self.getState().orientation)
        self.performAction(partial(self.moveByVelocityZ(), vx=velocityEarth[0], vy=velocityEarth[1]))

    def yaw(self, rate):
        # temporary hack for performance
        velocityEarth = self.getState().linearVelocity
        self.performAction(
            partial(self.actions['hover'], vx=velocityEarth[0], vy=velocityEarth[1], yaw_mode=YawMode(True, rate)))

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

    def reset(self):
        self.logger.info("Resetting")
        # resetting environment , the old way. make sure simulator window is active!
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
            a = next(callback)
            # perform action selected by the agent
            self.performAction(self.actions[a])

            # delay to match agent's decision freq.
            while time.time() - start < period - error:
                continue

            self.updateState()

            s, r, isTerminal = self.getState(), self.reward(), self.isTerminal()

            # send agent the the transition reward, new state and isTerminal and wait until the agent yields (OK signal)
            next(callback)
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
