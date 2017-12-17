from Common import *
from Utilities import *
from AgentHelpers import *
from abc import ABCMeta


def isVirtualAgent(method):
    methodName = method.__name__

    def initialize(self):
        self.state = self.alternativeModel.initialize(self.initialState)
        actions = ['moveForward', 'yawCW', 'yawCCW', 'hover']
        self.actions = {a: lambda: None for a in actions}

    def isTerminal(self):
        return self.timeStep >= self.maxDepth

    def updateState(self):
        self.state = self.alternativeModel.updateState(state=self.getState(), action=self.getCurrentAction())

    def doNothing(*args, **kwargs):
        pass

    methods = {'initialize': initialize, 'isTerminal': isTerminal, 'updateState': updateState}

    f1, f2 = method, methods.get(methodName, doNothing)

    def selector(*args):
        self = args[0]
        if self.alternativeModel:
            return f2(*args)

        else:
            return f1(*args)

    return selector


class RLAgent(threading.Thread):
    __metaclass__ = ABCMeta

    def __init__(self, name, alternativeModel=None, maxDepth=20, initialState=None, serverIpAddress='127.0.0.1',
                 defaultSpeed=3, defaultAltitude=1.5, yawRate=70, decisionFrequency=10, learningRate=0.01, discount=1,
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

        self.alternativeModel, self.initialState, self.maxDepth, self.timeStep =\
            alternativeModel, initialState, maxDepth, 0

        self.reward = self.rl = lambda: 0

    @isVirtualAgent
    def initialize(self):
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

        # enable programmatic control of the multi-rotor and reset collision flag
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

    @isVirtualAgent
    def isTerminal(self):
        return len([True for i in self.isTerminalConditions if i(agent=self)])

    def addTerminal(self, condition):
        self.isTerminalConditions.append(condition)

    def getState(self):
        return self.state

    def getGoal(self):
        return self.goal

    @isVirtualAgent
    def updateState(self):
        self.state = self.state.updateState(self)

    def getActions(self, all=False):
        # all parameter=true will return all actions the agents have regardless of context,
        # as some actions might not be accessible to the agent depending on state (feature to be completed)
        return list(self.actions.keys())

    @isVirtualAgent
    def performAction(self, action):
        try:
            action()
        except Exception as e:
            self.logger.critical(e)

    def moveByVelocityZ(self, vx=0, vy=0, yawMode=YawMode(True, 0), drivetrain=DrivetrainType.MaxDegreeOfFreedom,
                        duration=10.0):

        self.client.moveByVelocityZ(vx=vx, vy=vy, z=-self.defaultAltitude, yaw_mode=yawMode, duration=duration,
                                    drivetrain=drivetrain)

    def hover(self):
        self.performAction(self.moveByVelocityZ())

    def moveForward(self):
        velocityBody = np.array([self.defaultSpeed, 0, 0])
        # temporary hack for performance
        velocityEarth = transformToEarthFrame(velocityBody, self.getState().orientation)
        self.performAction(partial(self.moveByVelocityZ, vx=velocityEarth[0], vy=velocityEarth[1]))

    def yaw(self, rate):
        # temporary hack for performance
        self.performAction(partial(self.actions['hover'], vx=0.0, vy=0.0, yawMode=YawMode(True, rate)))

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

    @isVirtualAgent
    def reset(self):
        self.logger.info("Resetting")
        # resetting environment , the old way. make sure simulator window is active!
        #self.shell.SendKeys('\b')

    def run(self, error=0):
        self.initialize()
        callback = self.rl()

        self.timeStep, period = 0, 1 / self.decisionFrequency
        stateKeys = self.getState().getKeys()

        while not self.isTerminal():
            self.timeStep += 1
            start = time.time()

            # give turn to the agent
            a = next(callback)
            self.setCurrentAction(a)
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

            # if self.timeStep % self.decisionFrequency == 0:
            #     self.logger.debug((['{}={}'.format(key, getattr(s, key)) for key in stateKeys], isTerminal))

        # disarm agent
        # self.performAction(partial(self.client.armDisarm, False))
        # reset environment
        # self.reset()
        # get agent into initial state
        # self.initializeState()

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
