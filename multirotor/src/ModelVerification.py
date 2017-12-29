from RLAgent import *
from Common import *
from Model import VelocityModel
import pygame


def verifyModel(agent):
    keymap = defaultdict(lambda: 'hover')
    keymap.update([('Key.up', 'moveForward'), ('Key.left', 'yawCCW'), ('Key.right', 'yawCW'), ('Key.down', 'hover')])

    while True:
        initialState, a = agent.getState(), agent.keyPressed.value
        yield keymap[a]

        r, nextState, isTerminal = (yield)
        draw_multirotor(nextState)

        yield


def draw_multirotor(state):
    roll, pitch, yaw = toEulerianAngle(state.orientation)
    x, y, z = state.position

    screen.fill((58, 58, 58))

    img_r = pygame.transform.rotate(img, -np.rad2deg(yaw))
    screen.blit(img_r, (x*10, y*10))

    pygame.display.update()


def main():
    model = VelocityModel(regressionModel=joblib.load('models/gradient.model'), frequency=10.0)
    agent = RLAgent('agent', decisionFrequency=10.0, defaultSpeed=4, defaultAltitude=6, yawRate=60,
                    alternativeModel=model, maxDepth=math.inf, initialState=None)

    agent.setRl(verifyModel)
    agent.start()
    agent.join()


pygame.init()
screen = pygame.display.set_mode((300, 600))
pygame.display.set_caption('Model Verification')
img = pygame.image.load('quadcopter.jpg')
main()