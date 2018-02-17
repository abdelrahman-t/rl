from rl_agent import *
from collections import defaultdict
from sklearn.externals import joblib

from time import sleep
from pynput import keyboard
import pygame

action_names = ['move_forward', 'yaw_ccw', 'yaw_cw', 'hover']


def verify_model(frequency, state):
    keymap = defaultdict(lambda: 'hover')
    keymap.update([('Key.up', 'move_forward'), ('Key.left', 'yaw_ccw'), ('Key.right', 'yaw_cw'), ('Key.down', 'hover')])

    key_pressed = KeyT()
    keyboard_listener = keyboard.Listener(on_press=key_pressed.on_press,
                                          on_release=key_pressed.on_release)
    keyboard_listener.start()

    while True:
        draw_multirotor(state, keymap[key_pressed.value])

        roll, pitch, yaw = to_euler_angles(state.orientation)
        selected_action = [0 if a != keymap[key_pressed.value] else 1 for a in action_names]
        s0 = np.concatenate((state.linear_velocity, state.angular_velocity, [roll, pitch], selected_action))
        s1 = model.predict(s0.reshape(1, -1)).reshape(6, )

        orientation, position, linear_velocity, angular_velocity = \
            next(integrate_trajectory_velocity_body(position=state.position, orientation=state.orientation,
                                                    linear_velocities=[s1[:3]], angular_velocities=[s1[3:6]],
                                                    frequency=[frequency]))

        state = StateT(orientation=orientation,
                       linear_velocity=linear_velocity,
                       angular_velocity=angular_velocity,
                       position=position, update=False)

        sleep(1/frequency)


def draw_multirotor(state, a):
    roll, pitch, yaw = to_euler_angles(state.orientation)
    x, y, z = state.position

    screen.fill((58, 58, 58))

    img_r = pygame.transform.rotate(img, -np.rad2deg(yaw))
    screen.blit(img_r, img_r.get_rect(center=(x * 10, y * 10)))
    textsurface = myfont.render(a, False, (255, 255, 255))
    screen.blit(textsurface, (x * 10, y * 10))

    textsurface = myfont.render(str('20') + 'Hz', False, (255, 255, 255))
    screen.blit(textsurface, (10, 10))

    pygame.display.update()


def main():
    state = StateT(orientation=Quaternion(euler_to_quaternion(0, 0, 0)),
                   linear_velocity=np.array([0, 0, 0]),
                   angular_velocity=np.array([0, 0, 0]),
                   position=np.array([0.0, 0.0, 0.0]), update=False)
    verify_model(10.0, state)



pygame.init()
screen = pygame.display.set_mode((950, 400))
pygame.display.set_caption('2d Model Verification')
img = pygame.image.load('quadcopter.jpg')
pygame.font.init()
myfont = pygame.font.SysFont('Comic Sans MS', 40)
model = joblib.load('models/nn-m.model')
main()
