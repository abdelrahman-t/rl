from collections import defaultdict, OrderedDict, namedtuple, deque
from itertools import starmap
from functools import partial
from pyquaternion import Quaternion
from pynput import keyboard
from datetime import datetime
import numpy as np
from numpy import array, zeros, float32
from copy import deepcopy
import win32com.client
from AirSim import *
import math
import pythoncom
import threading
import win32gui
import warnings
import logging
import sys
import socket
import types
import time
import cv2

warnings.filterwarnings("ignore")
globalLoggingLevel = logging.INFO
