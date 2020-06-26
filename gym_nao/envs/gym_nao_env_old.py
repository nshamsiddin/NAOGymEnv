import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box, Tuple, Discrete

import vision_definitions as vd
from naoqi import ALProxy
import qi

import motion as mtn
import numpy as np
import time as tm
import math
import almath
import argparse
import logging
from cv2 import cv2

from PIL import Image

logger = logging.getLogger(__name__)


IP = '127.0.0.1'
PORT = 9559

CAM_SPECS = {
	0: {'focal_legth' : 569, 'angle_x' : 60.97, 'angle_y' : 47.64, 'res' : vd.kVGA, 'fps' : 20, 'height' : 54, 'color_space' : vd.kRGBColorSpace},
	1: {'focal_legth' : 415, 'angle_x' : 60.97, 'angle_y' : 47.64, 'res' : vd.k16VGA, 'fps' : 20, 'height' : 48, 'color_space' : vd.kRGBColorSpace},
}

COLOR_BOUNDS = {
	'red' : [(0, 100, 100), (10, 255, 255)],
	'blue' : [(120,50,50), (130,255,255)]
}

HOUGH_PARAMS = {
	'dp': 1,
	'minDist': 10,
	'param1': 50,
	'param2': 30,
	'minRadius' : 0,
	'maxRadius' : 0
}


HEAD_PARAMS = [
	{'angle' : [60.0 * almath.TO_RAD, 10.0 * almath.TO_RAD], 'time' : 1.0},
	{'angle' : [0.0, 25.0 * almath.TO_RAD], 'time' : 2.0},
	{'angle' : [-60.0 * almath.TO_RAD, 10.0 * almath.TO_RAD], 'time' : 3.0},
	{'angle' : [0.0, 0.0], 'time' : 4.0},
]

names  = ["HeadYaw","HeadPitch"]

BALL_SIZE = 20

motion_proxy = ALProxy('ALMotion', IP, PORT)
cam_proxy = ALProxy('ALVideoDevice', IP, PORT)
posture_proxy = ALProxy('ALRobotPosture', IP, PORT)
# motion_proxy.angleInterpolationWithSpeed()

# ses = qi.Session()
# ses.connect(IP)
# per = qi.PeriodicTask()
# motion_proxy = ses.service('ALMotion')
# posture_proxy = ses.service('ALRobotPosture')
# cam_proxy = ses.service('ALVideoDevice')


class GymNaoEnv(gym.Env):

	############### gym standard methods ###############
	metadata = {'render.modes': ['human', 'approach_blue', 'avoid_red', 'combined']}
	
	# initialize the calss and set initial state
	def __init__(self):

		# penalty when getting closer
		self.reward = 0.0
		self.prev_reward = 0.0
		self.step_index = 0

		self.red = 0
		self.blue = 0

		self.prev_red = 0
		self.prev_blue = 0
		
		self.x = 0.0
		self.angle = 0.0
		self.kick = 0
		
		# all possible actions for an agent (forward, turn, kick)
		# self.action_space = spaces.Box(low = np.array([+0.0, (-math.pi/12), 0.0]), high = np.array([+0.4, (+math.pi/12), 1.0]), dtype = np.float64)
		self.action_space = Tuple(spaces = (Box(low = np.array([+0.0, (-math.pi/12)]), high = np.array([+0.4, (+math.pi/12)]), dtype = np.float64),
		Discrete(2)))

		# environment's data to be observed by the agent; 
		# need to check with raw pixels as well
		N = 100 # number of balls
		MAX = 1000 # distance to balls
		self.observation_space = spaces.Box(low = 0, high = MAX, shape=[N, 2], dtype = np.int16)

	def step(self, action):
		self.step_index += 1
		step_reward = 0.0
		print('TEST')
		# DON'T FORGET ABOUT KICK AS WELL
		self.x = action[0]
		self.angle = action[1]
		self._move()

		image = self._take_image(0)
		self.red, self.blue = self._scan_surroundings()
		self.prev_red = self.red
		self.prev_blue = self.blue
		print(self.red, self.blue)

		self.state = np.array(image)

		reward_factor = np.mean(self.blue) - np.mean(self.red)/10

		done = False

		if action is not None:

			self.reward -= 1.0
			step_reward = self.reward - self.prev_reward + float(reward_factor/1000)

			self.prev_reward = self.reward
			
			if reward_factor >= 100:
				if np.mean(self.blue) < 100:
					self.reward += 100.0
				else:
					self.reward -= 100.0
					step_reward = -10

				done = True

		print("step_reward: " + str(step_reward) + "\n")
		return self.state, step_reward, done, {}

	# reset and return state
	def reset(self):
		motion_proxy.wakeUp()
		self.reward = 0.0

		image = self._take_image(0)
		self.prev_red, self.prev_blue = self._scan_surroundings()
		print('RED')
		print(self.prev_red)
		print('BLUE')
		print(self.prev_blue)

		# self.prev_red, self.prev_blue = self._process()

		self.state = [self.prev_red, self.prev_blue]
		print(self.state)
		
		return self.step(None)[0]

	# visualize state of environment
	def render(self, mode='human', close=False):
		pass


	############### custom methods ###############
	
	def _scan_surroundings(self):
		# scan environment
		motion_proxy.setStiffnesses("Head", 1.0)
		red = []
		blue = []
		# print(red, blue)

		(red, blue).append(self._turn_n_shot(0))
		(red, blue).append(self._turn_n_shot(0))
		(red, blue).append(self._turn_n_shot(0))

	

		# for param in HEAD_PARAMS:
		# 	angle = param['angle']
		# 	time = param['time']
		# 	# motion_proxy.angleInterpolationWithSpeed(names, angle, time, absolute)
		# 	motion_proxy.angleInterpolationWithSpeed(names, angle, 0.2)
		# 	# motion.angleInterpolationWithSpeed("Head", [-maxAngleScan, 0.035], 0.1)
		# 	tm.sleep(0.5)
		# 	if time != 4.0:
		# 		for i in [0, 1]:
		# 			filename = 'img_' + str(int(time)) + ('_top' if i == 0 else '_bottom')  + '.png' 
		# 			self._take_image(i, filename)
		# 			# red.append(self._get_balls(filename, 'red', i))
		# 			# blue.append(self._get_balls(filename, 'blue', i))
		# 			for r in self._get_balls(filename, 'red', i):
		# 				red.append(r)
		# 			for b in self._get_balls(filename, 'blue', i):
		# 				blue.append(b)
		motion_proxy.setStiffnesses("Head", 0.0)
		return red, blue

	def _turn_n_shot(self, i):
		red = []
		blue = []

		angle, time, filename0, filename1 = self._get_params(i)
		motion_proxy.angleInterpolationWithSpeed(names, angle, time)

		self._take_image(0, filename0) # top camera
		self._take_image(1, filename1) # bottom camera

		for r in self._get_balls(filename0, 'red', 0):
			red.append(r)
		for r in self._get_balls(filename0, 'red', 0):
			red.append(r)
		for b in self._get_balls(filename1, 'blue', 1):
			blue.append(b)
		for b in self._get_balls(filename1, 'blue', 1):
			blue.append(b)

		return red, blue

	def _take_image(self, cam_id, filename = 'default.png'):
		# filename = 'cam_' + cam_id + '_' + str(int(time)) + '.png'

		res = CAM_SPECS[cam_id]['res']
		fps = CAM_SPECS[cam_id]['fps']
		color_space = CAM_SPECS[cam_id]['color_space']

		cam_proxy.setParam(vd.kCameraSelectID, cam_id)
		videoClient = cam_proxy.subscribe('python', res, color_space, fps)
		naoImage = cam_proxy.getImageRemote(videoClient)
		cam_proxy.unsubscribe(videoClient)

		width = naoImage[0]
		height = naoImage[1]
		array = naoImage[6]

		image = Image.frombytes('RGB', (width, height), array)
		image.save(filename, 'PNG')

	def _get_balls(self, filename, color, cam_id):
		# print('looking for ' + color)
		image = cv2.imread(filename)

		bounds = COLOR_BOUNDS[color]

		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		bright = cv2.inRange(hsv, bounds[0], bounds[1])
		blur = cv2.GaussianBlur(bright, (9, 9), 3, 3)
		
		# cv2.imshow('circled_orig', blur)
		# cv2.waitKey(0)

		detected_circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, **HOUGH_PARAMS)
		balls = []

		if detected_circles is not None:
			for circle in detected_circles[0, :]:
				x = circle[0]
				y = circle[1]
				r = circle[2]
				# circled_orig = cv2.circle(image, (x, y), r, (0, 255, 0), thickness = 2)
				
				# locate detected balls
				distance = self._find_distance(x, y, r, cam_id)
				balls.append(distance)
				# balls.append((x, y, r, color))
				# balls.append()
			# cv2.imshow('circled_orig', circled_orig)
			# cv2.waitKey(0)
		# print('found')
		# print(balls)
		return balls

	def _find_distance(self, x, y, r, cam_id = 1):
		# focal length calculated with F = (PxD)/W where
		# W - actual width of an object (diameter of a ball)
		# D - distance to the object
		# P - width of the object in the image
		W = BALL_SIZE
		F = CAM_SPECS[cam_id]['focal_legth']
		# from formula above we can find distance D = (WxF)/P
		D = (W * F) / (2 * r)
		return D

	def _get_average(self):
		
		pass

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _move(self):
		motion_proxy.moveTo(float(self.x), float(0.0), float(self.angle))
	
	def _kick(self, leg):
		posture_proxy.goToPosture('StandInit', 0.5)

		motion_proxy.wbEnable(True)

		stateName = 'Fixed'
		supportLeg = 'Legs'
		motion_proxy.wbFootState(stateName, supportLeg)

		supportLeg = 'Legs'
		motion_proxy.wbEnableBalanceConstraint(True, supportLeg)

		supportLeg = 'LLeg' if leg == 'R' else 'RLeg'
		duration = 2.0
		motion_proxy.wbGoToBalance(supportLeg, duration)

		stateName = 'Free'
		supportLeg = 'RLeg' if leg == 'R' else 'LLeg'
		motion_proxy.wbFootState(stateName, supportLeg)

		effectorName = 'RLeg' if leg == 'R' else 'LLeg'
		axisMask = 63
		space = mtn.FRAME_WORLD

		dx = 0.15
		dz = 0.09
		dwy = 5.0 * math.pi / 180. # rotation axis Y (radian)
		times = [1.0, 1.7, 3.5]
		isAbsolute = False

		targetList = [
			[-dx, 0.0, dz, 0.0, +dwy, 0.0],
			[+dx, 0.0, dz, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
		motion_proxy.positionInterpolation(effectorName, space, targetList, axisMask, times, isAbsolute)

		motion_proxy.wbEnable(False)

		posture_proxy.goToPosture('StandInit', 0.5)

	def _get_params(self, i):
		angle = HEAD_PARAMS[i]['angle']
		time = HEAD_PARAMS[i]['time']
		filename0 = 'img_' + str(int(time)) + '_top.png'
		filename1 = 'img_' + str(int(time)) + '_bottom.png'
		return angle, time, filename0, filename1

gym = GymNaoEnv()
gym.reset()
