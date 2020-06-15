import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box, Tuple, Discrete
import time
import numpy as np
from cv2 import cv2

from PIL import Image

import vision_definitions as vd
from naoqi import ALProxy


IP = '127.0.0.1'
PORT = 9559

CAM_SPECS = {
	0: {'focal_legth' : 569, 'angle_x' : 60.97, 'angle_y' : 47.64, 'res' : vd.kVGA, 'fps' : 20, 'height' : 54, 'color_space' : vd.kRGBColorSpace},
	1: {'focal_legth' : 415, 'angle_x' : 60.97, 'angle_y' : 47.64, 'res' : vd.k16VGA, 'fps' : 20, 'height' : 48, 'color_space' : vd.kRGBColorSpace},
}

COLOR_BOUNDS = {
	'red' : [(0, 100, 100), (10, 255, 255)],
	'blue' : [(110,50,50), (130,255,255)]
}

HOUGH_PARAMS = {
	'dp': 1,
	'min_dist': 100,
	'param1': 100,
	'param2': 30,
	'min_radius' : 0,
	'max_radius' : 0
}

import motion as mtn
import time
import math
import argparse
import logging
logger = logging.getLogger(__name__)


BALL_SIZE = 20

motion_proxy = ALProxy('ALMotion', IP, PORT)
camProxy = ALProxy('ALVideoDevice', IP, PORT)
posture = ALProxy('ALRobotPosture', IP, PORT)


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
		self.action_space = Tuple(spaces = 
		(Box(low = np.array([+0.0, (-math.pi/12)]), high = np.array([+0.4, (+math.pi/12)]), dtype = np.float64),
		Discrete(2)))

		# environment's data to be observed by the agent; 
		# need to check with raw pixels as well
		N = 100 # number of balls
		MAX = 1000 # distance to balls
		self.observation_space = spaces.Box(low = 0, high = MAX, shape=[N, 2], dtype = np.int16)

	def step(self, action):
		self.step_index += 1
		step_reward = 0.0

		# DON'T FORGET ABOUT KICK AS WELL
		self.x = action[0]
		self.angle = action[1]
		self._move()

		image = self._take_image()
		self.red, self.blue = self._process(image)
		self.prev_red = self.red
		self.prev_blue = self.blue

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

		image = self._take_image()
		self.prev_red, self.prev_blue = self._process(image)

		self.state = np.array(image)

		return self.step(None)[0]

	# visualize state of environment
	def render(self, mode='human', close=False):
		pass




	############### custom methods ###############
	def _get_average(self):
		
		pass

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _move(self):
		print('\nX, angle: {}, {}'.format(float(self.x),float(self.angle)))
		motion_proxy.moveTo(float(self.x), float(0.0), float(self.angle))

	def _take_image(self, cam_id = 1):

		filename = time.strftime('%Y%m%d-%H%M%S') + '.png'

		res = CAM_SPECS[cam_id]['res']
		fps = CAM_SPECS[cam_id]['fps']
		color_space = CAM_SPECS[cam_id]['color_space']

		camProxy.setParam(vd.kCameraSelectID, cam_id)
		videoClient = camProxy.subscribe('python', res, color_space, fps)
		naoImage = camProxy.getImageRemote(videoClient)
		camProxy.unsubscribe(videoClient)

		width = naoImage[0]
		height = naoImage[1]
		array = naoImage[6]

		image = Image.frombytes('RGB', (width, height), array)
		image.save(filename, 'PNG')
		
		return filename

	def _get_balls(self, image, color, bounds, cam_id = 1):

		# converting the input stream into HSV color space
		hsv_conv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

		bright_mask = cv2.inRange(hsv_conv_img, bounds[0], bounds[1])

		blurred_mask = cv2.GaussianBlur(bright_mask, (9, 9), 3, 3)

		# some morphological operations (closing) to remove small blobs
		erode_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		dilate_element = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
		eroded_mask = cv2.erode(blurred_mask, erode_element)
		dilated_mask = cv2.dilate(eroded_mask, dilate_element)

		detected_circles = cv2.HoughCircles(dilated_mask, cv2.HOUGH_GRADIENT, **HOUGH_PARAMS)
		balls = []

		if detected_circles is not None:
			for circle in detected_circles[0, :]:
				x = circle[0]
				y = circle[1]
				r = circle[2]
				circled_orig = cv2.circle(image, (x, y), r, (0, 255, 0), thickness=2)
				
				# locate detected balls
				# https_www.pyimagesearch.com/?url=https%3A%2F%2Fwww.pyimagesearch.com%2F2015%2F01%2F19%2Ffind-distance-camera-objectmarker-using-python-opencv%2F
				distance = self.find_position(x, y, r, cam_id)
				balls.append((x, y, r, color))
			cv2.imshow('circled_orig', circled_orig)
			cv2.waitKey(0)

		return balls

	def _find_position(self, x, y, r, cam_id = 1):
		# focal length calculated with F = (PxD)/W where
		# W - actual width of an object (diameter of a ball)
		# D - distance to the object
		# P - width of the object in the image
		W = BALL_SIZE
		F = CAM_SPECS[cam_id]['focal_legth']
		# from formula above we can find distance D = (WxF)/P
		D = (W * F) / (2 * r)

		# from NAO specs
		position = 0

		return D

	def _process(self, image_path):
		image = cv2.imread(image_path)

		# red balls
		red = self.get_balls(image, 'red', COLOR_BOUNDS['red'])
		# print('Red balls I can see:')
		# print(red)

		# blue balls
		blue = self.get_balls(image, 'blue', COLOR_BOUNDS['blue'])
		# print('Blue balls I can see:')
		# print(blue)

		return red, blue


	def _stiffness_on(self, ):
		# We use the 'Body' name to signify the collection of all joints
		pNames = 'Body'
		pStiffnessLists = 1.0
		pTimeLists = 1.0
		motion_proxy.stiffnessInterpolation(pNames, pStiffnessLists, pTimeLists)

	def _kick(self, leg):
		posture.goToPosture('StandInit', 0.5)

		# Activate Whole Body Balancer
		motion_proxy.wbEnable(True)

		# Legs are constrained fixed
		stateName = 'Fixed'
		supportLeg = 'Legs'
		motion_proxy.wbFootState(stateName, supportLeg)

		# Constraint Balance motion_proxy
		supportLeg = 'Legs'
		motion_proxy.wbEnableBalanceConstraint(True, supportLeg)

		# Com go to LLeg
		supportLeg = 'LLeg' if leg == 'R' else 'RLeg'
		duration = 2.0
		motion_proxy.wbGoToBalance(supportLeg, duration)

		# RLeg is free
		stateName = 'Free'
		supportLeg = 'RLeg' if leg == 'R' else 'LLeg'
		motion_proxy.wbFootState(stateName, supportLeg)

		# RLeg is optimized
		effectorName = 'RLeg' if leg == 'R' else 'LLeg'
		axisMask = 63
		space = mtn.FRAME_WORLD

		# motion_proxy of the RLeg
		dx = 0.15 # translation axis X (meters)
		dz = 0.09 # translation axis Z (meters)
		dwy = 5.0 * math.pi / 180. # rotation axis Y (radian)
		times = [1.0, 1.7, 3.5]
		isAbsolute = False

		targetList = [
			[-dx, 0.0, dz, 0.0, +dwy, 0.0],
			[+dx, 0.0, dz, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
		motion_proxy.positionInterpolation(effectorName, space, targetList, axisMask, times, isAbsolute)

		# Deactivate Whole Body Balancer
		motion_proxy.wbEnable(False)

		# send robot to Pose Init
		posture.goToPosture('StandInit', 0.5)




