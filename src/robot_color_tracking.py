from scipy import *
from pylab import *
from numpy import *
from abc import ABC, abstractmethod
from enum import Enum
from PIL import Image
import cv2
from scipy.ndimage import measurements, morphology

#https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/

class Colors(Enum):
	red = 0
	green = 60
	blue = 120

	yellow = 30
	cyan = 90
	magenta = 150

	orange = 15
	chartreuse = 45
	springGreen = 75
	skyBlue = 105
	violet = 135
	pink = 165

class RobotTracking(ABC):
	def __init__(self):

		self._image = None
		self._nbr_objects = {}
		self._pose = {}
		self._robotID = []

	@abstractmethod
	def track(self,image_name):
		pass

	def printRobotLocation(self):
		figure(figsize=(50,50))
		gray()
		imshow(self._image)

		for robotID in self._robotID:
			for i in range(self._nbr_objects[robotID]):
			    text(self._pose[robotID][i][0], self._pose[robotID][i][1], robotID, color='red', horizontalalignment='center',verticalalignment='center')

	def getPoses(self):
		if len(self._pose)>0:
			return self._pose
		else:
			print('There is no poses calculated yet')
			return []

	def getPoseByID(self, robotID):
		return self._pose[robotID]

	#Define debug functions:

class ColorTrack(RobotTracking):
	def __init__(self,nbr_colors = 3, binaryThreshold = 110, hueTolerance = 7, satTolerance = 60, kernel=ones((20,20)), debug = False):
		super().__init__()

		self.binaryThreshold= binaryThreshold
		self.hueTolerance = hueTolerance
		self.satTolerance = satTolerance
		self.kernel = kernel
		self._debug = debug

		self._colors = []
		i = 0
		for color in Colors:
			self._robotID.append(color.name)
			self._colors.append(color)
			i+=1
			if i >= nbr_colors:
				break

		self._labels = {}
		self._segmentedImages = {}

	def _segmentColor(self,image, color):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		if color == 0:
			image[:,:,2] = (image[:,:,0] >= 180-self.hueTolerance) * image[:,:,2] + (image[:,:,0] < 0+self.hueTolerance) * image[:,:,2]
		else:
			image[:,:,2] = (image[:,:,0] >= color - self.hueTolerance) * image[:,:,2]
			image[:,:,2] = (image[:,:,0] < color + self.hueTolerance) * image[:,:,2]
		image[:,:,2] = (image[:,:,1] > self.satTolerance) * image[:,:,2]
		image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_HSV2BGR),cv2.COLOR_BGR2GRAY)

		return image

	def  _filterImage(self,image):
		image = 1*(image>self.binaryThreshold)
		im_open = morphology.binary_opening(image, self.kernel, iterations=1)
		im_closed = morphology.binary_closing(im_open, self.kernel)

		return im_closed

	def _trackByColor(self,image, color):
		image = self._segmentColor(image, color.value)
		if(self._debug):
			self._segmentedImages[color.name] = image
		image = self._filterImage(image)
		labels, nbr_objects = measurements.label(image)
		center_of_mass = array(measurements.center_of_mass(image, labels=labels, index=range(1,nbr_objects+1) ), dtype=float)

		return flip(center_of_mass, 1), 1*(labels!=0), nbr_objects


	def track(self,image_name):
		image = cv2.imread(image_name)
		self._image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		for color in self._colors:
			if(self._debug):
				self._pose[color.name], self._labels[color.name], self._nbr_objects[color.name] = self._trackByColor(image, color)
			else:
				self._pose[color.name], _, self._nbr_objects[color.name] = self._trackByColor(image, color)

	# debug functions
	def printLabel(self, nbr):
		if(self._debug):
			imshow(self._labels[self._colors[nbr].name])
		else:
			print('Use debug=True for utilizing this method')
	def printSegmentedImage(self, nbr):
		if(self._debug):
			imshow(self._segmentedImages[self._colors[nbr].name])
		else:
			print('Use debug=True for utilizing this method')

class HoughColorTrack(RobotTracking):
	def __init__(self, nbr_colors = 3, binaryThreshold = 110, hueTolerance = 7, satTolerance = 60, debug = False):
		super().__init__()

		self.binaryThreshold= binaryThreshold
		self.hueTolerance = hueTolerance
		self.satTolerance = satTolerance

		self._debug = debug

		self._colors = []
		i = 0
		for color in Colors:
			self._robotID.append(color.name)
			self._colors.append(color)
			i+=1
			if i >= nbr_colors:
				break

		self._segmentedImages = {}
	def _segmentColor(self,image, color):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		if color == 0:
			image[:,:,2] = (image[:,:,0] >= 180-self.hueTolerance) * image[:,:,2] + (image[:,:,0] < 0+self.hueTolerance) * image[:,:,2]
		else:
			image[:,:,2] = (image[:,:,0] >= color - self.hueTolerance) * image[:,:,2]
			image[:,:,2] = (image[:,:,0] < color + self.hueTolerance) * image[:,:,2]
		image[:,:,2] = (image[:,:,1] > self.satTolerance) * image[:,:,2]
		image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_HSV2BGR),cv2.COLOR_BGR2GRAY)

		return image

	def _trackByColorHough(self, image, color):
		image = self._segmentColor(image, color.value)
		if(self._debug):
			self._segmentedImages[color.name] = image
		image = cv2.medianBlur(image, 5)
		rows = image.shape[0]
		circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=30, param2=20, minRadius=100, maxRadius=500)
		if type(circles) == type(None):
			circles = array([[]])
		#print(color.name)
		#print(str(circles.shape)+"    "+ str(circles[0, :].shape[0]))
    	#labels debug
		return circles[0, :], circles[0, :].shape[0]

	def track(self,image_name):
		image = cv2.imread(image_name)
		self._image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		for color in self._colors:
			self._pose[color.name], self._nbr_objects[color.name] = self._trackByColorHough(image, color)

	#debug functions
	def printSegmentedImage(self, nbr):
		if(self._debug):
			imshow(self._segmentedImages[self._colors[nbr].name])
		else:
			print('Use debug=True for utilizing this method')