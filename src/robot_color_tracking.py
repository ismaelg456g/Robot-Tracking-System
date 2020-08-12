from scipy import *
from pylab import *
from numpy import *
from enum import Enum
from PIL import Image
import cv2
from scipy.ndimage import measurements, morphology

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

class RobotColorTracking(object):
	def __init__(self,nbr_colors = 3, binaryThreshold = 110, hueTolerance = 7, satTolerance = 60, kernel=ones((20,20))):
		self.binaryThreshold= binaryThreshold
		self.hueTolerance = hueTolerance
		self.satTolerance = satTolerance
		self.kernel = kernel

		self._labels = None
		self._image = None
		self.nbr_objects = None
		self._pose = {}

		self.colors = []
		i = 0
		for color in Colors:
			self.colors.append(color)
			i+=1
			if i >= nbr_colors:
				break

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
		im_closed = morphology.binary_closing(image, self.kernel)

		return im_closed

	def _trackByColor(self,image, color):
		image = self._segmentColor(image, color)
		image = self._filterImage(image)
		self._labels, self.nbr_objects = measurements.label(image)
		center_of_mass = array(measurements.center_of_mass(image, labels=self._labels, index=range(1,self.nbr_objects+1) ), dtype=float)

		return center_of_mass

	def track(self,image_name):
		image = cv2.imread(image_name)
		self._image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		for color in self.colors:
			self._pose[color.name] = self._trackByColor(image, color.value)


	def printRobotLocation(self):
		#if(self._labels == None or self.nbr_objects == None or self.pose == None):
		#	return
		#print(self.pose)
		#print("number of objects: "+str(self.nbr_objects)+"    labels:"+str(self._labels.shape))
		figure(figsize=(50,50))
		gray()
		imshow(self._image)
		for color in self.colors:
			for i in range(self.nbr_objects):
			    text(self._pose[color.name][i][1], self._pose[color.name][i][0], color.name, color='red', horizontalalignment='center',verticalalignment='center')
	# Alterar visibilidade dos atributos
	#problema com nbr_objects
	def getPoses(self):
		if len(self._pose)>0:
			return self._pose
		else:
			print('There is no poses calculated yet')
	def getPoseByColor(self, color):
		return self._pose[color.name]







