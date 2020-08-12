from scipy import *
from pylab import *
from numpy import *
from enum import Enum
from PIL import Image
import cv2
from scipy.ndimage import measurements, morphology

class Colors(Enum):
	red = 180
	orange = 15
	yellow = 30
	chartreuse = 45
	green = 60
	springGreen = 75
	cyan = 90
	skyBlue = 105
	blue = 120
	violet = 135
	magenta = 150
	pink = 165

class RobotColorTracking(object):
	def __init__(self, binaryThreshold = 110, hueTolerance = 7, satTolerance = 60, kernel=ones((20,20))):
		self.binaryThreshold= binaryThreshold
		self.hueTolerance = hueTolerance
		self.satTolerance = satTolerance
		self.kernel = kernel

		self.labels = None
		self.image = None
		self.nbr_objects = None
		self.pose = None

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
		self.image = image
		self.labels, self.nbr_objects = measurements.label(image)
		center_of_mass = array(measurements.center_of_mass(image, labels=self.labels, index=range(1,self.nbr_objects+1) ), dtype=float)

		return center_of_mass

	def track(self,image_name, colors): #lista de cores
		image = cv2.imread(image_name)
		#for percorrendo cores
		self.pose = self._trackByColor(image, Colors.orange.value) #all colors


	def printRobotLocation(self): #precisa dos labels e nbr_objects
		#if(self.labels == None or self.nbr_objects == None or self.pose == None):
		#	return
		#print(self.pose)
		#print("number of objects: "+str(self.nbr_objects)+"    labels:"+str(self.labels.shape))
		figure(figsize=(50,50))
		gray()
		imshow(self.image)

		for i in range(self.nbr_objects):
		    text(self.pose[i][1], self.pose[i][0], 'Objeto', color='black', horizontalalignment='center',verticalalignment='center')







