from scipy import *
from pylab import *
from numpy import *
from enum import Enum
from PIL import Image
import cv2
from scipy.ndimage import measurements, morphology

class Colors(Enum):
	RED = 180
	ORANGE = 15
	YELLOW = 30
	CHARTREUSE = 45
	GREEN = 60
	SPRINGGREEN = 75
	CYAN = 90
	SKYBLUE = 105
	BLUE = 120
	VIOLET = 135
	MAGENTA = 150
	PINK = 165

class RobotColorTracking(object):
	def __init__(self):
		pass

	def _segmentColor(image, color):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		if color == 0:
			image[:,:,2] = (image[:,:,0] >= 170) * image[:,:,2] + (image[:,:,0] < 10) * image[:,:,2]
		else:
			image[:,:,2] = (image[:,:,0] >= color - 7) * image[:,:,2] # tolerância do matiz
			image[:,:,2] = (image[:,:,0] < color +7) * image[:,:,2]
		image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_HSV2BGR),cv2.COLOR_BGR2GRAY)
		# saturation
		return image

	def  _filterImage(image):
		image = 1*(image>120) # limiar de binarização
		image = 1*(image!=1)
		im_open = morphology.binary_opening(image, ones((9,5)), iterations=1) #kernel
		im_closed = morphology.binary_closing(image, ones((9,5)))

		return im_closed

	def _trackByColor(image, color):
		image = self._segmentColor(image, color)
		image = self._filterImage(image)

		labels, nbr_objects = measurements.label(image)
		center_of_mass = array(measurements.center_of_mass(image, labels=labels, index=range(1,nbr_objects+1) ), dtype=float)

		return center_of_mass # gravar em atributo

	def track(image_name, colors): #lista de cores
		image = cv2.imread(image_name)
		#for percorrendo cores
		pose = _trackByColor(image, Colors.RED) #all colors


	def printRobotLocation(): #precisa dos labels e nbr_objects
		figure(figsize=(20,100))
		imshow(labels)

		for i in range(nbr_objects):
		    text(center_of_mass[i][1], center_of_mass[i][0], 'Objeto', color='black', horizontalalignment='center',verticalalignment='center')







