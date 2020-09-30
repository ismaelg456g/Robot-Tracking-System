import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from PIL import Image
import cv2
import imutils
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
		plt.figure(figsize=(50,50))
		plt.imshow(self._image)


		for robotID in self._robotID:
			for i in range(self._nbr_objects[robotID]):
				if self._pose[robotID][i].shape[0] == 2 :
					plt.text(self._pose[robotID][i][0], self._pose[robotID][i][1], robotID, color='white', horizontalalignment='center',verticalalignment='center')


	def getPoses(self):
		if len(self._pose)>0:
			return self._pose
		else:
			print('There is no poses calculated yet')
			return {}

	def getPoseByID(self, robotID):
		return self._pose[robotID]

	#Define debug functions:

class ColorTrack(RobotTracking):
	def __init__(self,nbr_colors = 3, binaryThreshold = 110, hueTolerance = 7, satTolerance = 60, kernel=np.ones((20,20)), debug = False):
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
		center_of_mass = np.array(measurements.center_of_mass(image, labels=labels, index=range(1,nbr_objects+1) ), dtype=float)

		return np.flip(center_of_mass, 1), 1*(labels!=0), nbr_objects


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
			plt.imshow(self._labels[self._colors[nbr].name])
		else:
			print('Use debug=True for utilizing this method')
	def printSegmentedImage(self):
		if(self._debug):
			lines = int(len(self._segmentedImage))
			i=1
			for img in self._segmentedImage:
				plt.figure(figsize=(50,50))
				plt.subplot(lines, 1,i)
				plt.imshow(img)
				i+=1
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
			circles = np.array([[[]]])
		#print(color.name)
		#print(str(circles.shape)+"    "+ str(circles[0, :].shape[0]))
    	#labels debug

		return circles[0, :, :2], circles[0, :].shape[0]

	def track(self,image_name):
		image = cv2.imread(image_name)
		self._image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		for color in self._colors:
			self._pose[color.name], self._nbr_objects[color.name] = self._trackByColorHough(image, color)

	#debug functions
	def printSegmentedImage(self):
		if(self._debug):
			lines = int(len(self._segmentedImages))
			i=1
			for img in self._segmentedImages:
				plt.figure(figsize=(50,50))
				plt.subplot(lines, 1,i)
				plt.imshow(self._segmentedImages[img])
				i+=1
		else:
			print('Use debug=True for utilizing this method')


class GeometricTrack(RobotTracking):
	def __init__(self, areaBounds = (500.0, 20000.0), binaryThreshold = 140, sigma = 3, segmentMethod = 'simple', color = 'green', hueTolerance = 12, satTolerance = 60, debug = False):
		super().__init__()
		self.areaBounds = areaBounds
		self.binaryThreshold = binaryThreshold
		self.hueTolerance = hueTolerance
		self.satTolerance = satTolerance
		self.sigma = sigma
		self.segmentMethod = segmentMethod
		if(segmentMethod=='oneColor'):
			if type(color) == str:
				self._color = Colors[color].value
			else:
				print("color must be a string. color will be set to 'green'")
				self._color = 'green'
		elif(segmentMethod == "multipleColors"):
			self._color = []
			if type(color) == list:
				for col in color:
					self._color.append(Colors[col].value)
			else:
				i=0
				for col in Colors:
					self._color.append(col.value)
					i+=1
					if i>=4:
						break
		self._segmentedImage = []

		self._debug = debug
	def _segmentNaive(self, image):
		plt.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		blurred = cv2.GaussianBlur(plt.gray, (5, 5), self.sigma, self.sigma)
		thresh = cv2.threshold(blurred, self.binaryThreshold, 255, cv2.THRESH_BINARY)[1]
		thresh = cv2.bitwise_not(thresh)

		return thresh
	def _segmentColor(self, image, color):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		if color == 0:
			image[:,:,2] = (image[:,:,0] >= 180-self.hueTolerance) * image[:,:,2] + (image[:,:,0] < 0+self.hueTolerance) * image[:,:,2]
		else:
			image[:,:,2] = (image[:,:,0] >= color - self.hueTolerance) * image[:,:,2]
			image[:,:,2] = (image[:,:,0] < color + self.hueTolerance) * image[:,:,2]
		image[:,:,2] = (image[:,:,1] > self.satTolerance) * image[:,:,2]
		image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_HSV2BGR),cv2.COLOR_BGR2GRAY)
		image = cv2.threshold(image, self.binaryThreshold, 255, cv2.THRESH_BINARY)[1]

		return image


	def track(self, image_name):
		self._image = cv2.imread(image_name)
		resized = imutils.resize(self._image, width=300)
		self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
		ratio = self._image.shape[0] / float(resized.shape[0])
		if self.segmentMethod == 'oneColor':
			thresh = self._segmentColor(resized, self._color)
			if self._debug==True:
				self._segmentedImage.append(thresh)
			self._pose = {}
			self._getPose(thresh, ratio)
		elif self.segmentMethod == 'multipleColors':
			shapes = ['triangle', 'square', 'pentagon', 'circle']
			i=0
			self._pose = {}
			for color in self._color:
				thresh = self._segmentColor(resized, color)
				if self._debug==True:
					self._segmentedImage.append(thresh)
				self._getPose(thresh, ratio, shapeToTrack=shapes[i])
				i+=1
		else:
			thresh = self._segmentNaive(resized)
			if self._debug==True:
				self._segmentedImage.append(thresh)
			self._pose = {}
			self._getPose(thresh, ratio)



	def _getPose(self, thresh, ratio, shapeToTrack = ''):
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		for c in cnts:
			# compute the center of the contour, then detect the name of the
			# shape using only the contour
			M = cv2.moments(c)
			if(M["m00"]!=0 and cv2.contourArea(c)>self.areaBounds[0] and cv2.contourArea(c)<self.areaBounds[1]):
				cX = int((M["m10"] / M["m00"]) * ratio)
				cY = int((M["m01"] / M["m00"]) * ratio)
				shape = self._detect(c)
				if(shape == shapeToTrack or shapeToTrack==''):
					if shape in self._pose:
						self._pose[shape].append([cX, cY])
						self._nbr_objects[shape] += 1
					else:
						self._robotID.append(shape)
						self._nbr_objects[shape] = 1
						self._pose[shape] = [[cX, cY]]
	def _detect(self, c):
		# initialize the shape name and approximate the contour
		shape = "unidentified"
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.04 * peri, True)
		# if the shape is a triangle, it will have 3 vertices
		if len(approx) == 3:
			shape = "triangle"
		# if the shape has 4 vertices, it is either a square or
		# a rectangle
		elif len(approx) == 4:
			shape = "square"
		# if the shape is a pentagon, it will have 5 vertices
		elif len(approx) == 5:
			shape = "pentagon"
		# otherwise, we assume the shape is a circle
		else:
			shape = "circle"
		# return the name of the shape
		return shape

	#debug functions
	def printSegmentedImage(self):
		if(self._debug):
			lines = int(len(self._segmentedImage))
			i=1
			for img in self._segmentedImage:
				plt.figure(figsize=(50,50))
				plt.subplot(lines, 1,i)
				plt.imshow(img)
				i+=1
		else:
			print('Use debug=True for utilizing this method')
