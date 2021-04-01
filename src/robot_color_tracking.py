import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from PIL import Image
import cv2
import imutils
from scipy.ndimage import measurements, morphology
from scipy.ndimage import gaussian_filter
import time

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
	def __init__(self, img_width=300):
		self.img_width = img_width
		self._image = None
		self._nbr_objects = {}
		self._pose = {}
		self.angle = {}
		self._robotID = []
		self.time = []

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
			#print('There is no poses calculated yet')
			return {}

	def getPoseByID(self, robotID):
		return self._pose[robotID]
	def getRobotIDs(self):
		return self._robotID
	def getAngle(self, poses):
		poses2 = np.array([poses[1],poses[2],poses[0]])
		dist = ((poses - poses2)**2).sum(axis=1)**(0.5)
		print('poses: '+ str(poses))
		print('poses2: '+ str(poses2))
		print('dist: '+ str(dist))
		p1 = np.delete(poses, dist.argmin()-1, 0)
		p1 = p1.sum(axis=0)/2

		p2 = poses[dist.argmin()-1]
		print('p1: '+ str(p1))
		print('p2: '+ str(p2))

		sub = (p2-p1)*(1,-1)
		print('sub: '+ str(sub))
		angle = np.arctan(sub[1]/sub[0])*360/(2*np.pi)
		if(sub[0]<0 and sub[1]> 0):
			angle+= 180
		elif(sub[0]<0 and sub[1]< 0):
			angle-= 180



		return angle




	#Define debug functions:

class ColorTrack(RobotTracking):
	def __init__(self,img_width=300,colors = [],nbr_colors = 3, binaryThreshold = 110, hueTolerance = 7, satTolerance = 60, kernel=np.ones((20,20)), debug = False):
		super().__init__(img_width)

		self.binaryThreshold= binaryThreshold
		self.hueTolerance = hueTolerance
		self.satTolerance = satTolerance
		self.kernel = kernel
		self._debug = debug
		self._colors = []
		if(len(colors)==0):
			i = 0
			for color in Colors:
				self._robotID.append(color.name)
				self._colors.append(color)
				i+=1
				if i >= nbr_colors:
					break
		else:
			for c in colors:
				self._robotID.append(c)
				self._colors.append(Colors[c])

		self._labels = {}
		self.segmentedImages = {}

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
		
		image = self._filterImage(image)
		

		
		while True:
			labels, nbr_objects = measurements.label(image)
			areas = measurements.sum(image, labels=labels, index=range(1,nbr_objects+1) )
			if(areas.max()>100):
				image = morphology.binary_erosion(image)
			else:
				break
		if(self._debug):
			self.segmentedImages[color.name] = image
		
		center_of_mass = np.array(measurements.center_of_mass(image, labels=labels, index=range(1,nbr_objects+1) ), dtype=float)
		if(len(center_of_mass)!=0):
			return np.flip(center_of_mass, 1), 1*(labels!=0), nbr_objects
		else:
			return [], 1*(labels!=0), nbr_objects

	def postProcessing(self):
		distance_matrix = {}

		for c in self._robotID:
			n_poses = len(self._pose[c])
			if n_poses > 3:
				distance_matrix[c] = np.zeros((n_poses, n_poses))
				# optimize later with c++
				for i in range(n_poses):
					for j in range(i+1, n_poses):
						distance_matrix[c][i,j] = distance_matrix[c][j,i] = ((self._pose[c][i] - self._pose[c][j])**2).sum()**(0.5)
				distance_sum = distance_matrix[c].sum(axis=1)

				while(len(distance_sum)>3):
					argmax = distance_sum.argmax()
					distance_sum = np.delete(distance_sum, argmax, 0)
					self._pose[c] = np.delete(self._pose[c], argmax, 0)
					self._nbr_objects[c] -=1
					# distance_matrix[c] = np.delete(distance_matrix[c], argmax, 0)
					# distance_matrix[c] = np.delete(distance_matrix[c], argmax, 1)
			if len(self._pose[c]) == 3:
				self.angle[c] =  self.getAngle(self._pose[c])




		return distance_matrix



	def track(self,image_name):
		image = cv2.imread(image_name)
		beginning = time.time()
		resized = imutils.resize(image, width=self.img_width)
		# resized = cv2.medianBlur(resized, 5)
		self._image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		ratio = self._image.shape[0] / float(resized.shape[0])
		for color in self._colors:
			if(self._debug):
				self._pose[color.name], self._labels[color.name], self._nbr_objects[color.name] = self._trackByColor(resized, color)
				if(len(self._pose[color.name])!=0):
					self._pose[color.name]*=ratio
			else:
				self._pose[color.name], _, self._nbr_objects[color.name] = self._trackByColor(resized, color)
				if(len(self._pose[color.name])!=0):
					self._pose[color.name]*=ratio
		# self.postProcessing()
		end = time.time()
		self.time.append(end-beginning)
	# debug functions
	def printLabel(self, nbr):
		if(self._debug):
			plt.imshow(self._labels[self._colors[nbr].name])
		else:
			print('Use debug=True for utilizing this method')
	def printSegmentedImage(self):
		if(self._debug):
			lines = int(len(self.segmentedImages))
			i=1
			print(self.segmentedImages)
			for img in self.segmentedImages:
				plt.figure(figsize=(50,50))
				plt.gray()
				plt.subplot(lines, 1,i)
				plt.imshow(img)
				i+=1
		else:
			print('Use debug=True for utilizing this method')

class HoughColorTrack(RobotTracking):
	def __init__(self,img_width=300,colors = [], nbr_colors = 3, binaryThreshold = 110, hueTolerance = 7, satTolerance = 60,param1=30, param2=20, minRadius=100, maxRadius=500, debug = False):
		super().__init__(img_width)
		self.param1=param1
		self.param2=param2
		self.minRadius=minRadius
		self.maxRadius=maxRadius

		self.binaryThreshold= binaryThreshold
		self.hueTolerance = hueTolerance
		self.satTolerance = satTolerance

		self._debug = debug

		self._colors = []
		i = 0
		self._colors = []
		if(len(colors)==0):
			i = 0
			for color in Colors:
				self._robotID.append(color.name)
				self._colors.append(color)
				i+=1
				if i >= nbr_colors:
					break
		else:
			for c in colors:
				self._robotID.append(c)
				self._colors.append(Colors[c])

		self.segmentedImages = {}
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
			self.segmentedImages[color.name] = image
		#image = cv2.medianBlur(image, 5)
		rows = image.shape[0]
		circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=self.param1, param2=self.param2, minRadius=self.minRadius, maxRadius=self.maxRadius)
		if type(circles) == type(None):
			circles = np.array([[[]]])
		#print(color.name)
		#print(str(circles.shape)+"    "+ str(circles[0, :].shape[0]))
    	#labels debug

		return circles[0, :, :2], circles[0, :].shape[0]

	def track(self,image_name):
		image = cv2.imread(image_name)
		beginning = time.time()
		resized = imutils.resize(image, width=self.img_width)
		self._image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		ratio = self._image.shape[0] / float(resized.shape[0])
		for color in self._colors:
			self._pose[color.name], self._nbr_objects[color.name] = self._trackByColorHough(resized, color)
			if(len(self._pose[color.name])!=0):
				self._pose[color.name]*=ratio
		end = time.time()
		self.time.append(end-beginning)
	#debug functions
	def printSegmentedImage(self):
		if(self._debug):
			lines = int(len(self.segmentedImages))
			i=1
			for img in self.segmentedImages:
				plt.figure(figsize=(50,50))
				plt.gray()
				plt.subplot(lines, 1,i)
				plt.imshow(self.segmentedImages[img])
				i+=1
		else:
			print('Use debug=True for utilizing this method')


class GeometricTrack(RobotTracking):
	def __init__(self,img_width=300, areaBounds = (500.0, 20000.0), binaryThreshold = 140, sigma = 3, segmentMethod = 'simple', color = 'green', hueTolerance = 12, satTolerance = 60, debug = False):
		super().__init__(img_width)
		self.areaBounds = areaBounds
		self.binaryThreshold = binaryThreshold
		self.hueTolerance = hueTolerance
		self.satTolerance = satTolerance
		self.sigma = sigma
		self.segmentMethod = segmentMethod
		self._robotID = ['triangle', 'square', 'pentagon', 'circle']

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
		beginning = time.time()
		resized = imutils.resize(self._image, width=self.img_width)
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
		end = time.time()
		self.time.append(end-beginning)


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
						self._pose[shape].append(np.array([cX, cY]))
						self._nbr_objects[shape] += 1
					else:
						self._nbr_objects[shape] = 1
						self._pose[shape] = [np.array([cX, cY])]
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

class LedTrack(ColorTrack):
	def __init__(self):
		super().__init__()
		self.satTolerance = 255
class AchromaticTrack(ColorTrack):
	def __init__(self,convolution=True,img_width=300,colors = [],nbr_colors = 3, binaryThreshold = 30, hueTolerance = 30, satTolerance = 0, kernel=np.ones((20,20)), debug = False, d=30):
		super().__init__(img_width,colors,nbr_colors, binaryThreshold, hueTolerance, satTolerance, kernel, debug)
		self.d = d
		self.convolution = convolution
	def _segmentColor(self,image, color):
		d= self.d

		rgb = image[:,:,0].astype(float) + image[:,:,1].astype(float) + image[:,:,2].astype(float)
		im_new = image.astype(float)

		red = im_new[:,:,0] = 255*im_new[:,:,0]/rgb
		green = im_new[:,:,1] = 255*im_new[:,:,1]/rgb
		blue = im_new[:,:,2] = 255*im_new[:,:,2]/rgb

		
		cad = (np.abs(red-green)+np.abs(green-blue)+np.abs(blue-red))/(3*d)
		im_new = im_new.astype(np.uint8)
		cad = (cad>1)
		im_new[:,:,0] = im_new[:,:,0]*cad
		im_new[:,:,2] = im_new[:,:,2]*cad
		im_new[:,:,1] = im_new[:,:,1]*cad

		# image = cv2.cvtColor(im_new, cv2.COLOR_RGB2GRAY)
		image = cv2.cvtColor(im_new, cv2.COLOR_BGR2HSV)
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
		if(self.convolution):
			im_open = morphology.binary_opening(image, self.kernel, iterations=1)
			im_closed = morphology.binary_closing(im_open, self.kernel)
			return im_closed
		else:
			return image

		

class SignColorTrack(ColorTrack):
	def __init__(self,img_width=300,colors = [],nbr_colors = 3, binaryThreshold = 30, hueTolerance = 30, satTolerance = 0, kernel=np.ones((20,20)), debug = False):
		super().__init__()
	def _segmentColor(self,image, color):
		d=50 #30 is the article standard

		rgb = image[:,:,0].astype(float) + image[:,:,1].astype(float) + image[:,:,2].astype(float)
		im_new = image.astype(float)

		red = im_new[:,:,0] = 255*im_new[:,:,0]/rgb
		green = im_new[:,:,1] = 255*im_new[:,:,1]/rgb
		blue = im_new[:,:,2] = 255*im_new[:,:,2]/rgb

		
		cad = (np.abs(red-green)+np.abs(green-blue)+np.abs(blue-red))/(3*d)
		im_new = im_new.astype(np.uint8)
		cad = (cad>1)
		im_new[:,:,0] = im_new[:,:,0]*cad
		im_new[:,:,2] = im_new[:,:,2]*cad
		im_new[:,:,1] = im_new[:,:,1]*cad

		image = cv2.cvtColor(im_new, cv2.COLOR_BGR2HSV)
		if color == 0:
			image[:,:,2] = (image[:,:,0] >= 180-self.hueTolerance) * image[:,:,2] + (image[:,:,0] < 0+self.hueTolerance) * image[:,:,2]
		else:
			image[:,:,2] = (image[:,:,0] >= color - self.hueTolerance) * image[:,:,2]
			image[:,:,2] = (image[:,:,0] < color + self.hueTolerance) * image[:,:,2]
		image[:,:,2] = (image[:,:,1] > self.satTolerance) * image[:,:,2]
		image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_HSV2BGR),cv2.COLOR_BGR2GRAY)

		return image