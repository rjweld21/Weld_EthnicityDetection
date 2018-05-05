from time import sleep
from random import shuffle
from scipy.interpolate import spline, BSpline, make_interp_spline, interp1d
from datetime import datetime
from multiprocessing.dummy import Pool as ThreadPool

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
#import imutils
import dlib
import cv2
import os
import sys
from multiprocessing.dummy import Pool as ThreadPool

### NOTE BEFORE RUNNING ##
# This file was edited to take the extracted frames located in /mnt/yuanjie/rj-cssff-crops/vid_frames_NOT_CROPPED
# And crop the images into /mnt/yuanjie/rj-cssff-crops/frame_crops
# While renaming the images as well to fit standard formatting
##########################

# Sets up argument parser and gets arguments from command line
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image_folder', required=True,
                help='path to input image folder')
args = vars(ap.parse_args())

# Turns a rectangle of shape coords into a bounding box
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right()
    h = rect.bottom()

    return (x, y, w, h)

# Turns a shape from dlib into a numpy array
def shape_to_np(shape, dtype='int'):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

# Gets the point average of two coords when given the shape, s,
	# the indicies to average, and the coord (x or y) for the indicies
	# to average
def point_avg(s, ind, xy):
    avg = []

    assert xy.lower() == 'y' or xy.lower() == 'x', 'Must input x or y'

    if xy.lower() == 'y':
        xy = 1
    else:
        xy = 0

    for p in ind:
        avg.append(s[p][xy])

    # Returns average as truncated integer since this coord will be plotted on
    	# pixel map
    return int(np.mean(avg))

# Function to get points for top of forehead for outlining of face
# Variable names are in reference to order that points are traced on face
	# Therefore, for outer face, stock marks 1-16 will be used then F17-F24
	# to trace the outline of a face
def getForeheadPoints(shape):
	F17 = [shape[14][0], shape[16][1]*2-shape[15][1]]
	F18 = [point_avg(shape, [25, 16], 'x'), shape[26][1]*2-shape[15][1]]
	F19 = [point_avg(shape, [24, 25], 'x'), shape[26][1]*2-point_avg(shape, [15, 14], 'y')]
	F20 = [shape[23][0], shape[23][1]*2-shape[29][1]]
	F21 = [shape[27][0], shape[22][1]*2-shape[30][1]]
	F22 = [shape[20][0], shape[20][1]*2-shape[29][1]]
	F23 = [shape[18][0], point_avg(shape, [17, 0], 
			'y')*2-point_avg(shape, [1, 50], 'y')]
	F24 = [shape[0][0], shape[0][1]*2-shape[1][1]]

	return np.array([F17, F18, F19, F20, F21, F22, F23, F24])

# Function for centering text on an image
def centerText(image, text, fontWeight=1, fontScale=1, color=(0, 0, 255),
				font=cv2.FONT_HERSHEY_SIMPLEX):
	# Gets the text size of the font with input text
	textSize = cv2.getTextSize(text, font, 1, 2)[0]

	# Gets center of image
	textCoord = (int((image.shape[1]-textSize[0])/2), 
				int((image.shape[0]+textSize[1])/2))

	# Puts text on center of image with given inputs
	cv2.putText(image, text, textCoord, font, fontScale, color, fontWeight)

def textTopLeft(image, text, fontWeight=1, fontScale=1, color=(0, 0, 255),
				font=cv2.FONT_HERSHEY_SIMPLEX):
	textSize = cv2.getTextSize(text, font, 1, 2)[0]

	textCoord = (0, 10)
	for oneLine in text.split('\n'):
		cv2.putText(image, oneLine, textCoord, font, fontScale, color, fontWeight)
		textCoord = (0, textCoord[1] + 20)


# Function to crop out portion of image based on input points
	# Then overlays crop out on top of 0-mask
def cropFace(image, croppingPoints):
	x_right = croppingPoints[:, 0].max()
	x_left = croppingPoints[:, 0].min()
	y_bot = croppingPoints[:, 1].max()
	y_top = croppingPoints[:, 1].min()
    
	featureMask = np.zeros((image.shape[0], image.shape[1]))
	outFace = np.zeros_like(image)

	cv2.fillConvexPoly(featureMask, croppingPoints, 1)

	featureMask = featureMask.astype(np.bool)
	outFace = image[y_top:y_bot, x_left:x_right]
	#outFace = outFace[y_top:y_bot, x_left:x_right]
	#cv2.imshow('Cropping Mask', outFace)

	try:
		return cv2.resize(outFace, (224, 224))
	except:
		print('Cannot crop! Skipping file...')
		return 'dontsave'

def getCorrectFace(faces):
	keep = []
	for i, oneFace in enumerate(faces):
		shape = predictor(gray, oneFace)
		shape = shape_to_np(shape)
		if i == 0:
			keep = shape
		elif (abs(shape[16][0]-shape[0][0]) > abs(keep[16][0]-keep[0][0]) and
			abs(shape[25][1]-shape[8][1]) > abs(keep[25][1]-keep[8][1])):
			keep=shape

	return keep

# Reads and resizes image
def handleImage(image_path):
	image = cv2.imread(image_path)
	#image = imutils.resize(image, width=500)
	return image

# For videos... finds best height/width of frame
def best_height_width(imShape):
	botWidthThres = 0.85
	topWidthThres = 1.15
	botHeightThres = 0.75
	topHeightThres = 1.25
	rWidth = abs(imShape[0][0] - imShape[30][0])
	lWidth = abs(imShape[16][0]- imShape[30][0])
	tHeight = abs(imShape[25][1]-imShape[30][1])
	bHeight = abs(imShape[8][1] -imShape[50][1])

	if (rWidth/lWidth > botWidthThres and rWidth/lWidth < topWidthThres) and \
		(tHeight/bHeight > botHeightThres and tHeight/bHeight < topHeightThres):
		return [rWidth/lWidth, tHeight/bHeight], 1
	else:
		return [rWidth/lWidth, tHeight/bHeight], 0

# For videos... finds if image points have moved a lot from frame to frame
def track_wobble(imShape, prevShape):
	wobbleThres = 300
	newWobble = np.sum(abs(imShape - prevShape))
	if newWobble < wobbleThres:
		return newWobble, 1
	else:
		return newWobble, 0 

# For videos... finds if point 50 (on lips) is closer to center than previous best frames
def best_center(imShape, imDims):
	centerThres = 75
	newCenterDist = np.sqrt(np.power((imShape[50][0]-np.divide(imDims[1],2)),2) + 
						np.power((imShape[50][1]-np.divide(imDims[0],2)),2))

	if newCenterDist < centerThres:
		return newCenterDist, 1

	else:
		return newCenterDist, 0

# Sets shape so that best frame is captured as first frame
def setLastShape(shape):
	newShape = shape.copy() * np.inf

	return newShape

# Gets all image and video paths within master input folder from cmd line
def getAllMediaPaths(master_folder):
	print('\n' + 'Collecting image and video paths...')
	allFiles = []
	for root, dirs, files in os.walk(master_folder):
		if not 'cropping' in root.split('/')[-1].lower():
			for _file in files:
				if not '.zip' in _file.lower():
					allFiles.append(os.path.join(root, _file))

	# Gets only .png, .jpg and .jpeg files
	images = [oneFile for oneFile in allFiles if '.png' in oneFile.lower() or 
						'.jpg' in oneFile.lower() or '.jpeg' in oneFile.lower()]
	# Gets only .mov and .mp4
	videos = [oneFile for oneFile in allFiles if '.mov' in oneFile.lower() or
						'.mp4' in oneFile.lower()]

	print('Paths collected!\n')
	return images, videos

# For pretty printing, mostly for videos to show how percent completion
def addReplaceText(toPrint, to_replace=False, replaceWith=' '):
	if to_replace:
		rep = replaceWith*len(to_replace)
		repStat = toPrint.replace(to_replace, rep)
	else:
		repStat = toPrint

	sys.stdout.write('\r' + repStat)
	sys.stdout.flush()
	sys.stdout.write('\r' + toPrint)
	sys.stdout.flush()

# Gets media and choice for user...
def getMedia(image_folder):
	images, videos = getAllMediaPaths(args['image_folder'])
	print(len(images), 'images found and', len(videos), 'videos found!')
	choice = input('Enter...\n\t1. Process Images (%s total)' \
					'\n\t2. Process Videos (%s total)\n\t' \
					'Choice: ' % (len(images), len(videos)))
	if int(choice)==1:
		toIter = images
	elif int(choice)==2:
		toIter = videos
	else:
		raise('Invalid selection')

	return toIter, int(choice)

def getVideoFrame(cap):
	flag = 0
	readCap = True
	fail = 0
	while not flag and fail < 3:
		# Gets frame and flag
		flag, frame = cap.read()

		# If frame sucessfully captured... rotate and get frame position
		if flag:
			frame = imutils.rotate_bound(frame, 90)
			posFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)
		
		else:
			fail += 1
			cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES)-1)
			cv2.waitKey(100)

	if not fail < 3:
		return frame, cap.get(cv2.CAP_PROP_POS_FRAMES), False, cap

	if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
		cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT))
		readCap = False

	toPrint = oneFile.split('/')[-1] + ' --> %.2f' % \
				((cap.get(cv2.CAP_PROP_POS_FRAMES)/cap.get(cv2.CAP_PROP_FRAME_COUNT))*100)
	#print(toPrint)
	addReplaceText(toPrint, str(((cap.get(cv2.CAP_PROP_POS_FRAMES)-1)/cap.get(cv2.CAP_PROP_FRAME_COUNT))*100))

	return frame, posFrame, readCap, cap


#¯\_(ツ)_/¯ don't remember why I made this
def findCenter(coords):
    return int(np.mean([coords.min(), coords.max()]))    
    
# Sets up dlib module for face detection
detector = dlib.get_frontal_face_detector()
shape_file = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(shape_file)

# Gets all media paths in two lists then outputs choices to users
toIter, choice = getMedia(args['image_folder'])

c = 0
startTime = lastTime = datetime.now()
allTime = []
# For-loop, enumerating through chosen media paths
for imgInd, oneFile in enumerate(toIter):
	c += 1
	# Initializes 'best' variables for new file under eval
	print('\n')
	bestFrame = bestCrop = None
	lastShape = skip = 0
	
	# Gets current file path
	filePath = oneFile
	
	# Creates output folder and output file path
	outFolder = 'images/cropped'#os.path.join(args['image_folder'], 'CROPPING_RESULTS')
	failFolder = 'images/failures'#os.path.join(args['image_folder'], 'CROPPING_FAILURES')
	if not os.path.exists(outFolder):
		os.mkdir(outFolder)
	if not os.path.exists(failFolder):
		os.mkdir(failFolder)
        
	sub = '.'.join(oneFile.split('/')[-1].split('_')[:2])
	frameNum = oneFile.split('_')[-1].split('.')[0]
	ext = oneFile.split('.')[-1]
	#outFile = os.path.join(outFolder, 'BESTCROP-' + frameNum + '.1-' + sub + '_cssfacefacts.' + ext)
	outFile = os.path.join(outFolder, oneFile.split('/')[-1])
	print(outFile)
	#print(outFile.split('/')[-1])
    #outFile = os.path.join(outFolder, 'BESTCROP-1.1-' + oneFile.split('/')[-1])

	# Checks if output file already exists... if so, skips rest of iteration
	if os.path.exists(outFile) or outFile.lower().count('bestcrop-') > 1:
		print('\nFILE FOUND - SKIPPING\n')
		lastTime = datetime.now()
		continue

	# Sets up video frame capture if keyframe eval chosen
	if choice == 2:
		# Creates capture object
		cap = cv2.VideoCapture(filePath)

		# If capture object couldn't be opened, pauses and tries again
		while not cap.isOpened():
			print('Waiting for frame...')
			cap = cv2.VideoCapture(filePath)
			cv2.waitKey(100)

		# Gets current position of frame within video
		posFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)

	if cv2.waitKey(10) == 27:
		break
	# Inits while-loop bool to true and inits printing variable
	readCap = True
	toPrint = ''
	frameCollect = 0
	while readCap and frameCollect < 11:
		if cv2.waitKey(10) == 27:
			break

		# Gets image or current video frame and processes it
		if choice == 1:
			frame = handleImage(filePath)
			print(oneFile.split('/')[-1] + ' --> 100%%')
			readCap=False
		elif choice == 2:
			# Gets frame, its position and readCap bool for later
			frame, posFrame, readCap, cap = getVideoFrame(cap)

		else:
			raise('Error')

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		face = detector(gray, 1)
		if len(face) == 0:
			#handleNoFace(frame, choice)
			centerText(frame, 'FACE NOT DETECTED', fontWeight=2)
			if choice == 1:
				addReplaceText('FACE NOT DETECTED\n', toPrint)
			#cv2.imshow('Crop', frame)
			skip=1
		elif len(face) > 0:
			shape = getCorrectFace(face)

			if type(lastShape) is int:
				lastShape = setLastShape(shape)

		if not skip:
			#(x, y, w, h) = face_utils.rect_to_bb(rect)
			#for i, (x1, y1) in enumerate(shape):
			#	cv2.circle(frame, (x1, y1), 2, (0, 0, 255), -1)
			#	cv2.putText(frame, '%s' % i, (x1+2, y1+2), 
			#	    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

			origColor = (0, 0, 0)
			foreheadTrace = getForeheadPoints(shape)
			for oneShape in shape[:17]:
				foreheadTrace = np.append(foreheadTrace, [oneShape], axis=0)

			#points = [17, 18, 19, 20, 21, 22, 23, 24]
			#points.extend(range(17))
			cropFrame = cropFace(frame, foreheadTrace)

			if choice == 2:
				outFile = os.path.join(outFolder, 'BESTCROP-%s.%s-%s' % (int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
									int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), oneFile.split('/')[-1][:-4] + '.jpg'))
				if os.path.exists(outFile):
					frameCollect += 1
					readCap = False
					print('\nVideo alread processed, skipping...')
					continue

				wh, wxhFlag = best_height_width(shape)
				wob, wobbleFlag = track_wobble(shape, lastShape)
				cent, centerFlag = best_center(shape, frame.shape)
				lastShape = shape

				flags = [wxhFlag, wobbleFlag, centerFlag]
				if flags.count(1) > 0:
					#print('\n' + 'Width x Height:', wh,
					#	'\n' + 'Wobble:', wob,
					#	'\n' + 'Center:', cent)
					pass
				#print(flags)
				if flags.count(1) > 1:
					#print('Frame Save!')
					bestFrame = frame
					bestCrop = cropFrame
					
					cv2.imwrite(outFile, cropFrame)
					frameCollect += 1
					cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)*0.05)+posFrame)
					centerText(cropFrame, 'FRAME SAVED', fontWeight=2)
					evalPrint = '[%s] WOBBLE: %s\n[%s] CENTER: %s\n[%s] WIDTH, HEIGHT RATIOS: \n%s\nSAVED FRAMES: %s' % (
									flags[1], wob, flags[2], cent, flags[0], wh, frameCollect)
					textTopLeft(cropFrame, evalPrint, fontWeight=1, fontScale=0.25)
					#cv2.imshow('Crop', cropFrame)

				else:
					centerText(cropFrame, '!NOT SAVED!', fontWeight=2)
					evalPrint = '[%s] WOBBLE: %s\n[%s] CENTER: %s\n[%s] WIDTH, HEIGHT RATIOS: \n%s\nSAVED FRAMES: %s' % (
									flags[1], wob, flags[2], cent, flags[0], wh, frameCollect)
					textTopLeft(cropFrame, evalPrint, fontWeight=1, fontScale=0.25)
					#cv2.imshow('Crop', cropFrame)

			if str(cropFrame) in 'dontsave':
				continue
			elif choice == 1:
				cv2.imwrite(outFile, cropFrame)
				#cv2.imshow('Crop', cropFrame)
				cv2.waitKey(100)
				break

		elif choice == 1:
			cv2.imwrite(os.path.join(failFolder, oneFile.split('/')[-1]), frame)

		skip=0

	allTime.append(datetime.now() - lastTime)
	lastTime = datetime.now()
	subsLeft = len(toIter) - c

	print('Estimated time left: %s' % (np.mean(allTime)*subsLeft))

cv2.waitKey(0)
