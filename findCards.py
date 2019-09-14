from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
from PIL import Image
import numpy as np
import Cards

camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640,480))

time.sleep(0.1)

font = cv2.FONT_HERSHEY_SIMPLEX

train_ranks = Cards.load_ranks('/home/pi/roboticsFinalProject/Card_Imgs/')
train_suits = Cards.load_ranks('/home/pi/roboticsFinalProject/Card_Imgs/')

freq = cv2.getTickFrequency()

for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):

	#get first time stamp
	t1 = cv2.getTickCount()

	#get image from stream
	image = frame.array

	#grayscale image
#	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#	blur = cv2.GaussianBlur(gray,(5,5),0)

	#preprocess image (using 'Cards' function)
	pre_proc = Cards.preprocess_image(image)

	#find and sort contours of cards in frame
	cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

	if len(cnts_sort) != 0:
		cards = []
		k = 0 #count

		for i in range(len(cnts_sort)):
			if (cnt_is_card[i] ==1):

				cards.append(Cards.preprocess_card(cnts_sort[i],image))

				#find best rank and suit match
				cards[k].best_rank_match,cards[k].best_suit_match,cards[k].rank_diff,cards[k].suit_diff = Cards.match_card(cards[k],train_ranks,train_suits)

				#draw center point and match result on image
				image = Cards.draw_results(image, cards[k])
				k = k+1

		if (len(cards) != 0):
			temp_cnts = []
			for i in range (len(cards)):
				temp_cnts.append(cards[i].contour)
			cv2.drawContours(image,temp_cnts, -1, (255,0,0), 2)

	#display framerate
	t2 = cv2.getTickCount()
	timeElapsed =(t2 -t1)/freq
	frameRate = 1/timeElapsed
	cv2.putText(image,"FPS: " + str(int(frameRate)),(10,26),font,0.7,(255,0,255),2,cv2.LINE_AA)

	#print image
	cv2.imshow("Card Detector",image)

	#TESTING
#	print(str(len(cnts_sort)))

	#clear buffer
	rawCapture.truncate(0)

	#break loop if q pressed
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		cv2.destroyAllWindows()
		break

