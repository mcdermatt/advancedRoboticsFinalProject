import cv2
import numpy as np
import VideoStream
import Card
import time
import os

IM_WIDTH = 1280
IM_HEIGHT = 720
FRAME_RATE = 10

freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

#init camera object using multithreaded VideoStream class
videostream = VideoStream.VideoStream((IM_WIDTH,IM_HEIGHT),FRAME_RATE).start()
time.sleep(1)

#define path to cards
train_ranks = Card.load_ranks('/home/pi/roboticsFinalProject/Card_Imgs/')
train_suits = Card.load_suits('/home/pi/roboticsFinalProject/Card_Imgs/')

#------------------Main----------
cam_quit = 0

while cam_quit == 0:
	#get frame
	image = videostream.read()

	#start timer
	t1 = cv2.getTickCount()

	#pre-process image
	pre_proc = Card.preprocess_image(image)

	#find and sort the contours in the image
	cnts_sort, cnt_is_card = Card.find_cards(pre_proc)

	if len(cnts_sort) != 0:

		#inits cards list
		cards = []
		k = 0

		for i in range(len(cnts_sort)):
			if (cnt_is_card[i] == 1):

				#create card object and add to list of cards
				#preprocess_card gets corner points and transforms projection of card
				cards.append(Card.preprocess_card(cnts_sort[i],image))

				#get rank and suit
				cards[k].best_rank_match,cards[k].best_suit_match,cards[k].rank_diff,cards[k].suit_diff = Card.match_card(cards[k],train_ranks,train_suits)

				#draw center point
				image = Card.draw_results(image, cards[k])
				k = k + 1

			#draw contours on image
			if (len(cards) != 0):
				temp_cnts = []
				for i in range(len(cards)):
					temp_cnts.append(cards[i].contour)
				cv2.drawContours(image,temp_cnts, -1, (0,255,0),2)


	#calc and and put fps on frame
	t2 = cv2.getTickCount()
	dt = t2 - t1
	fps = 1/(dt / freq)
	cv2.putText(image,"FPS: " + str(int(fps)),(10,30),font,1,(0,0,0),3,cv2.LINE_AA)
	cv2.putText(image,"FPS: " + str(int(fps)),(10,30),font,1,(255,255,255),2,cv2.LINE_AA)

	#display frame
	cv2.imshow("findCardsV2", image)

	#stupid openCV keyboard interrupt (if you remove this imshow wont work)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		cam_quit = 1

cv2.destroyAllWindows()
videostream.stop()
