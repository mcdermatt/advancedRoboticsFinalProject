#classes and functions to be used in the findCardsV2.py progam

import numpy as np
import cv2
import time

#----------constants-----------
#threshold levels
BKG_THRESH = 60
CARD_THRESH = 30
#width and height of box containing card rank and suit
CORNER_WIDTH = 32
CORNER_HEIGHT = 84
#dimensions of rank and suit train images
RANK_WIDTH = 70
RANK_HEIGHT = 125
SUIT_WIDTH = 70
SUIT_HEIGHT =100
#Max suit and rank diff
RANK_DIFF_MAX = 2000
SUIT_DIFF_MAX = 2000
#Size of card
CARD_MAX_AREA = 120000
CARD_MIN_AREA = 5000

font = cv2.FONT_HERSHEY_SIMPLEX


#----Structures to hold query card information--
class Query_card:
	"""contains information on cards detected in image"""
	def __init__(self):
		self.contour = []
		self.width = 0
		self.height = 0
		self.corner_pts = []
		self.center = []
		self.warp = []	#200x300 gray blurred flattened image
		self.rank_img = [] #img of cards rank
		self.suit_img = [] #img of cards suit
		self.best_rank_match = " "
		self.best_suit_match = " "
		self.rank_diff = 0 #diff between rank img and best train img
		self.suit_diff = 0 #same but for suit

class Train_ranks:
	"""stores information on rank train images"""

	def __init__(self):
		self.img = [] #preprocessed rank image
		self.name = "Placeholder"

class Train_suits:
	"""stores information on suit train images"""
	def __init__(self):
		self.img = []
		self.name = "Placeholder"



#-----Functions------
def load_ranks(filepath):
	train_ranks = []
	i = 0

	for Rank in ['Ace','Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten','Jack','Queen','King']:
		train_ranks.append(Train_ranks())
		train_ranks[i].name = Rank
		filename = Rank + '.jpg'
		train_ranks[i].img = cv2.imread(filepath+filename,cv2.IMREAD_GRAYSCALE)
		i = i + 1
	return train_ranks

def load_suits(filepath):
	train_suits = []
	i = 0

	for Suit in ['Spades','Diamonds','Clubs','Hearts']:
		train_suits.append(Train_suits())
		train_suits[i].name = Suit
		filename = Suit + '.jpg'
		train_suits[i].img = cv2.imread(filepath+filename,cv2.IMREAD_GRAYSCALE)
		i = i + 1
	return train_suits

def preprocess_image(image):
	"""returns grayed blurred and transformed image"""
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),0)

	#get background color of frame to determine proper values for thresholding 
	img_w, img_h = np.shape(image)[:2]
	bkg_level = gray[int(img_h/100)][int(img_w/2)]
	thresh_level = bkg_level + BKG_THRESH

	retval, thresh = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)
	return thresh


def find_cards(thresh_image):
	"""finds card sized contours in frame returns number of cards and
		list of card contours from largest to smallest """

	#find contours within frame and sort by size
	dummy, cnts, hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	index_sort = sorted(range(len(cnts)), key = lambda i : cv2.contourArea(cnts[i]),reverse=True)

	if len(cnts) == 0:
		return [],[]

	cnts_sort = []
	hier_sort = []
	cnt_is_card = np.zeros(len(cnts),dtype=int)

	#fill lists with sorted contour and heigherarchy
	#heierarchy array can be used to see if contours have parents or not
	for i in index_sort:
		cnts_sort.append(cnts[i])
		hier_sort.append(hier[0][i])

	#determine which of contours are cards
	#1-smaller area than max card area
	#2-larger area than min
	#3-have no parents lol
	#4-have four corners
	for i in range(len(cnts_sort)):
		size = cv2.contourArea(cnts_sort[i])
		peri = cv2.arcLength(cnts_sort[i],True)
		approx = cv2.approxPolyDP(cnts_sort[i],0.01*peri,True)

		if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA) and
			(hier_sort[i][3] == -1) and (len(approx) == 4)):
			cnt_is_card[i] = 1

	return cnts_sort, cnt_is_card


def preprocess_card(contour, image):
	"""uses contour to find information about query card, isolates rank
		and suit images from card"""
	#init new Query_card object
	qCard = Query_card()

	qCard.contour = contour

	#find perimeter of card and use to approximate corner pts
	peri = cv2.arcLength(contour,True)
	approx = cv2.approxPolyDP(contour,0.01*peri,True)
	pts = np.float32(approx)
	qCard.corner_pts = pts

	#find w and h of cards bounding rectangle
	x,y,w,h = cv2.boundingRect(contour)
	qCard.width, qCard.height = w,h

	#get center point of card
	average = np.sum(pts, axis=0)/len(pts)
	cent_x = int(average[0][0])
	cent_y = int(average[0][1])
	qCard.center = [cent_x,cent_y]

	#transform card to 200x300 image
	qCard.warp = flattener(image, pts, w, h)

	#zoom in on corner of transformed image
	Qcorner = qCard.warp[0:CORNER_HEIGHT,0:CORNER_WIDTH]
	Qcorner_zoom = cv2.resize(Qcorner, (0,0), fx=4, fy=4)

	#determine ambient brightness level to determine proper thresh val
	white_level = Qcorner_zoom[15,int((CORNER_WIDTH*4)/2)]
	thresh_level = white_level - CARD_THRESH
	if (thresh_level <= 0):
		thresh_level = 1
	retval, query_thresh = cv2.threshold(Qcorner_zoom, thresh_level, 255, cv2.THRESH_BINARY_INV)

	#split corner into rank and suit parts
	Qrank = query_thresh[20:185,0:128]
	Qsuit = query_thresh[186:336,0:128]
#	Qrank = query_thresh[20:185,0:128]
#	Qsuit = query_thresh[186:300,0:128]

	#find rank contour and bounding box
	dummy, Qrank_cnts, heir = cv2.findContours(Qrank, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#find largest contour
	Qrank_cnts = sorted(Qrank_cnts, key=cv2.contourArea,reverse=True)

	#use bounding box of largest contour to resize corner to match train image size
	if len(Qrank_cnts) != 0:
		x1, y1, w1, h1 = cv2.boundingRect(Qrank_cnts[0])
		Qrank_roi = Qrank[y1:y1+h1,x1:x1+w1]
		Qrank_sized = cv2.resize(Qrank_roi, (RANK_WIDTH,RANK_HEIGHT), 0,0)
		qCard.rank_img = Qrank_sized

	#find suit contour and bounding box
	dummy, Qsuit_cnts, heir = cv2.findContours(Qsuit, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#isolate largest contour
	Qsuit_cnts = sorted(Qsuit_cnts, key=cv2.contourArea, reverse=True)

	#use bounding box of largest contour to rezise lower corner to match train img size
	if len(Qsuit_cnts) != 0:
		x2,y2,w2,h2 = cv2.boundingRect(Qsuit_cnts[0])
		Qsuit_roi = Qsuit[y2:y2+h2,x2:x2+w2]
		Qsuit_sized = cv2.resize(Qsuit_roi, (SUIT_WIDTH, SUIT_HEIGHT),0,0)
		qCard.suit_img = Qsuit_sized

	return qCard

def match_card(qCard, train_ranks, train_suits):
	"""finds best rank and suit match for query card"""
	best_rank_match_diff = 10000
	best_suit_match_diff = 10000
	best_rank_match_name = ' '
	best_suit_match_name = ' '
	i = 0

	if (len(qCard.rank_img) !=0) and (len(qCard.suit_img) != 0):
		for Trank in train_ranks:
			diff_img = cv2.absdiff(qCard.rank_img, Trank.img)
			rank_diff = int(np.sum(diff_img)/255)

			if rank_diff < best_rank_match_diff:
				best_rank_diff_img = diff_img
				best_rank_match_diff = rank_diff
				best_rank_name = Trank.name

		for Tsuit in train_suits:
			diff_img = cv2.absdiff(qCard.suit_img, Tsuit.img)
			suit_diff = int(np.sum(diff_img)/255)

			if suit_diff < best_suit_match_diff:
				best_suit_diff_img = diff_img
				best_suit_match_img = suit_diff
				best_suit_name = Tsuit.name

	#combine suit and rank, only proceed if diff is less than max diff constant
	if (best_rank_match_diff < RANK_DIFF_MAX):
		best_rank_match_name = best_rank_name

	if (best_suit_match_diff < SUIT_DIFF_MAX):
		best_suit_match_name = best_suit_name

	return best_rank_match_name, best_suit_match_name, best_rank_match_diff, best_suit_match_diff

def draw_results(image, qCard):
	"""Draw card name, center point, and contour on frame"""

	x = qCard.center[0]
	y = qCard.center[1]
	cv2.circle(image,(x,y),2,(0,0,0),-1,cv2.LINE_AA)

	rank_name = qCard.best_rank_match
	suit_name = qCard.best_suit_match

	#test
#	print(suit_name)

	#draw card name twice (so letters have a dank black outline)
	cv2.putText(image,(rank_name+' '),(x-60,y-10),font,.5,(0,0,0),2,cv2.LINE_AA)
	cv2.putText(image,(rank_name+' '),(x-60,y-10),font,.5,(255,255,255),1,cv2.LINE_AA)
	cv2.putText(image,suit_name,(x-60,y+25),font,.5,(0,0,0),2,cv2.LINE_AA)
	cv2.putText(image,suit_name,(x-60,y+25),font,.5,(255,255,255),1,cv2.LINE_AA)

	return image

def flattener(image, pts, w, h):
	"""returns 200x300 flattened image of card"""

	temp_rect = np.zeros((4,2), dtype="float32")
	s = np.sum(pts, axis=2)
	tl = pts[np.argmin(s)]
	br = pts[np.argmax(s)]

	diff = np.diff(pts, axis=-1)
	tr = pts[np.argmin(diff)]
	bl = pts[np.argmax(diff)]

	#create array [top left, top right, bottom right, bottom left] before transform

	#vertically oriented card
	if w <= 0.8*h:
		temp_rect[0] = tl
		temp_rect[1] = tr
		temp_rect[2] = br
		temp_rect[3] = bl

	#horizontally oriented card
	if w >= 1.2*h:
		temp_rect[0] = bl
		temp_rect[1] = tl
		temp_rect[2] = tr
		temp_rect[3] = br

	#card is neither horizontal or vertical
	if w > 0.8*h and w < 1.2*h:
		#tilted left
		if pts[1][0][1] <= pts[3][0][1]:
			temp_rect[0] = pts[1][0] #tl
			temp_rect[1] = pts[0][0] #tr
			temp_rect[2] = pts[3][0] #br
			temp_rect[3] = pts[2][0] #bl
		#tilted right
		if pts[1][0][1] > pts[3][0][1]:
			temp_rect[0] = pts[0][0] #tl
			temp_rect[1] = pts[3][0] #tr
			temp_rect[2] = pts[2][0] #br
			temp_rect[3] = pts[1][0] #bl

	maxWidth = 200
	maxHeight = 300

	#destination array
	dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], np.float32)
	#calc perspective transform matrix
	M = cv2.getPerspectiveTransform(temp_rect,dst)
	#warp card image
	warp = cv2.warpPerspective(image, M, (maxWidth,maxHeight))
	warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

	return warp
