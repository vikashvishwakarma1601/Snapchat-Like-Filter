import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np 



cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
nose_cascade = cv2.CascadeClassifier("Nose.xml")
eye_cascade = cv2.CascadeClassifier("frontalEyes.xml")

choice  = int(input("\n\t\tEnter 1.Moustache  2.Hat  3.Eye Glasses  4.Chilam  5.All_Filter   : "))


def Nose_Filter(frame,Nose,noses):
	

	noses = sorted(noses,key = lambda f:f[2]*f[3],reverse = True)

	for nose in noses[:1]:
		x,y,w,h = nose
		y = y+20
		w = w+50
		x = x-25
		h = h-10
		Nose = cv2.resize(Nose,(w,h))
		for i in range(h):
			for j in range(w):
				for k in range(3):
					if(Nose[i][j][k]<200):
						frame[y+i][x+j][k] = Nose[i][j][k]
						
		return frame




def Hat_Filter(frame,Hat,hats):

	hats = sorted(hats,key = lambda f:f[2]*f[3],reverse= True)

	for hat in hats[:1]:
		x,y,w,h = hat
		Hat = cv2.resize(Hat,(w+10,int(h*0.65)))

		for i in range(int(h*0.65)):
			for j in range(w+10):
				for k in range(3):
					if(Hat[i][j][k]<200):
						frame[y+i-75][x+j-3][k] = Hat[i][j][k]
		return frame






def Eye_Filter(frame,Eye,eyes):

	eyes = sorted(eyes,key = lambda f:f[2]*f[3],reverse= True)

	for eye in eyes[:1]:
		x,y,w,h = eye
		h = h+60
		w = w+60
		Eye = cv2.resize(Eye,(w,h))

		for i in range(h):
			for j in range(w):
				for k in range(3):
					if(Eye[i][j][k]<100):
						frame[y+i-25][x+j-28][k] = Eye[i][j][k]
		return frame





def Chilum(frame,Cigratte,cigrattes):

	cigrattes = sorted(cigrattes,key = lambda f:f[2]*f[3],reverse = True)

	for cigratte in cigrattes[:1]:
		x,y,w,h = cigratte
		y = y+20
		w = w+10
		x = x-25
		h = h
		Cigratte = cv2.resize(Cigratte,(w,h))
		for i in range(h):
			for j in range(w):
				for k in range(3):
					if(Cigratte[i][j][k]<200):
						frame[y+i+50][x+j-40][k] = Cigratte[i][j][k]
						
		return frame



def all_filter(frame,Cigratte,cigrattes,Eye,eyes,Hat,hats,Nose,noses):

	frame = Chilum(frame,Cigratte,cigrattes)

	frame = Hat_Filter(frame,Hat,hats)

	frame = Eye_Filter(frame,Eye,eyes)

	frame = Nose_Filter(frame,Nose,noses)

	return frame






#####################################################################################
while True:
	
	Nose = cv2.imread("./Train/moustache.png")
	Hat = cv2.imread("./Train/cowboy_hat.png")
	Eye = cv2.imread("./Train/thug.jpg")
	Cigratte = cv2.imread("./Train/cigratte.jpg")

	ret,frame = cap.read()

	
	g_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	cv2.imshow("Frame",g_frame)

	if(ret == False):
		continue

	noses = nose_cascade.detectMultiScale(g_frame,1.3,5)
	hats = face_cascade.detectMultiScale(g_frame,1.3,5)
	eyes = eye_cascade.detectMultiScale(g_frame,1.3,5)
	cigrattes = eyes
	if(len(noses)==0 or len(hats)==0 or len(eyes)==0):
		continue


	if(choice == 1):
		frame = Nose_Filter(frame,Nose,noses)
		cv2.imshow("Nose_Filter",frame)
	
	elif(choice==2):
		frame = Hat_Filter(frame,Hat,hats)
		cv2.imshow("Hat_Filter",frame)

	elif(choice==3):
		frame = Eye_Filter(frame,Eye,eyes)
		cv2.imshow("Eye Glasses Filter",frame)

	elif(choice==4):
		frame = Chilum(frame,Cigratte,cigrattes)
		cv2.imshow("Cigrattes Filter",frame)
	elif(choice ==5 ):
		all_filter(frame,Cigratte,cigrattes,Eye,eyes,Hat,hats,Nose,noses)
		cv2.imshow("All Filter",frame)

	key_pressed = cv2.waitKey(1) & 0xFF
	if(key_pressed == ord('q')):
		break

cap.release()
cv2.destroyAllWindows()