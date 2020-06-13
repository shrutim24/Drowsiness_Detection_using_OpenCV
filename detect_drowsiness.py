import numpy as np
import playsound
import argparse
import imutils
import cv2
import time
import dlib
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])

	eyeratio = (A + B) / (2.0 * C)
	return eyeratio

def sound_alarm(path):
	playsound.playsound(path)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="shape_predictor_68_face_landmarks.dat")
ap.add_argument("-a", "--alarm", type=str, default="",
	help="alarm.wav")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())
 
COUNTER = 0
ALARM_ON = False
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 45

print("Loading...")
dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor(args["shape_predictor"])

(lefts, lefte) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rights, righte) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:
	
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	r = dlib_detector(gray, 0)
	for rect in r:
		shape = dlib_predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		leftEye = shape[lefts:lefte]
		rightEye = shape[rights:righte]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		earatio = (leftEAR + rightEAR) / 2.0

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		if earatio < EYE_AR_THRESH:
			COUNTER += 1
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				if not ALARM_ON:
					ALARM_ON = True
					if args["alarm"] != "":
						t = Thread(target=sound_alarm,
							args=(args["alarm"],))
						t.deamon = True
						t.start()
				cv2.putText(frame, "ALERT! ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else:
			COUNTER = 0
			ALARM_ON = False
		cv2.putText(frame, "RATIO: {:.2f}".format(earatio), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	cv2.imshow("Screen", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break
cv2.destroyAllWindows()
vs.stop()