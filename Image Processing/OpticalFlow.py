import cv2
import numpy as np
from matplotlib import pyplot as plt

lk_params = dict(
    winSize  = (15,15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

cap = cv2.VideoCapture('./Data/road_walk.mp4')

ret, prev_frame = cap.read()
prev_frame = cv2.resize(prev_frame, (640, 480))

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

while (1):

    ret, current_frame = cap.read()
    current_frame = cv2.resize(current_frame, (640, 480))
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    current_points = cv2.goodFeaturesToTrack(current_gray, mask=None, **feature_params)

    current_points, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, prev_points, current_points, None, **lk_params)

    current_features = current_points[st == 1]
    prev_features = prev_points[st == 1]

    if len(current_features) > 0:
        print(len(current_features))
    else:
        print(0)

    for i, (current, prev) in enumerate(zip(current_features, prev_features)):

        pc0, pc1 = current.ravel()
        pp0, pp1 = prev.ravel()

        mask = cv2.line(np.zeros_like(current_frame), (pc0, pc1), (pp0, pp1), (0, 255, 0), 2)

    img = cv2.add(current_frame, mask)
    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    prev_gray = current_gray.copy()
    prev_points = current_features.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()