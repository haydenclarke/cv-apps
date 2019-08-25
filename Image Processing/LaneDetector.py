"""
To do:
1. Review yellow threshold values
2. Review Hough Lines parameters
3. Multiprocessing for speedup - Hough parameters may also work
"""


import cv2
import numpy as np
from matplotlib import pyplot as plt

class Lane:

    def __init__(self):

        self.image = None
        self.image_roi = None
        self.lines = None
        self.roi = []
        self.low_yellow = np.array([20, 100, 100])
        self.hi_yellow = np.array([30, 255, 255])
        self.white_thresh = 175

    def create_roi(self, roi):

        x0 = roi[0]
        x1 = roi[2]
        y0 = roi[1]
        y1 = roi[3]

        assert x0 < x1
        assert y0 < y1
        assert x1 <= self.image.shape[0]
        assert y1 <= self.image.shape[1]

        x0 = int(x0)
        y0 = int(y0)
        x1 = int(x1)
        y1 = int(y1)

        self.roi = [x0, y0, x1, y1]

        if len(self.image.shape) > 2:
            self.image_roi = self.image[x0:x1, y0:y1, :]
        else:
            self.image_roi = self.image[x0:x1, y0:y1]

    def mask_yellow_white_lines(self, gray, hsv):

        # white and yellow masks
        white_mask = cv2.inRange(gray, self.white_thresh, 255)
        yellow_mask = cv2.inRange(hsv, self.low_yellow, self.hi_yellow)

        # mix masks
        line_mask = cv2.bitwise_or(yellow_mask, white_mask)

        # masked grayscale image
        # cv2.imshow("Mask", cv2.bitwise_and(line_mask, gray))
        return cv2.bitwise_and(line_mask, gray)


    def preprocess(self):

        assert self.image_roi != [] or None

        # Colour conversions
        gray = cv2.cvtColor(self.image_roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(self.image_roi, cv2.COLOR_BGR2HSV)

        # Mask road
        masked_road = self.mask_yellow_white_lines(gray, hsv)

        # Remove noise
        return cv2.GaussianBlur(masked_road, (5, 5), 0)

    def drawLines(self):

        try:
            for [[x1, y1, x2, y2]] in self.lines:
                cv2.line(self.image_roi, (x1, y1), (x2, y2), (0, 0, 255), 2)

        except:
            pass
        # cv2.imshow("Lines", self.image_roi)
        

    def detect_lane(self, image, roi):
        self.image = image
        self.create_roi(roi)
        # cv2.imshow("ROI", self.image_roi)
        preprocessed_image = self.preprocess()
        edges = cv2.Canny(preprocessed_image, 50, 150)
        # cv2.imshow("edges", edges)
        self.lines = cv2.HoughLinesP(edges,1,np.pi/30,60,minLineLength=20, maxLineGap=100)
        self.drawLines()
        # cv2.imshow("Lane Detector", self.image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

"""
Pipeline:
1. Get image
2. Manually identify ROI
3. pass image and ROI to detect_lane()
4. show result
"""

if __name__ == '__main__':

    lane = Lane()

    cap = cv2.VideoCapture("./Data/road_walk.mp4")
    out = cv2.VideoWriter('./Data/Lane detection.mp4', -1, 60, (640, 480))


    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
        
            # Display the resulting frame
            res = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            # image = cv2.imread('./Data/road_image.jpeg')
            roi = [res.shape[0] * 3/5, 0, res.shape[0], res.shape[1]]
            lane.detect_lane(res, roi)
            cv2.imshow('Frame', lane.image)
            out.write(cv2.resize(lane.image, (640, 480)))
        
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        # Break the loop
        else: 
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()



