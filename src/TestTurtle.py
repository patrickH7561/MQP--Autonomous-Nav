import cv2
import apriltag
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import sys

sys.path.insert(0, '/home/patrick/catkin_ws/src/RBE3002_B22_Team12/Lab2/src')
from lab2 import *


class RealTurtle:
    def __init__(self):
        """
        Class constructor
        """
        rospy.init_node('mqp')
        self.raw = rospy.Subscriber('/camera/color/image_raw', Image, self.ProcessImage)
        self.L2 = Lab2()
        self.Tag = []

    def ViewRawImage(self, Image):
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(Image, desired_encoding='passthrough')
        cv2.imshow("Realsense Stream", cv_image)
        cv2.waitKey(1)

    def ProcessImage(self, Image):
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(Image, desired_encoding='passthrough')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        options = apriltag.DetectorOptions(families="tag36h11")
        detector = apriltag.Detector(options)
        results = detector.detect(gray)

        self.Tag = []

        for r in results:
            (ptA, ptB, ptC, ptD) = r.corners
            ptA = (int(ptA[0]), int(ptA[1]))
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))

            #cv2.line(image, ptA, ptB, (0, 255, 0), 2)
            #cv2.line(image, ptB, ptC, (0, 255, 0), 2)
            #cv2.line(image, ptC, ptD, (0, 255, 0), 2)
            #cv2.line(image, ptD, ptA, (0, 255, 0), 2)

            center = (int(r.center[0]), int(r.center[1]))
            #cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

            #tagfamily = r.tag_family.decode("utf-8")
            #cv2.putText(image, tagfamily, (ptA[0], ptA[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            self.Tag = [ptA, ptB, ptC, ptD, center]

        #cv2.imshow("IMAGE", image)
        cv2.waitKey(1)


    def Localization(self):
        while len(self.Tag) == 0:
            self.L2.send_speed(0, 0.1)
        while True:
            print(self.Tag[4])
            self.L2.send_speed(0, 0.1)
            if 645 > self.Tag[4][0] > 635:
                break
        self.L2.send_speed(0, 0)
        exit(0)


    def run(self):
        print("Launched")
        self.Localization()
        rospy.spin()