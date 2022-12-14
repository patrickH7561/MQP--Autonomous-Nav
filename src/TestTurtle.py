#!/usr/bin/env python3
import cv2
import apriltag
import rospy
from PIL import Image as TestImage
from matplotlib import cm
from rospy.numpy_msg import numpy_msg
import pyrealsense2 as rs
import numpy as np
import cv2
from apriltag_ros.msg import AprilTagDetectionArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import sys
import math

sys.path.insert(0, '/home/patrick/catkin_ws/src/RBE3002_B22_Team12/Lab2/src')
from lab2 import *


class RealTurtle:
    def __init__(self):
        """
        Class constructor
        """
        rospy.init_node('mqp')
        self.raw = rospy.Subscriber('/camera/color/image_raw', AprilTagDetectionArray,  self.TagDetect)
        self.L2 = Lab2()
        self.Tags = []
        '''self.FocalLength = 1.93
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        if device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(self.config)

        self.test()'''

    def test(self):
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                                 interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)

    def TagDetect(self, msg):
        print("Here")
        w = msg.pose.pose.position.x
        print(w)

    def ViewRawImage(self, Image):
        print("Running")
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(Image, desired_encoding='passthrough')
        cv2.imshow("Realsense Stream", cv_image)
        cv2.waitKey(10)

    def ProcessImage(self, Image):
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(Image, desired_encoding='passthrough')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        options = apriltag.DetectorOptions(families="tag36h11")
        detector = apriltag.Detector(options)
        results = detector.detect(gray)

        self.Tags = []

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

    def CalibrateImage(self, path):
        # convert the image to grayscale, blur it, and detect edges
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 35, 125)
        options = apriltag.DetectorOptions(families="tag36h11")
        detector = apriltag.Detector(options)
        results = detector.detect(gray)
        self.Tags = []
        for r in results:
            (ptA, ptB, ptC, ptD) = r.corners
            ptA = (int(ptA[0]), int(ptA[1]))
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
        self.Tags.append([ptA, ptB, ptC, ptD])

        # find the contours in the edged image and keep the largest one;
        # we'll assume that this is our piece of paper in the image
        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)

        # compute the bounding box of the of the paper region and return it
        return cv2.minAreaRect(c)

    def Triangulate(self, InputIMG):
        # initialize the known distance from the camera to the object, which
        # in this case is 24 inches
        KNOWN_DISTANCE = 61
        # initialize the known object width, which in this case, the piece of
        # paper is 11 inches wide
        KNOWN_WIDTH = 17
        # load the first image that contains an object that is KNOWN TO BE 2 feet
        # from our camera, then find the paper marker in the image, and initialize
        im = cv2.imread("/home/patrick/catkin_ws/src/MQP--Autonomous-Nav/src/Calibration_IMG.png")
        marker = self.CalibrateImage("/home/patrick/catkin_ws/src/MQP--Autonomous-Nav/src/Calibration_IMG.png")


        apriltag = (KNOWN_WIDTH**2)/abs(int(self.Tags[0][0][0]-self.Tags[0][1][0])**2) # CMs/Pixel @ 2m
        self.ProcessImage(InputIMG)

        distance_tag_1 = ((self.Tags[0][0][0] - self.Tags[0][1][0])**2) / apriltag
        distance_tag_2 = 2*(KNOWN_WIDTH * focalLength) / abs(self.Tags[1][0][0]-self.Tags[1][1][0])

        print(distance_tag_1, distance_tag_2)

        # Returning world coordinate of current position
        tag_1 = (7, 0)  # world coordinate of tag 1
        tag_2 = (12, 0)  # world coordinate of tag 2
        d = math.sqrt((tag_2[0] - tag_1[0]) ** 2 + (tag_2[1] - tag_1[1]) ** 2)  # distance between april tags in meters
        a = (distance_tag_1 ** 2 - distance_tag_2 ** 2 + d ** 2) / (2 * d)
        h = math.sqrt(abs(distance_tag_1 ** 2 - a ** 2))
        x2 = tag_1[0] + a * (tag_2[0] - tag_1[0]) / d
        y2 = tag_1[1] + a * (tag_2[1] - tag_1[1]) / d
        x3 = x2 + h * (tag_2[1] - tag_1[1]) / d  ## if the point is on the wrong side of the field make this x2-h...
        y3 = y2 - h * (tag_2[0] - tag_1[0]) / d  ## if the point is on the wrong side of the field make this y2+h...
        print(x3, y3)
        return (x3, y3)

    def Localization(self):
        while True:
            rospy.sleep(.01)
            self.L2.send_speed(0, 0.1)
            rospy.sleep(.01)
            if len(self.Tags) == 2:
                self.Triangulate(cv2.imread("Calibration_IMG.png"))
                break
        exit(0)


    def run(self):
        rospy.spin()


if __name__ == '__main__':
    RealTurtle().run()
