import numpy as np
import gym
from gym import spaces
import cv2
import mediapipe as mp
import sys

from pupil_apriltags import Detector

sys.path.append('/Users/ullrich/ullrich_ws/Socket_Connection/Socket_Connection')
import pepper_connector
from pepper_connector import socket_connection as connect
from camera_calibration import camera_calibration


class GreetingEnv(gym.Env):
    def __init__(self):
        super(GreetingEnv, self).__init__()
        '''
        # parameters for camera calibration
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # Load camera calibration parameters
        parameters_dir = "calibration_chessboard.yml"
        self.calibrate_camera(parameters_dir)

        self.tag_size = 0.12
        self.detector = Detector(families='tag36h11',
                                 nthreads=1,
                                 quad_decimate=1.0,
                                 quad_sigma=0.0,
                                 refine_edges=1,
                                 decode_sharpening=0.25,
                                 debug=0)

        # connector used to get image from robot camera
        self.connector = connect("192.168.0.196", 12345, 0, 2)
        '''

        self.cam = cv2.VideoCapture(0)

        # parameters related to Mediapipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.human_pose_detector = self.mp_pose.Pose(min_detection_confidence=0.5,
                                                          min_tracking_confidence=0.5)

        self.current_image = None

    ''' environment version using Mediapipe '''
    def reset(self):
        result, image = self.cam.read()
        state = None

        if not result:
            print("[Camera]: Fail to get image")
            return state

        state, self.current_image = self.estimate_human_pose(image)

        return state

    def step(self, action):
        result, image = self.cam.read()
        state = None

        if not result:
            print("[Camera]: Fail to get image")
        else:
            state, self.current_image = self.estimate_human_pose(image)

        # some dummy parameters
        done = False
        reward = 0
        info = {}

        return state, reward, done, info

    def render(self, mode='console'):
        cv2.imshow("Cam Image", self.current_image)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyWindow("Cam Image")

    ''' environment version using april-tag '''
    # used to get the initial state
    def reset_tag(self):
        image = self.connector.get_img()
        state, self.current_image = self.estimate_tag_pose(image)  # state is in the form of np array: [x, y, z]

        return state

    def step_tag(self, action):
        image = self.connector.get_img()
        state, self.current_image = self.estimate_tag_pose(image)  # state is in the form of np array: [x, y, z]

        # some dummy parameters
        done = False
        reward = 0
        info = {}

        return state, reward, done, info

    def render_tag(self, mode='console'):
        cv2.imshow("Cam Image", self.current_image)
        cv2.waitKey(1)

    def close_tag(self):
        cv2.destroyWindow("Cam Image")

    ''' Utility functions '''
    def calibrate_camera(self, parameters_dir):
        camera_met, dist_met = camera_calibration.load_coefficients(parameters_dir)
        print(camera_met)

        self.fx = camera_met[0][0]
        self.fy = camera_met[1][1]
        self.cx = camera_met[0][2]
        self.cy = camera_met[1][2]
        print("fx: {}, fy: {}".format(self.fx, self.fy))
        print("cx: {}, cy: {}".format(self.cx, self.cy))

    def estimate_tag_pose(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = self.detector.detect(gray, estimate_tag_pose=True, camera_params=[self.fx, self.fy, self.cx, self.cy], tag_size=self.tag_size)
        pose_estimation = None
        reshaped_pose = None

        # loop over the AprilTag detection results
        for r in results:
            # get human demo data (i.e., human hand pose)
            pose_estimation = r.pose_t  # in the form of 2d np.array: [[x], [y], [z]]

            # reshape into the form of np array: [x, y, z]
            reshaped_pose = np.array([pose_estimation[0][0], pose_estimation[1][0], pose_estimation[2][0]])

            # add lines and texts on the original image
            image = self.image_visual_processing(image, r, pose_estimation)

        state = reshaped_pose

        return state, image

    def estimate_human_pose(self, image):
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.human_pose_detector.process(image)

        pose_estimation = None

        # return None and flip the camera image when no human is detected
        if not results.pose_landmarks:
            image = cv2.flip(image, 1)
            return pose_estimation, image

        right_wrist_x = results.pose_world_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].x
        right_wrist_y = results.pose_world_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].y
        right_wrist_z = results.pose_world_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].z

        right_elbow_x = results.pose_world_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x
        right_elbow_y = results.pose_world_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y
        right_elbow_z = results.pose_world_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].z

        pose_estimation = np.array([right_wrist_x, right_wrist_y, right_wrist_z,
                                    right_elbow_x, right_elbow_y, right_elbow_z])

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

        image = cv2.flip(image, 1)

        return pose_estimation, image

    @staticmethod
    def image_visual_processing(image, detection_result, pose_estimation):
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = detection_result.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))

        # draw the bounding box of the AprilTag detection
        cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2)

        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(detection_result.center[0]), int(detection_result.center[1]))
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

        # add pose info on the image
        pose_text = "x: {:0.3f}, y: {:0.3f}, z: {:0.3f}".format(pose_estimation[0][0],
                                                                pose_estimation[1][0],
                                                                pose_estimation[2][0])
        cv2.putText(image, pose_text, (ptA[0], ptA[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image




