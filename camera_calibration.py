import cv2
import os
import numpy as np
import pathlib

import keyboard
import threading

import sys
sys.path.append('/Users/ullrich/ullrich_ws/Socket_Connection/Socket_Connection')
import pepper_connector
from pepper_connector import socket_connection as connect

import mediapipe as mp

import pybullet as p
from qibullet import SimulationManager
import time
import redis


class camera_calibration:
    def __init__(self, images_dir, images_format, square_size, width, height, images_num=20):
        self.images_dir = images_dir
        self.images_format = images_format
        self.square_size = square_size
        self.width = width
        self.height = height
        self.images_num = images_num
        self.get_enough_pictures = False
        self.take_picture = False

        self.connector = connect("10.15.3.171", 12345, 0, 2)

    # Use keyboard to take calibration pictures
    # Whenever the user wants to take the current moment as one of calibration pictures,
    # Press "s"
    def keyboard_control(self):
        n = 0
        while not self.get_enough_pictures:
            input = keyboard.read_hotkey()
            if input == "s" or input == "S":
                n += 1
                self.take_picture = True
                print("Keyboard Signal: {}, Input Content: {}".format(n, input))
                print("[Keyboard Control]: Receive request to take picture!")
            elif input == "q" or input == "Q":
                print("[Keyboard Control]: User forced to quit!")
                break

    def take_calibration_pictures(self, path):
        img_id = 0
        # # use web-camera to get image
        # cam_port = 0
        # cam = cv2.VideoCapture(cam_port)
        keyboard_control_thread = threading.Thread(target=self.keyboard_control, args=())
        keyboard_control_thread.start()

        while True:
            # result, image = cam.read()

            image = self.connector.get_img()

            # if result:
            cv2.imshow("Camera Calibration", image)

            if self.take_picture:
                # cv2.imwrite("cali_img_" + str(img_id) + ".jpg", image)
                cv2.imwrite(os.path.join(path, "cali_img_" + str(img_id) + ".jpg"), image)
                print("[Calibration Picture Taker]: Just stored picture {}".format(img_id))
                print("******************************")
                img_id += 1
                self.take_picture = False

                if img_id >= self.images_num:
                    self.get_enough_pictures = True
                    print("[Calibration Picture Taker]: Taken enough pictures, waiting for user to force quit ... ")
                    cv2.destroyWindow("Camera Calibration")
                    keyboard_control_thread.join()
                    print("[Calibration Picture Taker]: Succeeded to quit!")
                    break

            cv2.waitKey(10)

    def calibrate_chessboard(self):
        '''Calibrate a camera using chessboard images.'''
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
        objp = np.zeros((self.height * self.width, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.width, 0:self.height].T.reshape(-1, 2)

        objp = objp * self.square_size

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = pathlib.Path(self.images_dir).glob(f'*.{self.images_format}')
        # Iterate through all images
        for fname in images:
            img = cv2.imread(str(fname))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (self.width, self.height), None)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        return [ret, mtx, dist, rvecs, tvecs]

    @staticmethod
    def save_coefficients(mtx, dist, path):
        '''Save the camera matrix and the distortion coefficients to given path/file.'''
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        cv_file.write('K', mtx)
        cv_file.write('D', dist)
        # note you *release* you don't close() a FileStorage object
        cv_file.release()

    @staticmethod
    def load_coefficients(path):
        '''Loads camera matrix and distortion coefficients.'''
        # FILE_STORAGE_READ
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

        # note we also have to specify the type to retrieve other wise we only get a
        # FileNode object back instead of a matrix
        camera_matrix = cv_file.getNode('K').mat()
        dist_matrix = cv_file.getNode('D').mat()

        cv_file.release()
        return [camera_matrix, dist_matrix]


if __name__ == '__main__':
    """ Camera Calibration """
    IMAGES_DIR = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/calibration_images/webcam'
    IMAGES_FORMAT = 'jpg'
    SQUARE_SIZE = 2.0
    WIDTH = 9
    HEIGHT = 6

    camera_cali = camera_calibration(IMAGES_DIR, IMAGES_FORMAT, SQUARE_SIZE, WIDTH, HEIGHT)

    # camera_cali.take_calibration_pictures(IMAGES_DIR)
    # print("**** finish taking pictures ****")
    # Calibrate
    print("[Camera Calibration]: Start to calibrate camera ...")
    ret, mtx, dist, rvecs, tvecs = camera_cali.calibrate_chessboard()

    # Save coefficients into a file
    results_saving_path = "calibration_chessboard.yml"
    camera_cali.save_coefficients(mtx, dist, results_saving_path)

    camera_met, dist_met = camera_cali.load_coefficients(results_saving_path)
    print(camera_met)

    fx = camera_met[0][0]
    fy = camera_met[1][1]
    cx = camera_met[0][2]
    cy = camera_met[1][2]

    print("fx: {}, fy: {}".format(fx, fy))
    print("cx: {}, cy: {}".format(cx, cy))
    print("[Camera Callibration]: Finish camera calibration")
    print("*************************************************")

    """ Detect Human Pose """
    print("[Human Pose Detection]: Start to detect human pose")
    # connect to the camera
    cap = cv2.VideoCapture(0)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    hand_detector = mp_hands.Hands(static_image_mode=True,
                                   max_num_hands=1,
                                   model_complexity=0,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

    z_offset = 1.153
    z_offset_laptop = 1.42
    z_offset_nao = 0.19014
    x_offset_laptop = -0.59

    z_c = 0.5 - x_offset_laptop

    # prepare to solve IK
    simulation_manager = SimulationManager()
    client = simulation_manager.launchSimulation(gui=False, auto_step=False)
    pepper = simulation_manager.spawnPepper(client, translation=[0.0, 0.0, 0.0], spawn_ground_plane=False)
    r = redis.Redis(host='localhost', port=6379, db=0)
    time.sleep(3.0)

    # Get Robot Info
    bodyUniqueId = pepper.getRobotModel()
    endEffectorLinkIndex = pepper.link_dict["r_hand"].getIndex()

    try:
        while True:
            # image = camera_cali.connector.get_img()
            success, image = cap.read()
            if not success:
                continue
            image.flags.writeable = False
            image_height, image_width, _ = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # estimate hand pose
            hand_results = hand_detector.process(image)
            if not hand_results.multi_hand_landmarks:
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = cv2.flip(image, 1)
            else:
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width
                    wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height

                    middle_finger_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width
                    middle_finger_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height
                    # hand_estimation = np.array([wrist_x, wrist_y])


                    # Draw the pose annotation on the image.
                    mp_drawing.draw_landmarks(image,
                                              hand_landmarks,
                                              mp_hands.HAND_CONNECTIONS,
                                              mp_drawing_styles.get_default_hand_landmarks_style(),
                                              mp_drawing_styles.get_default_hand_connections_style())

                x_c = (middle_finger_x  - cx) * z_c / fx # wrist x coordinate in 3D camera frame (in meter)
                y_c = (middle_finger_y - cy) * z_c / fy # wrist y coordinate in 3D camera frame (in meter)

                hand_position_c = [x_c, y_c, z_c] # wrist position in camera frame
                # hand_position_w = [0.3, -x_c, -y_c + z_offset_nao]  # wrist position in world frame
                hand_position_w = [0.5, -x_c, -y_c + z_offset] # wrist position in world frame
                # hand_position_w = [z_c, -x_c + y_offset_laptop, -y_c + z_offset_laptop] # wrist position in world frame

                target_hand_position = str(hand_position_w[0]) + " " + str(hand_position_w[1]) + " " + str(hand_position_w[2])

                # solve IK
                ik = p.calculateInverseKinematics(bodyUniqueId, endEffectorLinkIndex, hand_position_w)

                # fetch angle values from ik solution
                knee_pitch = ik[0]
                hip_pitch = ik[1]
                hip_roll = ik[2]
                rshoulder_pitch = ik[25]
                rshoulder_roll = ik[26]
                relbow_yaw = ik[27]
                relbow_roll = ik[28]
                joint_values = str(knee_pitch) + " " + str(hip_pitch) + " " + str(hip_roll) + " " + \
                               str(rshoulder_pitch) + " " + str(rshoulder_roll) + " " + \
                               str(relbow_yaw) + " " + str(relbow_roll)

                r.publish('target_joint_values', joint_values)
                # r.publish('target_hand_position', target_hand_position)
                time.sleep(0.01)

                # draw text label on the image
                position_text = "x: {:0.3f}, y: {:0.3f}, z: {:0.3f}".format(hand_position_c[0],
                                                                            hand_position_c[1],
                                                                            hand_position_c[2])
                image = cv2.flip(image, 1)
                # cv2.putText(image, position_text, (int(middle_finger_x), int(middle_finger_y) - 15),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(image, position_text, (int(image_width - middle_finger_x), int(middle_finger_y) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print("[Human Pose Detection]: " + position_text)

            cv2.imshow("Cam Image", image)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        print("User forced to quit")
        pass



