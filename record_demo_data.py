import threading
from time import sleep
import redis

import cv2
import numpy as np
import mediapipe as mp
import keyboard
from pupil_apriltags import Detector
import argparse
import os

from camera_calibration import camera_calibration

import sys
sys.path.append('/Users/ullrich/ullrich_ws/Socket_Connection/Socket_Connection')
import pepper_connector
from pepper_connector import socket_connection as connect

sys.path.append('/Users/ullrich/ullrich_ws/connectors/python')
import social_interaction_cloud
from social_interaction_cloud.action import ActionRunner
from social_interaction_cloud.basic_connector import BasicSICConnector

from simplejson import loads
from agent import Agent


class demo_data_recorder:
    def __init__(self, detector, mp_pose, mp_drawing, mp_drawing_styles, server_ip='127.0.0.1', subject_id=1):
        # parameters for tag detection
        self.detector = detector
        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles
        self.mp_pose = mp_pose
        # parameters for camera calibration
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # parameters for demo recording
        # self.human_demo_data = [np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])]
        self.human_demo_data = []
        self.robot_demo_data = []
        self.human_hand_position_data = []
        self.robot_hand_position_data = []
        self.robot_hand_ori_data = []
        # self.human_demo_path = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/demo_data/side_view/multi_pos/expert_observations.csv'
        # self.human_demo_path = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/demo_data/pepper/expert_observations.csv'

        self.subject_id = subject_id
        self.human_demo_path = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/demo_data/pepper/' + str(subject_id) + '/expert_observations.csv'
        # self.robot_demo_path = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/demo_data/side_view/multi_pos/expert_actions.csv'
        # self.robot_demo_path = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/demo_data/pepper/expert_actions.csv'
        self.robot_demo_path = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/demo_data/pepper/' + str(subject_id) + '/expert_actions.csv'
        self.human_hand_position_data_path = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/demo_data/pepper/' + str(subject_id) + '/human_hand_position_trajectories.csv'
        self.robot_hand_position_data_path = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/demo_data/pepper/' + str(subject_id) + '/robot_hand_position_trajectories.csv'
        self.robot_hand_ori_data_path = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/demo_data/pepper/' + str(subject_id) + '/robot_hand_ori_trajectories.csv'
        self.whether_started = False
        self.whether_paused = False
        self.whether_start_robot_demo = False
        self.episode_control_thread = threading.Thread(target=self.keyboard_control, args=())
        self.robot_demo_thread = threading.Thread(target=self.record_robot_motion, args=())

        # self.sic = BasicSICConnector(server_ip)
        # self.action_runner = ActionRunner(self.sic)
        self.motion = None
        self.decompressed_motion = None
        self.joints = ['RArm']

        # Redis-based client to send command to and retrieve data from the robot side
        # self.r = redis.Redis(host='localhost', port=6379, db=0)
        # self.sub_robot_data = self.r.pubsub()
        self.whether_received_robot_data = False
        self.latest_robot_data = None
        # self.start_robot_data_listener()

    def reset_data_saving_path(self, new_data_dir, factor_id=None):
        if factor_id is None:
            self.human_demo_path = new_data_dir + 'subject_' + str(self.subject_id) + '/expert_observations.csv'
            self.robot_demo_path = new_data_dir + 'subject_' + str(self.subject_id) + '/expert_actions.csv'
            self.human_hand_position_data_path = new_data_dir + 'subject_' + str(self.subject_id) + '/human_hand_position_trajectories.csv'
            self.robot_hand_position_data_path = new_data_dir + 'subject_' + str(self.subject_id) + '/robot_hand_position_trajectories.csv'
            self.robot_hand_ori_data_path = new_data_dir + 'subject_' + str(self.subject_id) + '/robot_hand_ori_trajectories.csv'

            new_path = new_data_dir + 'subject_' + str(self.subject_id)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
                print("[Demo Data Recorder]: Created a new saving path for subject {}".format(self.subject_id))
        else:
            self.human_demo_path = new_data_dir + 'subject_' + str(self.subject_id) + '/factor_' + str(factor_id) + '/expert_observations.csv'
            self.robot_demo_path = new_data_dir + 'subject_' + str(self.subject_id) + '/factor_' + str(factor_id) + '/expert_actions.csv'
            self.human_hand_position_data_path = new_data_dir + 'subject_' + str(self.subject_id) + '/factor_' + str(factor_id)+ '/human_hand_position_trajectories.csv'
            self.robot_hand_position_data_path = new_data_dir + 'subject_' + str(self.subject_id) + '/factor_' + str(factor_id)+ '/robot_hand_position_trajectories.csv'
            self.robot_hand_ori_data_path = new_data_dir + 'subject_' + str(self.subject_id) + '/factor_' + str(factor_id)+ '/robot_hand_ori_trajectories.csv'

            new_path = new_data_dir + 'subject_' + str(self.subject_id) + '/factor_' + str(factor_id)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
                print("[Demo Data Recorder]: Created a new saving path for subject {} factor {}".format(self.subject_id, factor_id))

    def calibrate_camera(self, parameters_dir):
        camera_met, dist_met = camera_calibration.load_coefficients(parameters_dir)
        print(camera_met)

        self.fx = camera_met[0][0]
        self.fy = camera_met[1][1]
        self.cx = camera_met[0][2]
        self.cy = camera_met[1][2]
        print("fx: {}, fy: {}".format(self.fx, self.fy))
        print("cx: {}, cy: {}".format(self.cx, self.cy))

    def estimate_pose(self, image):
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(image)

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

        # print("wrist_x: {:0.3f}, wrist_y: {:0.3f}, wrist_z: {:0.3f}".
        #       format(pose_estimation[0],
        #              pose_estimation[1],
        #              pose_estimation[2]))

        # print("elbow_x: {:0.3f}, elbow_y: {:0.3f}, elbow_z: {:0.3f}".
        #       format(pose_estimation[3],
        #              pose_estimation[4],

        #              pose_estimation[5]))

        # print(" ##################################### ")

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

        image = cv2.flip(image, 1)

        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # results = self.detector.detect(gray, estimate_tag_pose=True, camera_params=[self.fx, self.fy, self.cx, self.cy], tag_size=self.tag_size)
        # pose_estimation = None

        # # loop over the AprilTag detection results
        # for r in results:
        #     # get human demo data (i.e., human hand pose)
        #     pose_estimation = r.pose_t  # in the form of 2d np.array: [[x], [y], [z]]
        #     # human_demo_data.append(np.array([pose_estimation[0][0], pose_estimation[1][0], pose_estimation[2][0]]))
        #     print(pose_estimation)
        #
        #     # add lines and texts on the original image
        #     image = self.image_visual_processing(image, r, pose_estimation)

        return pose_estimation, image

    def start(self):
        self.episode_control_thread.start()

    def collect_human_demo_data(self, human_pose):
        # reshaped_pose = np.array([human_pose[0][0], human_pose[1][0], human_pose[2][0]])
        reshaped_pose = human_pose

        self.human_demo_data.append(reshaped_pose)

    def start_collect_robot_demo_data(self):
        self.robot_demo_thread.start()

    def record_robot_motion(self):
        self.sic.start()
        self.action_runner.run_waiting_action('set_stiffness', self.joints, 0)
        self.action_runner.run_waiting_action('start_record_motion', self.joints, 100)

    def stop_record_robot_motion(self):
        self.action_runner.run_waiting_action('stop_record_motion', additional_callback=self.retrieve_recorded_motion)
        print("stop record robot motion")

    def retrieve_recorded_motion(self, motion):
        """
        :param motion: json string in the following format
        {'robot': 'nao/pepper',
         'compress_factor_angles': int,
         'compress_factor_times: int
         'motion': {'joint1': {'angles': [...], 'times': [...]},
                    'joint2': {...}
                    }
        }
        """
        self.motion = motion
        self.decompressed_motion = self.decompress_motion(self.motion)['motion']  # in the form of dictionary

        traj = []  # in the shape of [joint_num, trajectory_length] as a 2d list
        for joint in self.decompressed_motion.keys():
            traj.append(self.decompressed_motion[joint]['angles'])

        np_traj = np.array(traj)
        traj_length = np_traj.shape[1]  # get the total num of recorded time steps (i.e., trajectory length)
        for t in range(traj_length):
            joints_values = np_traj[:, t]
            self.robot_demo_data.append(joints_values)

    @staticmethod
    def decompress_motion(motion):
        motion = loads(motion)
        precision_factor_angles = float(motion['precision_factor_angles'])
        precision_factor_times = float(motion['precision_factor_times'])
        for joint in motion['motion'].keys():
            motion['motion'][joint]['angles'] = [float(a / precision_factor_angles) for a in
                                                 motion['motion'][joint]['angles']]
            motion['motion'][joint]['times'] = [float(t / precision_factor_times) for t in
                                                motion['motion'][joint]['times']]
        return motion

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

    def stop(self):
        self.listening_thread_robot_data.stop()
        # self.episode_control_thread.join()
        # self.sic.stop()

    def reset(self):
        self.human_demo_data = [np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])] # state dimension is 6
        self.robot_demo_data = [np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])] # action dimension is 6
        self.motion = []

        self.human_hand_position_data = [np.array([np.inf, np.inf, np.inf])]  # (x, y, z) of human right hand in the world frame
        self.robot_hand_position_data = [np.array([np.inf, np.inf, np.inf])] # (x, y, z) of robot right hand in the world frame
        self.robot_hand_ori_data = [np.array([np.inf, np.inf, np.inf])]  # (wx, wy, wz) of robot right hand in the world frame

    def keyboard_control(self):
        while True:
            input = keyboard.read_hotkey()

            if input == "s" or input == "S":
                print("[Keyboard Control]: START")
                self.whether_started = True
                if self.whether_paused:
                    self.whether_paused = False

            elif input == "p" or input == "P":
                print("[Keyboard Control]: PAUSE")
                self.whether_paused = True
                self.stop_record_robot_motion()
                self.save_demo_data()
                print("Succeeded to save demo data!")
                self.reset()
                print("Finish resetting for recording new episode")

            elif input == "q" or input == "Q":
                print("[Keyboard Control]: QUIT")
                if not self.whether_paused:
                    self.whether_paused = True
                    self.stop_record_robot_motion()
                    self.save_demo_data()
                    print("Succeeded to save demo data!")
                break

    def save_demo_data(self):
        """
        :param human_demo_path: type==string
        :param robot_demo_path: type==string:
        """

        # save human demo data to human demo path
        try:
            with open(self.human_demo_path, 'ab') as f_handle:
                np.savetxt(f_handle, self.human_demo_data, fmt='%s')
        except FileNotFoundError:
            with open(self.human_demo_path, 'wb') as f_handle:
                np.savetxt(f_handle, self.human_demo_data, fmt='%s')

        # save robot demo data to robot demo path
        try:
            with open(self.robot_demo_path, 'ab') as f_handle:
                np.savetxt(f_handle, self.robot_demo_data, fmt='%s')
        except FileNotFoundError:
            with open(self.robot_demo_path, 'wb') as f_handle:
                np.savetxt(f_handle, self.robot_demo_data, fmt='%s')

        self.save_human_hand_position_data()
        self.save_robot_hand_position_data()
        self.save_robot_hand_ori_data()

    def save_human_hand_position_data(self):
        try:
            with open(self.human_hand_position_data_path, 'ab') as f_handle:
                np.savetxt(f_handle, self.human_hand_position_data, fmt='%s')
        except FileNotFoundError:
            with open(self.human_hand_position_data_path, 'wb') as f_handle:
                np.savetxt(f_handle, self.human_hand_position_data, fmt='%s')

    def save_robot_hand_position_data(self):
        try:
            with open(self.robot_hand_position_data_path, 'ab') as f_handle:
                np.savetxt(f_handle, self.robot_hand_position_data, fmt='%s')
        except FileNotFoundError:
            with open(self.robot_hand_position_data_path, 'wb') as f_handle:
                np.savetxt(f_handle, self.robot_hand_position_data, fmt='%s')

    def save_robot_hand_ori_data(self):
        try:
            with open(self.robot_hand_ori_data_path, 'ab') as f_handle:
                np.savetxt(f_handle, self.robot_hand_ori_data, fmt='%s')
        except FileNotFoundError:
            with open(self.robot_hand_ori_data_path, 'wb') as f_handle:
                np.savetxt(f_handle, self.robot_hand_ori_data, fmt='%s')

    def say(self, message):
        self.r.publish('say', message)

    def go_home_pos(self):
        self.r.publish('go_home_pose', str(0))

    def set_stiffness(self, stiffness):
        self.r.publish('set_stiffness', str(stiffness))

    def collect_human_data(self, human_pose):
        self.human_demo_data.append(human_pose)

    def collect_robot_data(self, joint_values):
        # self.latest_robot_data = agent.collect_robot_data(role=role)
        self.robot_demo_data.append(joint_values)
        # self.robot_demo_data.append(self.latest_robot_data)

    def collect_human_hand_position(self, human_hand_position):
        self.human_hand_position_data.append(human_hand_position)

    def collect_robot_hand_position(self, robot_hand_position):
        self.robot_hand_position_data.append(robot_hand_position)

    def collect_robot_hand_ori(self, robot_hand_ori):
        self.robot_hand_ori_data.append(robot_hand_ori)

    def collect_robot_data_old(self):
        # send command to robot for retrieving the current joint values
        self.r.publish('retrieve_joint_values', str(0))
        # print("[Collect Robot Data]: Just sent command to retrieve data, going to wait ...")

        while not self.whether_received_robot_data:
            pass

        # print("[Collect Robot Data]: Received robot data")
        self.whether_received_robot_data = False
        self.robot_demo_data.append(self.latest_robot_data)

    def robot_data_message_handler(self, message):
        # print("[Robot Data Handler]: Received robot data is: {}".format(message['data']))
        # print(type(message['data']))
        joint_values = [float(i) for i in message['data'].split()]
        self.latest_robot_data = np.array(joint_values)
        self.whether_received_robot_data = True

    def start_robot_data_listener(self):
        self.sub_robot_data.subscribe(**{'latest_joint_values': self.robot_data_message_handler})
        self.listening_thread_robot_data = self.sub_robot_data.run_in_thread(sleep_time=0.01)



def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_id', help='id of human subject', default=0, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    # # Detector used to detect april-tag and estimate human hand poses
    # tag_size = 0.12  # the real size of the printed april-tag (in meters)
    # detector = Detector(families='tag36h11',
    #                     nthreads=1,
    #                     quad_decimate=1.0,
    #                     quad_sigma=0.0,
    #                     refine_edges=1,
    #                     decode_sharpening=0.25,
    #                     debug=0)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    robot_ip = "192.168.0.121"
    args = argparser()

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        agent = Agent(puppeteer_mode=True)
        recorder = demo_data_recorder(pose, subject_id=args.sub_id)

        # # Load camera calibration parameters
        # parameters_dir = "calibration_chessboard.yml"
        # recorder.calibrate_camera(parameters_dir)

        # Get real-time camera image
        cam_port = 0
        cam = cv2.VideoCapture(cam_port)
        # connector = connect(robot_ip, 12345, 0, 2)
        print("[Connector]: Succeeded to connect to camera")

        # Main loop to get image, estimate pose, and record trajectories of both the human and the robot
        # pose_shift_window = 10  # the initial number of human demo data to collect before starting to record robot demo data
        record_delta_t = 0.05
        episodes_num = 10
        episode_time_length = 3.0
        episode_total_steps = int(episode_time_length / record_delta_t)


        """agent.say('Hello! Nice to meet you!', role='performer')
        sleep(0.5)
        agent.say('I am so happy that you could teach me to do the greeting', role='performer')
        sleep(1.0)"""
        agent.say("Please do greeting with me after I say start", role='performer')
        sleep(1.0)

        for i in range(episodes_num):
            current_step = 0
            recorder.reset()

            # recorder.say("I am going to record a new demo")
            choice = np.random.choice(3, 1)[0]
            if i >= 1:
                if choice == 0:
                    agent.say("Okay, let's do it again", role='performer')
                elif choice == 1:
                    agent.say("Alright, time to do it one more time", role='performer')
                else:
                    agent.say("Come on, let's do another group", role='performer')
            else:
                agent.say("Now let's record our first greeting", role='performer')
            sleep(0.5)
            # sleep(2.0)
            # recorder.say("Episode " + str(i))
            agent.say("Episode " + str(i + 1), role='performer')
            sleep(1.0)
            # sleep(1.0)
            # recorder.say("Start")
            agent.say("Ready?", role='performer')
            sleep(1.0)
            agent.say("Start!", role='performer')
            # recorder.set_stiffness(0.0)
            agent.set_stiffness(0.0, role='puppeteer')
            agent.set_stiffness(1.0, role='performer')
            # sleep(1.0)

            print("********************************")
            print("[Episode {}]: Start".format(i + 1))

            while current_step < episode_total_steps:
                # image = connector.get_img()
                result, image = cam.read()
                if not result:
                    print("[Camera]: Fail to get image")
                    continue

                human_pose, image = recorder.estimate_pose(image)

                if human_pose is None:
                    # recorder.say("Sorry, I didn't see any human")
                    agent.say("Oh no, I can't see your nice looking face", role='performer')
                    # sleep(2.0)
                    print("[Camera]: Fail to detect human pose")
                    continue
                else:
                    recorder.collect_human_data(human_pose)
                    # print("[Episode {}, step: {}]: Succeeded to collect human data".format(i, current_step))
                    # recorder.collect_robot_data(agent)
                    data = agent.collect_robot_data(role='puppeteer')
                    recorder.collect_robot_data(joint_values=data)

                    agent.take_action(action=data, role='performer')
                    # recorder.collect_robot_data_old()
                    print("[Episode {}, step: {}]: Succeeded to collect robot data".format(i + 1, current_step))
                    current_step += 1

                    # cv2.imshow("Cam Image", image)
                    # cv2.waitKey(int(record_delta_t * 1000))  # in the unit of millisecond
                    sleep(record_delta_t)
                    # sleep(0.01)

            print("[Episode {}]: Finished".format(i + 1))
            # recorder.say("Episode " + str(i) + " finished")
            agent.say("Great! Episode " + str(i + 1) + " finished", role='performer')
            # print("Going to set stiffness... ")
            # agent.say('Going to set stiffness to 1')
            agent.set_stiffness(1.0, role='puppeteer')
            agent.set_stiffness(1.0, role='performer')
            # agent.say('Finished setting stiffness to 1')
            print("Finished set stiffness to 1.0")

            # sleep(1.0)
            recorder.save_demo_data()
            # recorder.say("Demo data are saved")
            print("[Episode {}]: Demo data saved".format(i + 1))

            choice = np.random.choice(3, 1)[0]
            if choice == 0:
                agent.say("Take a breath! Let me save the data first", role='performer')
            elif choice == 1:
                agent.say("You could take a rest now! I need to save the data", role='performer')
            else:
                agent.say("Take a rest! Let me save the data", role='performer')
            # sleep(0.5)
            # recorder.set_stiffness(1.0)
            # sleep(1.0)
            # recorder.say("I am going back to the home position")
            # sleep(2.0)
            # recorder.go_home_pos()
            agent.go_home_pose(role='performer')
            agent.go_home_pose(role='puppeteer')
            # sleep(2.0)

        '''
        recorder.start()
        print("[Recorder]: Start to record")

        while not keyboard.is_pressed("q") or not keyboard.is_pressed("Q"):
            # reading the input using the camera
            # result, image = cam.read()
            image = connector.get_img()

            # if result:
            human_pose, image = recorder.estimate_pose(image)
            # store human pose (i.e., april tag) when detected
            if (human_pose is not None) and recorder.whether_started and (not recorder.whether_paused):
                recorder.collect_human_demo_data(human_pose)

                if len(recorder.human_demo_data) > pose_shift_window and not recorder.whether_start_robot_demo:
                    recorder.whether_start_robot_demo = True
                    recorder.start_collect_robot_demo_data()
                    print("[Recorder]: Start to record robot motion")

            cv2.imshow("Cam Image", image)
            cv2.waitKey(int(record_delta_t * 1000))  # in the unit of millisecond
                # cv2.destroyWindow("Test")

            # If captured image is corrupted, moving to else part
            # else:
            #     print("No image detected. Please! try again")
        '''

        # Stop recording and close the cam
        # print("[Recorder]: User quits")
        # recorder.say("Great! We have finished all the recordings! Thank you so much")
        agent.say("Great! We have finished all the recordings! Thank you so much", role='performer')
        # sleep(2.0)
        # recorder.stop()
        agent.stop()
        # cv2.destroyWindow("Cam Image")
        # cam.release()
        print("Process finished")
