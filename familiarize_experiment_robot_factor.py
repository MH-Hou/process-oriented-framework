import numpy as np
from time import sleep
import mediapipe as mp
import cv2

from agent import Agent
from record_demo_data import argparser, demo_data_recorder
from robot_factor_experiment_wizard import get_human_hand_position


def main():
    # prepare episode-related parameters
    step_delta_t = 0.05
    episode_time_length = 3.0
    episode_total_steps = int(episode_time_length / step_delta_t)
    total_trial_num = 4

    # prepare camera-related variables
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    # get human subject id for data saving
    args = argparser()  # --sub_id
    experiment_data_dir = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/experiment_data/experiment_intro/'

    """ Start Experiments """
    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        agent = Agent(puppeteer_mode=True)
        recorder = demo_data_recorder(pose, mp_pose=mp_pose, mp_drawing=mp_drawing, mp_drawing_styles=mp_drawing_styles,
                                      subject_id=args.sub_id)
        recorder.reset_data_saving_path(experiment_data_dir)

        cam_port = 0
        cam = cv2.VideoCapture(cam_port)
        print("[Connector]: Succeeded to connect to camera")

        # agent.set_stiffness_whole_body(0.0, role='puppeteer')

        user_input = input("Press Enter to continue next section ...")
        trial_id = 0

        while not user_input == 'q':
            current_step = 0
            recorder.reset()

            agent.say("We are going to practice a new trial of greeting", role='performer')
            sleep(1.0)
            agent.say("Trial {}".format(trial_id + 1), role='performer')
            sleep(0.5)
            agent.say("Please do greeting with me after I say start", role='performer')
            sleep(1.0)
            agent.say("Ready?", role='performer')
            sleep(1.0)
            agent.say("Start!", role='performer')

            agent.set_stiffness(0.0, role='puppeteer')
            agent.set_stiffness(1.0, role='performer')

            while current_step < episode_total_steps:
                result, image = cam.read()
                if not result:
                    print("[Camera]: Fail to get image")
                    continue

                human_pose, image = recorder.estimate_pose(image)

                if human_pose is None:
                    agent.say("Oh no, I can't see you", role='performer')
                    print("[Camera]: Fail to detect human pose")
                    continue
                else:
                    recorder.collect_human_data(human_pose)
                    human_hand_position = get_human_hand_position(hand_x_in_hip_frame=human_pose[0],
                                                                  hand_y_in_hip_frame=human_pose[1],
                                                                  hand_z_in_hip_frame=human_pose[2])
                    performer_data = agent.collect_robot_data_whole_body(role='performer')
                    robot_hand_pose = agent.get_hand_pose(role='performer')
                    robot_hand_position = np.array([robot_hand_pose[0], robot_hand_pose[1], robot_hand_pose[2]])
                    robot_hand_ori = np.array([robot_hand_pose[3], robot_hand_pose[4], robot_hand_pose[5]])
                    recorder.collect_human_hand_position(human_hand_position=human_hand_position)
                    recorder.collect_robot_data(joint_values=performer_data)
                    recorder.collect_robot_hand_position(robot_hand_position=robot_hand_position)
                    recorder.collect_robot_hand_ori(robot_hand_ori=robot_hand_ori)

                    data = agent.collect_robot_data_whole_body(role='puppeteer')
                    agent.take_action(action=data, role='performer')

                    print("[Trial {}]: Step {} finished".format(trial_id + 1, current_step + 1))
                    current_step += 1

                    # sleep(step_delta_t)
                    sleep(0.01)

            agent.set_stiffness(1.0, role='puppeteer')
            agent.set_stiffness(1.0, role='performer')
            print("Finished set stiffness to 1.0")

            agent.say("Great, we finished this trial", role='performer')
            sleep(1.0)
            agent.say("Take a rest!", role='performer')
            sleep(0.5)
            agent.say("I need some time to save the data", role='performer')
            recorder.save_demo_data()
            print("[Trial {}]: Demo data saved".format(trial_id + 1))

            agent.go_home_pose(role='performer')
            agent.go_home_pose(role='puppeteer')

            print("[Trial {}]: Finished".format(trial_id + 1))
            print('*****************************')
            print('******************************')

            user_input = input("Press Enter to continue next section ...")
            trial_id += 1

        agent.stop()
        cam.release()
        print("Robot-factor familiarize experiments finished")


if __name__ == '__main__':
    main()