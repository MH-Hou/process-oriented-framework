import numpy as np
from time import sleep
import mediapipe as mp
import cv2

from agent import Agent
from record_demo_data import argparser, demo_data_recorder


def get_human_hand_position(hand_x_in_hip_frame, hand_y_in_hip_frame, hand_z_in_hip_frame):
    P_ori_hip2robot = np.array([0.75, 0.0, 0.9])

    # human hand position in human hip frame
    wrist_x_h = hand_x_in_hip_frame
    wrist_y_h = hand_y_in_hip_frame
    wrist_z_h = hand_z_in_hip_frame

    # transform to robot base frame
    wrist_x_r = wrist_z_h + P_ori_hip2robot[0]
    wrist_y_r = - wrist_x_h
    wrist_z_r = - wrist_y_h + P_ori_hip2robot[2]

    wrist_pos_r = np.array([wrist_x_r, wrist_y_r, wrist_z_r])

    return wrist_pos_r


def start_normal_mode(agent, recorder, episode_total_steps, cam, step_delta_t, section_id=1):
    current_step = 0
    recorder.reset()

    """agent.set_stiffness(0.0, role='puppeteer')"""
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
            human_hand_position = get_human_hand_position(hand_x_in_hip_frame=human_pose[0], hand_y_in_hip_frame=human_pose[1], hand_z_in_hip_frame=human_pose[2])
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

            print("[Normal Mode]: Step {} finished".format(current_step + 1))
            current_step += 1

            # sleep(step_delta_t)
            sleep(0.01)

    """agent.set_stiffness(1.0, role='puppeteer')"""
    agent.set_stiffness(1.0, role='performer')
    print("Finished set stiffness to 1.0")


def start_diversion_mode(agent, recorder, episode_total_steps, cam, step_delta_t, section_id):
    if section_id == 0:
        vary_reaching_delay(agent=agent, recorder=recorder, episode_total_steps=episode_total_steps, cam=cam, step_delta_t=step_delta_t, section_id=section_id)
    elif section_id == 1:
        vary_arm_velocity(agent=agent, recorder=recorder, episode_total_steps=episode_total_steps, cam=cam, step_delta_t=step_delta_t, section_id=section_id)
    elif section_id == 2:
        vary_hand_position(agent=agent, recorder=recorder, episode_total_steps=episode_total_steps, cam=cam, step_delta_t=step_delta_t, section_id=section_id)
    elif section_id == 3:
        vary_hip_movement(agent=agent, recorder=recorder, episode_total_steps=episode_total_steps, cam=cam, step_delta_t=step_delta_t, section_id=section_id)
    elif section_id == 4:
        vary_wrist_orientation(agent=agent, recorder=recorder, episode_total_steps=episode_total_steps, cam=cam, step_delta_t=step_delta_t, section_id=section_id)
    elif section_id == 5:
        vary_withdraw_delay(agent=agent, recorder=recorder, episode_total_steps=episode_total_steps, cam=cam, step_delta_t=step_delta_t, section_id=section_id)


def vary_reaching_delay(agent, recorder, episode_total_steps, cam, step_delta_t, section_id=1):
    """agent.set_stiffness(0.0, role='puppeteer')"""
    agent.set_stiffness(1.0, role='performer')

    current_step = 0
    recorder.reset()
    old_data = None
    delay_started = False
    delay_finished = False
    delay_steps = 0
    delay_thres = 10  # total number of steps for reaching delay
    delay_init_thres = 0.005  # joint value threshold to trigger the delay
    depth_thred = 0.02
    starting_depth = None


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

            # add time delay to response
            robot_arm_depth = robot_hand_position[0]
            data = agent.collect_robot_data_whole_body(role='puppeteer')
            if not delay_started:
                if old_data is None:
                    old_data = data
                    starting_depth = robot_arm_depth

                if (robot_arm_depth - starting_depth) >= depth_thred and np.linalg.norm(old_data - data) > delay_init_thres:
                    delay_steps += 1
                    delay_started = True
                    """agent.set_stiffness(1.0, role='puppeteer')"""
                    agent.set_led(1.0, role='puppeteer', led_name='ear_front')  # turn on right ear led
                    print("[Diversion Mode]: Delay started")
                else:
                    old_data = data
            else:
                if not delay_finished:
                    if delay_steps < delay_thres:
                        delay_steps += 1
                        # print("[Diversion Mode]: Delay step {}".format(delay_steps))
                    else:
                        """agent.set_stiffness(0.0, role='puppeteer')"""
                        agent.set_led(0.0, role='puppeteer', led_name='ear_front')  # turn off right ear led
                        # print("[Diversion Mode]: Delay finished")
                        delay_finished = True
                else:
                    pass

            if delay_started and not delay_finished:
                pass
            else:
                agent.take_action(action=data, role='performer')

            print("[Diversion Mode]: Step {} finished".format(current_step + 1))
            current_step += 1

            # sleep(step_delta_t)
            sleep(0.01)

    """agent.set_stiffness(1.0, role='puppeteer')"""
    agent.set_stiffness(1.0, role='performer')
    print("Finished set stiffness to 1.0")


def vary_arm_velocity(agent, recorder, episode_total_steps, cam, step_delta_t, section_id=1):
    """agent.set_stiffness(0.0, role='puppeteer')"""
    agent.set_stiffness(1.0, role='performer')

    current_step = 0
    robot_arm_depth_last = None
    robot_arm_depth = None
    back_steps_total = 0
    back_steps_thres = 5
    whether_stop_vel_vary = False
    recorder.reset()

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

            robot_arm_depth = robot_hand_position[0]
            if not whether_stop_vel_vary:
                # for initiation
                if robot_arm_depth_last is None:
                    robot_arm_depth_last = robot_arm_depth
                # check whether robot arm is withdrawing, with x direction pointing from robot to human
                if robot_arm_depth - robot_arm_depth_last < 0:
                    back_steps_total += 1
                # stop vary arm velocity if robot is withdrawing hands, instead of reaching hands
                if back_steps_total >= back_steps_thres:
                    whether_stop_vel_vary = True
                # update arm depth
                robot_arm_depth_last = robot_arm_depth

            # change the velocity factor to 0.1
            data = agent.collect_robot_data_whole_body(role='puppeteer')

            if not whether_stop_vel_vary:
                agent.take_action(action=data, role='performer', vel=0.05)
            else:
                agent.take_action(action=data, role='performer')

            print("[Diversion Mode]: Step {} finished".format(current_step + 1))
            current_step += 1

            # sleep(step_delta_t)
            sleep(0.01)

    """agent.set_stiffness(1.0, role='puppeteer')"""
    agent.set_stiffness(1.0, role='performer')
    print("Finished set stiffness to 1.0")


def vary_hand_position(agent, recorder, episode_total_steps, cam, step_delta_t, section_id=1):
    """agent.set_stiffness(0.0, role='puppeteer')"""
    agent.set_stiffness(1.0, role='performer')

    current_step = 0
    robot_arm_depth_last = None
    robot_arm_depth = None
    init_depth_thres = 0.1
    back_steps_total = 0
    back_steps_thres = 15
    whether_stop_pos_vary = False
    starting_depth = None
    recorder.reset()

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

            robot_arm_depth = robot_hand_position[0]

            if starting_depth is None:
                starting_depth = robot_arm_depth

            if not whether_stop_pos_vary:
                # for initiation
                if robot_arm_depth_last is None:
                    robot_arm_depth_last = robot_arm_depth
                # check whether robot arm is withdrawing, with x direction pointing from robot to human
                if robot_arm_depth - robot_arm_depth_last < 0:
                    back_steps_total += 1
                # stop vary arm velocity if robot is withdrawing hands, instead of reaching hands
                if back_steps_total >= back_steps_thres:
                    whether_stop_pos_vary = True
                # update arm depth
                robot_arm_depth_last = robot_arm_depth

            # add diversion to original command from puppeteer
            data = agent.collect_robot_data_whole_body(role='puppeteer')
            # if not whether_stop_pos_vary and robot_arm_depth > init_depth_thres:
            #     data[0] -= 0.25  # change the RShoulderPitch, moving it higher than original command
            #     data[3] += 0.5  # change the RShoulderRoll, moving it more to the right than original command
            data[0] -= 0.4  # change the RShoulderPitch, moving it higher than original command
            data[3] += 0.25  # change the RElbowRoll, moving it more to the left than original command

            agent.take_action(action=data, role='performer')

            print("[Diversion Mode]: Step {} finished".format(current_step + 1))
            current_step += 1

            # sleep(step_delta_t)
            sleep(0.01)

    """agent.set_stiffness(1.0, role='puppeteer')"""
    agent.set_stiffness(1.0, role='performer')
    print("Finished set stiffness to 1.0")


def vary_hip_movement(agent, recorder, episode_total_steps, cam, step_delta_t, section_id=1):
    agent.set_stiffness(1.0, role='performer')

    current_step = 0
    recorder.reset()

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

            # allow whole body movement of puppeteer
            data = agent.collect_robot_data_whole_body(role='puppeteer')
            agent.take_action(action=data, role='performer')

            print("[Diversion Mode]: Step {} finished".format(current_step + 1))
            current_step += 1

            # sleep(step_delta_t)
            sleep(0.01)

    agent.set_stiffness(1.0, role='performer')
    # agent.set_stiffness_whole_body(1.0, role='puppeteer')
    # print("Finished set puppeteer stiffness to 1.0")


def vary_wrist_orientation(agent, recorder, episode_total_steps, cam, step_delta_t, section_id=5):
    """agent.set_stiffness(0.0, role='puppeteer')"""
    agent.set_stiffness(1.0, role='performer')

    current_step = 0
    robot_arm_depth_last = None
    robot_arm_depth = None
    arm_depth_init_thres = 0.1
    back_steps_total = 0
    back_steps_thres = 10
    whether_stop_ori_vary = False
    starting_depth = None
    recorder.reset()

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

            robot_arm_depth = robot_hand_position[0]

            if starting_depth is None:
                starting_depth = robot_arm_depth

            if not whether_stop_ori_vary:
                # for initiation
                if robot_arm_depth_last is None:
                    robot_arm_depth_last = robot_arm_depth
                # check whether robot arm is withdrawing, with x direction pointing from robot to human
                if robot_arm_depth - robot_arm_depth_last < 0:
                    back_steps_total += 1
                # stop vary arm velocity if robot is withdrawing hands, instead of reaching hands
                if back_steps_total >= back_steps_thres:
                    whether_stop_ori_vary = True
                # update arm depth
                robot_arm_depth_last = robot_arm_depth

            # allow whole body movement of puppeteer
            data = agent.collect_robot_data_whole_body(role='puppeteer')
            if not whether_stop_ori_vary and (robot_arm_depth - starting_depth) >= arm_depth_init_thres:
                data[4] = 45.0 / 180.0 * np.pi

            agent.take_action(action=data, role='performer')
            # if not whether_stop_ori_vary and robot_arm_depth >= arm_depth_init_thres:
            #     agent.take_action_wrist(action=0.0 / 180.0 * np.pi, role='performer')
            # else:
            #     # agent.take_action_wrist(action=0.0, role='performer')
            #     pass

            print("[Diversion Mode]: Step {} finished".format(current_step + 1))
            current_step += 1

            # sleep(step_delta_t)
            sleep(0.01)

    """agent.set_stiffness(1.0, role='puppeteer')"""
    agent.set_stiffness(1.0, role='performer')
    print("Finished set stiffness to 1.0")


def vary_withdraw_delay(agent, recorder, episode_total_steps, cam, step_delta_t, section_id=6):
    """agent.set_stiffness(0.0, role='puppeteer')"""
    agent.set_stiffness(1.0, role='performer')

    current_step = 0
    old_data = None
    delay_started = False
    delay_finished = False
    delay_steps = 0
    still_steps = 0
    delay_thres = 20  # total number of steps for withdrawing delay
    still_steps_thres = 3
    delay_init_thres = 0.02  # joint value threshold to trigger the delay
    depth_thred = 0.1
    starting_depth = None
    recorder.reset()

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

            robot_arm_depth = robot_hand_position[0]

            # add time delay to response
            data = robot_arm_depth
            if not delay_started:
                if old_data is None:
                    old_data = data
                    starting_depth = robot_arm_depth

                if (robot_arm_depth - starting_depth) >= depth_thred and abs(data - old_data) <= delay_init_thres:
                    still_steps += 1
                    if still_steps >= still_steps_thres:
                        delay_steps += 1
                        delay_started = True
                        """agent.set_stiffness(1.0, role='puppeteer')"""
                        agent.set_led(1.0, role='puppeteer', led_name='ear_front')  # turn on right ear led
                        print("[Diversion Mode]: Delay started")
                        print("robot arm depth: {}".format(robot_arm_depth))
                    else:
                        old_data = data
                else:
                    old_data = data
            else:
                if not delay_finished:
                    if delay_steps < delay_thres:
                        delay_steps += 1
                        # print("[Diversion Mode]: Delay step {}".format(delay_steps))
                    else:
                        """agent.set_stiffness(0.0, role='puppeteer')"""
                        agent.set_led(0.0, role='puppeteer', led_name='ear_front')  # turn off right ear led
                        # print("[Diversion Mode]: Delay finished")
                        delay_finished = True
                else:
                    pass

            if delay_started and not delay_finished:
                pass
            else:
                # allow whole body movement of puppeteer
                data = agent.collect_robot_data_whole_body(role='puppeteer')
                agent.take_action(action=data, role='performer')

            print("[Diversion Mode]: Step {} finished".format(current_step + 1))
            current_step += 1

            # sleep(step_delta_t)
            sleep(0.01)

    """agent.set_stiffness(1.0, role='puppeteer')"""
    agent.set_stiffness(1.0, role='performer')
    print("Finished set stiffness to 1.0")

#########################################################

def vary_reaching_delay_old(agent, recorder, episode_total_steps, cam, step_delta_t, section_id=1):
    # current_step = 0
    # recorder.reset()

    agent.say("We are going to start a new section of experiments", role='performer')
    sleep(1.0)
    agent.say("Section {}".format(section_id), role='performer')
    sleep(0.5)

    """ Start "Normal Mode" experiment """
    agent.say("Please do greeting with me for the first time", role='performer')
    sleep(1.0)
    agent.say("Ready?", role='performer')
    sleep(1.0)
    agent.say("Start!", role='performer')

    # agent.set_stiffness(0.0, role='puppeteer')
    # agent.set_stiffness(1.0, role='performer')

    start_normal_mode(agent=agent, recorder=recorder, episode_total_steps=episode_total_steps, cam=cam, step_delta_t=step_delta_t, section_id=section_id)

    # while current_step < episode_total_steps:
    #     result, image = cam.read()
    #     if not result:
    #         print("[Camera]: Fail to get image")
    #         continue
    #
    #     human_pose, image = recorder.estimate_pose(image)
    #
    #     if human_pose is None:
    #         agent.say("Oh no, I can't see you", role='performer')
    #         print("[Camera]: Fail to detect human pose")
    #         continue
    #     else:
    #         recorder.collect_human_data(human_pose)
    #         human_hand_position = get_human_hand_position(hand_x_in_hip_frame=human_pose[0], hand_y_in_hip_frame=human_pose[1], hand_z_in_hip_frame=human_pose[2])
    #         performer_data = agent.collect_robot_data(role='performer')
    #         robot_hand_pose = agent.get_hand_pose(role='performer')
    #         robot_hand_position = np.array([robot_hand_pose[0], robot_hand_pose[1], robot_hand_pose[2]])
    #         robot_hand_ori = np.array([robot_hand_pose[3], robot_hand_pose[4], robot_hand_pose[5]])
    #         recorder.collect_human_hand_position(human_hand_position=human_hand_position)
    #         recorder.collect_robot_data(joint_values=performer_data)
    #         recorder.collect_robot_hand_position(robot_hand_position=robot_hand_position)
    #         recorder.collect_robot_hand_ori(robot_hand_ori=robot_hand_ori)
    #
    #         data = agent.collect_robot_data(role='puppeteer')
    #         agent.take_action(action=data, role='performer')
    #
    #         print("[Normal Mode]: Step {} finished".format(current_step + 1))
    #         current_step += 1
    #
    #         # sleep(step_delta_t)
    #         sleep(0.01)

    # agent.set_stiffness(1.0, role='puppeteer')
    # agent.set_stiffness(1.0, role='performer')
    # print("Finished set stiffness to 1.0")

    agent.say("Great, we finished the first greeting", role='performer')
    sleep(1.0)

    # agent.set_stiffness(1.0, role='puppeteer')
    # agent.set_stiffness(1.0, role='performer')
    # print("Finished set stiffness to 1.0")

    agent.say("Wait a second, I need to save the data", role='performer')
    recorder.save_demo_data()
    print("[Normal Mode]: Demo data saved")

    agent.go_home_pose(role='performer')
    agent.go_home_pose(role='puppeteer')

    # sleep(1.0)

    print('#########################################')

    """ Start "Diversion Mode" Experiment """
    agent.say("Okay, Data saved! Let's continue!", role='performer')
    sleep(1.0)
    agent.say("Please do greeting with me again", role='performer')
    sleep(1.0)
    agent.say("Ready?", role='performer')
    sleep(1.0)
    agent.say("Start!", role='performer')

    agent.set_stiffness(0.0, role='puppeteer')
    agent.set_stiffness(1.0, role='performer')

    current_step = 0
    recorder.reset()
    old_data = None
    delay_started = False
    delay_finished = False
    delay_steps = 0
    delay_thres = 10 # total number of steps for reaching delay
    delay_init_thres = 0.02 # joint value threshold to trigger the delay

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
            performer_data = agent.collect_robot_data(role='performer')
            robot_hand_pose = agent.get_hand_pose(role='performer')
            robot_hand_position = np.array([robot_hand_pose[0], robot_hand_pose[1], robot_hand_pose[2]])
            robot_hand_ori = np.array([robot_hand_pose[3], robot_hand_pose[4], robot_hand_pose[5]])
            recorder.collect_human_hand_position(human_hand_position=human_hand_position)
            recorder.collect_robot_data(joint_values=performer_data)
            recorder.collect_robot_hand_position(robot_hand_position=robot_hand_position)
            recorder.collect_robot_hand_ori(robot_hand_ori=robot_hand_ori)

            # add time delay to response
            data = agent.collect_robot_data(role='puppeteer')
            if not delay_started:
                if old_data is None:
                    old_data = data

                if np.linalg.norm(old_data - data) > delay_init_thres :
                    delay_steps += 1
                    delay_started = True
                    agent.set_stiffness(1.0, role='puppeteer')
                    agent.set_led(1.0, role='puppeteer', led_name='ear') # turn on right ear led
                    print("[Diversion Mode]: Delay started")
                else:
                    old_data = data
            else:
                if not delay_finished:
                    if delay_steps < delay_thres:
                        delay_steps += 1
                        print("[Diversion Mode]: Delay step {}".format(delay_steps))
                    else:
                        agent.set_stiffness(0.0, role='puppeteer')
                        agent.set_led(0.0, role='puppeteer', led_name='ear') # turn off right ear led
                        print("[Diversion Mode]: Delay finished")
                        delay_finished = True
                else:
                    pass

            agent.take_action(action=data, role='performer')

            print("[Diversion Mode]: Step {} finished".format(current_step + 1))
            current_step += 1

            # sleep(step_delta_t)
            sleep(0.01)

    agent.set_stiffness(1.0, role='puppeteer')
    agent.set_stiffness(1.0, role='performer')
    print("Finished set stiffness to 1.0")

    agent.say("Great! We finished this section of experiment!", role='performer')
    sleep(1.0)

    # agent.set_stiffness(1.0, role='puppeteer')
    # agent.set_stiffness(1.0, role='performer')
    # print("Finished set stiffness to 1.0")
    agent.say("Please take a rest and fill out the questionnaire", role='performer')
    recorder.save_demo_data()
    print("[Diversion Mode]: Demo data saved")

    agent.go_home_pose(role='performer')
    agent.go_home_pose(role='puppeteer')

    # sleep(1.0)


def vary_arm_velocity_old(agent, recorder, episode_total_steps, cam, step_delta_t, section_id=2):
    current_step = 0
    # recorder.reset()

    agent.say("We are going to start a new section of experiments", role='performer')
    sleep(1.0)
    agent.say("Section {}".format(section_id), role='performer')
    sleep(0.5)

    """ Start "Normal Mode" experiment """
    agent.say("Please do greeting with me for the first time", role='performer')
    sleep(1.0)
    agent.say("Ready?", role='performer')
    sleep(1.0)
    agent.say("Start!", role='performer')

    # agent.set_stiffness(0.0, role='puppeteer')
    # agent.set_stiffness(1.0, role='performer')

    # while current_step < episode_total_steps:
    #     result, image = cam.read()
    #     if not result:
    #         print("[Camera]: Fail to get image")
    #         continue
    #
    #     human_pose, image = recorder.estimate_pose(image)
    #
    #     if human_pose is None:
    #         agent.say("Oh no, I can't see you", role='performer')
    #         print("[Camera]: Fail to detect human pose")
    #         continue
    #     else:
    #         recorder.collect_human_data(human_pose)
    #         human_hand_position = get_human_hand_position(hand_x_in_hip_frame=human_pose[0],
    #                                                       hand_y_in_hip_frame=human_pose[1],
    #                                                       hand_z_in_hip_frame=human_pose[2])
    #         performer_data = agent.collect_robot_data(role='performer')
    #         robot_hand_pose = agent.get_hand_pose(role='performer')
    #         robot_hand_position = np.array([robot_hand_pose[0], robot_hand_pose[1], robot_hand_pose[2]])
    #         robot_hand_ori = np.array([robot_hand_pose[3], robot_hand_pose[4], robot_hand_pose[5]])
    #         recorder.collect_human_hand_position(human_hand_position=human_hand_position)
    #         recorder.collect_robot_data(joint_values=performer_data)
    #         recorder.collect_robot_hand_position(robot_hand_position=robot_hand_position)
    #         recorder.collect_robot_hand_ori(robot_hand_ori=robot_hand_ori)
    #
    #         data = agent.collect_robot_data(role='puppeteer')
    #         agent.take_action(action=data, role='performer')
    #
    #         print("[Normal Mode]: Step {} finished".format(current_step + 1))
    #         current_step += 1
    #
    #         # sleep(step_delta_t)
    #         sleep(0.01)

    start_normal_mode(agent=agent, recorder=recorder, episode_total_steps=episode_total_steps, cam=cam, step_delta_t=step_delta_t, section_id=section_id)

    agent.say("Great, we finished the first greeting", role='performer')
    sleep(1.0)
    # agent.set_stiffness(1.0, role='puppeteer')
    # agent.set_stiffness(1.0, role='performer')
    # print("Finished set stiffness to 1.0")
    agent.say("Wait a second. I need to save the data", role='performer')
    recorder.save_demo_data()
    print("[Normal Mode]: Demo data saved")
    agent.go_home_pose(role='performer')
    agent.go_home_pose(role='puppeteer')

    # sleep(1.0)

    print('#########################################')

    """ Start "Diversion Mode" Experiment """
    agent.say("Okay, Data saved! Let's continue!", role='performer')
    sleep(1.0)
    agent.say("Please do greeting with me again", role='performer')
    sleep(1.0)
    agent.say("Ready?", role='performer')
    sleep(1.0)
    agent.say("Start!", role='performer')

    agent.set_stiffness(0.0, role='puppeteer')
    agent.set_stiffness(1.0, role='performer')

    current_step = 0
    robot_arm_depth_last = None
    robot_arm_depth = None
    back_steps_total = 0
    back_steps_thres = 5
    whether_stop_vel_vary = False
    recorder.reset()

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
            performer_data = agent.collect_robot_data(role='performer')
            robot_hand_pose = agent.get_hand_pose(role='performer')
            robot_hand_position = np.array([robot_hand_pose[0], robot_hand_pose[1], robot_hand_pose[2]])
            robot_hand_ori = np.array([robot_hand_pose[3], robot_hand_pose[4], robot_hand_pose[5]])
            recorder.collect_human_hand_position(human_hand_position=human_hand_position)
            recorder.collect_robot_data(joint_values=performer_data)
            recorder.collect_robot_hand_position(robot_hand_position=robot_hand_position)
            recorder.collect_robot_hand_ori(robot_hand_ori=robot_hand_ori)

            robot_arm_depth = robot_hand_position[0]
            if not whether_stop_vel_vary:
                # for initiation
                if robot_arm_depth_last is None:
                    robot_arm_depth_last = robot_arm_depth
                # check whether robot arm is withdrawing, with x direction pointing from robot to human
                if robot_arm_depth - robot_arm_depth_last < 0:
                    back_steps_total += 1
                # stop vary arm velocity if robot is withdrawing hands, instead of reaching hands
                if back_steps_total >= back_steps_thres:
                    whether_stop_vel_vary = True
                # update arm depth
                robot_arm_depth_last = robot_arm_depth

            # change the velocity factor to 0.1
            data = agent.collect_robot_data(role='puppeteer')

            if not whether_stop_vel_vary:
                agent.take_action(action=data, role='performer', vel=0.1)
            else:
                agent.take_action(action=data, role='performer')

            print("[Diversion Mode]: Step {} finished".format(current_step + 1))
            current_step += 1

            # sleep(step_delta_t)
            sleep(0.01)

    agent.set_stiffness(1.0, role='puppeteer')
    agent.set_stiffness(1.0, role='performer')
    print("Finished set stiffness to 1.0")

    agent.say("Great! We finished this section of experiment!", role='performer')
    sleep(1.0)
    # agent.set_stiffness(1.0, role='puppeteer')
    # agent.set_stiffness(1.0, role='performer')
    # print("Finished set stiffness to 1.0")
    agent.say("Please take a rest and fill out the questionnaire", role='performer')
    recorder.save_demo_data()
    print("[Diversion Mode]: Demo data saved")
    agent.go_home_pose(role='performer')
    agent.go_home_pose(role='puppeteer')

    # sleep(1.0)


def vary_hand_position_old(agent, recorder, episode_total_steps, cam, step_delta_t, section_id=3):
    current_step = 0
    # recorder.reset()

    agent.say("We are going to start a new section of experiments", role='performer')
    sleep(1.0)
    agent.say("Section {}".format(section_id), role='performer')
    sleep(0.5)

    """ Start "Normal Mode" experiment """
    agent.say("Please do greeting with me for the first time", role='performer')
    sleep(1.0)
    agent.say("Ready?", role='performer')
    sleep(1.0)
    agent.say("Start!", role='performer')

    # agent.set_stiffness(0.0, role='puppeteer')
    # agent.set_stiffness(1.0, role='performer')
    #
    # while current_step < episode_total_steps:
    #     result, image = cam.read()
    #     if not result:
    #         print("[Camera]: Fail to get image")
    #         continue
    #
    #     human_pose, image = recorder.estimate_pose(image)
    #
    #     if human_pose is None:
    #         agent.say("Oh no, I can't see you", role='performer')
    #         print("[Camera]: Fail to detect human pose")
    #         continue
    #     else:
    #         recorder.collect_human_data(human_pose)
    #         human_hand_position = get_human_hand_position(hand_x_in_hip_frame=human_pose[0],
    #                                                       hand_y_in_hip_frame=human_pose[1],
    #                                                       hand_z_in_hip_frame=human_pose[2])
    #         performer_data = agent.collect_robot_data(role='performer')
    #         robot_hand_pose = agent.get_hand_pose(role='performer')
    #         robot_hand_position = np.array([robot_hand_pose[0], robot_hand_pose[1], robot_hand_pose[2]])
    #         robot_hand_ori = np.array([robot_hand_pose[3], robot_hand_pose[4], robot_hand_pose[5]])
    #         recorder.collect_human_hand_position(human_hand_position=human_hand_position)
    #         recorder.collect_robot_data(joint_values=performer_data)
    #         recorder.collect_robot_hand_position(robot_hand_position=robot_hand_position)
    #         recorder.collect_robot_hand_ori(robot_hand_ori=robot_hand_ori)
    #
    #         # hand_position = agent.get_hand_position(role='performer')
    #         # print("hand position: x: {}, y: {}, z: {}".format(hand_position[0], hand_position[1], hand_position[2]))
    #
    #         data = agent.collect_robot_data(role='puppeteer')
    #         agent.take_action(action=data, role='performer')
    #
    #         print("[Normal Mode]: Step {} finished".format(current_step + 1))
    #         current_step += 1
    #
    #         # sleep(step_delta_t)
    #         sleep(0.01)
    start_normal_mode(agent=agent, recorder=recorder, episode_total_steps=episode_total_steps, cam=cam, step_delta_t=step_delta_t, section_id=section_id)


    agent.say("Okay, we finished the first greeting", role='performer')
    sleep(1.0)
    # agent.set_stiffness(1.0, role='puppeteer')
    # agent.set_stiffness(1.0, role='performer')
    # print("Finished set stiffness to 1.0")
    agent.say("Wait a second, I need to save the data.", role='performer')
    recorder.save_demo_data()
    print("[Normal Mode]: Demo data saved")
    agent.go_home_pose(role='performer')
    agent.go_home_pose(role='puppeteer')

    # sleep(1.0)

    print('#########################################')

    """ Start "Diversion Mode" Experiment """
    agent.say("Okay, data saved! Let's continue!", role='performer')
    sleep(1.0)
    agent.say("Please do greeting with me again", role='performer')
    sleep(1.0)
    agent.say("Ready?", role='performer')
    sleep(1.0)
    agent.say("Start!", role='performer')

    agent.set_stiffness(0.0, role='puppeteer')
    agent.set_stiffness(1.0, role='performer')

    current_step = 0
    robot_arm_depth_last = None
    robot_arm_depth = None
    back_steps_total = 0
    back_steps_thres = 10
    whether_stop_pos_vary = False
    recorder.reset()

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
            performer_data = agent.collect_robot_data(role='performer')
            robot_hand_pose = agent.get_hand_pose(role='performer')
            robot_hand_position = np.array([robot_hand_pose[0], robot_hand_pose[1], robot_hand_pose[2]])
            robot_hand_ori = np.array([robot_hand_pose[3], robot_hand_pose[4], robot_hand_pose[5]])
            recorder.collect_human_hand_position(human_hand_position=human_hand_position)
            recorder.collect_robot_data(joint_values=performer_data)
            recorder.collect_robot_hand_position(robot_hand_position=robot_hand_position)
            recorder.collect_robot_hand_ori(robot_hand_ori=robot_hand_ori)

            robot_arm_depth = robot_hand_position[0]
            if not whether_stop_pos_vary:
                # for initiation
                if robot_arm_depth_last is None:
                    robot_arm_depth_last = robot_arm_depth
                # check whether robot arm is withdrawing, with x direction pointing from robot to human
                if robot_arm_depth - robot_arm_depth_last < 0:
                    back_steps_total += 1
                # stop vary arm velocity if robot is withdrawing hands, instead of reaching hands
                if back_steps_total >= back_steps_thres:
                    whether_stop_pos_vary = True
                # update arm depth
                robot_arm_depth_last = robot_arm_depth

            # add diversion to original command from puppeteer
            data = agent.collect_robot_data(role='puppeteer')
            if not whether_stop_pos_vary:
                data[0] -= 0.4 # change the RShoulderPitch, moving it higher than original command
                data[1] -= 0.25 # change the RShoulderRoll, moving it more to the right than original command

            agent.take_action(action=data, role='performer')

            print("[Diversion Mode]: Step {} finished".format(current_step + 1))
            current_step += 1

            # sleep(step_delta_t)
            sleep(0.01)

    agent.set_stiffness(1.0, role='puppeteer')
    agent.set_stiffness(1.0, role='performer')
    print("Finished set stiffness to 1.0")

    agent.say("Great! We finished this section of experiment!", role='performer')
    sleep(1.0)
    # agent.set_stiffness(1.0, role='puppeteer')
    # agent.set_stiffness(1.0, role='performer')
    # print("Finished set stiffness to 1.0")
    agent.say("Please take a rest and fill out the questionnaire", role='performer')
    recorder.save_demo_data()
    print("[Diversion Mode]: Demo data saved")
    agent.go_home_pose(role='performer')
    agent.go_home_pose(role='puppeteer')

    # sleep(1.0)


def vary_hip_movement_old(agent, recorder, episode_total_steps, cam, step_delta_t, section_id=4):
    # current_step = 0
    # recorder.reset()

    agent.say("We are going to start a new section of experiments", role='performer')
    sleep(1.0)
    agent.say("Section {}".format(section_id), role='performer')
    sleep(0.5)

    """ Start "Normal Mode" experiment """
    agent.say("Please do greeting with me for the first time", role='performer')
    sleep(1.0)
    agent.say("Ready?", role='performer')
    sleep(1.0)
    agent.say("Start!", role='performer')

    # agent.set_stiffness(0.0, role='puppeteer')
    # agent.set_stiffness(1.0, role='performer')
    #
    # while current_step < episode_total_steps:
    #     result, image = cam.read()
    #     if not result:
    #         print("[Camera]: Fail to get image")
    #         continue
    #
    #     human_pose, image = recorder.estimate_pose(image)
    #
    #     if human_pose is None:
    #         agent.say("Oh no, I can't see you", role='performer')
    #         print("[Camera]: Fail to detect human pose")
    #         continue
    #     else:
    #         recorder.collect_human_data(human_pose)
    #         human_hand_position = get_human_hand_position(hand_x_in_hip_frame=human_pose[0],
    #                                                       hand_y_in_hip_frame=human_pose[1],
    #                                                       hand_z_in_hip_frame=human_pose[2])
    #         performer_data = agent.collect_robot_data(role='performer')
    #         robot_hand_pose = agent.get_hand_pose(role='performer')
    #         robot_hand_position = np.array([robot_hand_pose[0], robot_hand_pose[1], robot_hand_pose[2]])
    #         robot_hand_ori = np.array([robot_hand_pose[3], robot_hand_pose[4], robot_hand_pose[5]])
    #         recorder.collect_human_hand_position(human_hand_position=human_hand_position)
    #         recorder.collect_robot_data(joint_values=performer_data)
    #         recorder.collect_robot_hand_position(robot_hand_position=robot_hand_position)
    #         recorder.collect_robot_hand_ori(robot_hand_ori=robot_hand_ori)
    #
    #         # hand_position = agent.get_hand_position(role='performer')
    #         # print("hand position: x: {}, y: {}, z: {}".format(hand_position[0], hand_position[1], hand_position[2]))
    #
    #         data = agent.collect_robot_data(role='puppeteer')
    #         agent.take_action(action=data, role='performer')
    #
    #         print("[Normal Mode]: Step {} finished".format(current_step + 1))
    #         current_step += 1
    #
    #         # sleep(step_delta_t)
    #         sleep(0.01)

    start_normal_mode(agent=agent, recorder=recorder, episode_total_steps=episode_total_steps, cam=cam,
                      step_delta_t=step_delta_t, section_id=section_id)

    # agent.set_stiffness(1.0, role='puppeteer')
    # agent.set_stiffness(1.0, role='performer')
    # print("Finished set stiffness to 1.0")

    agent.say("Okay, we finished the first greeting", role='performer')
    sleep(1.0)

    agent.say("Wait a second, I need to save the data.", role='performer')
    recorder.save_demo_data()
    print("[Normal Mode]: Demo data saved")
    agent.go_home_pose(role='performer')
    agent.go_home_pose(role='puppeteer')

    # sleep(1.0)

    print('#########################################')

    agent.set_stiffness_whole_body(0.0, role='puppeteer')

    """ Start "Diversion Mode" Experiment """
    agent.say("Okay, data saved! Let's continue!", role='performer')
    sleep(1.0)
    agent.say("Please do greeting with me again", role='performer')
    sleep(1.0)
    agent.say("Ready?", role='performer')
    sleep(1.0)
    agent.say("Start!", role='performer')

    # agent.set_stiffness(0.0, role='puppeteer')
    # agent.set_stiffness(1.0, role='performer')
    #
    # current_step = 0
    # recorder.reset()
    #
    # while current_step < episode_total_steps:
    #     result, image = cam.read()
    #     if not result:
    #         print("[Camera]: Fail to get image")
    #         continue
    #
    #     human_pose, image = recorder.estimate_pose(image)
    #
    #     if human_pose is None:
    #         agent.say("Oh no, I can't see you", role='performer')
    #         print("[Camera]: Fail to detect human pose")
    #         continue
    #     else:
    #         recorder.collect_human_data(human_pose)
    #         human_hand_position = get_human_hand_position(hand_x_in_hip_frame=human_pose[0],
    #                                                       hand_y_in_hip_frame=human_pose[1],
    #                                                       hand_z_in_hip_frame=human_pose[2])
    #         performer_data = agent.collect_robot_data(role='performer')
    #         robot_hand_pose = agent.get_hand_pose(role='performer')
    #         robot_hand_position = np.array([robot_hand_pose[0], robot_hand_pose[1], robot_hand_pose[2]])
    #         robot_hand_ori = np.array([robot_hand_pose[3], robot_hand_pose[4], robot_hand_pose[5]])
    #         recorder.collect_human_hand_position(human_hand_position=human_hand_position)
    #         recorder.collect_robot_data(joint_values=performer_data)
    #         recorder.collect_robot_hand_position(robot_hand_position=robot_hand_position)
    #         recorder.collect_robot_hand_ori(robot_hand_ori=robot_hand_ori)
    #
    #         # allow whole body movement of puppeteer
    #         data = agent.collect_robot_data_whole_body(role='puppeteer')
    #         agent.take_action(action=data, role='performer')
    #
    #         print("[Diversion Mode]: Step {} finished".format(current_step + 1))
    #         current_step += 1
    #
    #         # sleep(step_delta_t)
    #         sleep(0.01)
    #
    # agent.set_stiffness(1.0, role='performer')
    # agent.set_stiffness_whole_body(1.0, role='puppeteer')
    # print("Finished set puppeteer stiffness to 1.0")

    agent.say("Great! We finished this section of experiment!", role='performer')
    sleep(1.0)
    agent.say("Please take a rest and fill out the questionnaire", role='performer')
    recorder.save_demo_data()
    print("[Diversion Mode]: Demo data saved")
    # agent.set_stiffness(1.0, role='performer')
    agent.go_home_pose(role='performer')
    # agent.set_stiffness_whole_body(1.0, role='puppeteer')
    # print("Finished set puppeteer stiffness to 1.0")
    agent.go_home_pose(role='puppeteer')

    # sleep(1.0)


def vary_wrist_orientation_old(agent, recorder, episode_total_steps, cam, step_delta_t, section_id=5):
    current_step = 0
    recorder.reset()

    agent.say("We are going to start a new section of experiments", role='performer')
    sleep(1.0)
    agent.say("Section {}".format(section_id), role='performer')
    sleep(0.5)

    """ Start "Normal Mode" experiment """
    agent.say("Please do greeting with me for the first time", role='performer')
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
            performer_data = agent.collect_robot_data(role='performer')
            robot_hand_pose = agent.get_hand_pose(role='performer')
            robot_hand_position = np.array([robot_hand_pose[0], robot_hand_pose[1], robot_hand_pose[2]])
            robot_hand_ori = np.array([robot_hand_pose[3], robot_hand_pose[4], robot_hand_pose[5]])
            recorder.collect_human_hand_position(human_hand_position=human_hand_position)
            recorder.collect_robot_data(joint_values=performer_data)
            recorder.collect_robot_hand_position(robot_hand_position=robot_hand_position)
            recorder.collect_robot_hand_ori(robot_hand_ori=robot_hand_ori)

            # hand_position = agent.get_hand_position(role='performer')
            # print("hand position: x: {}, y: {}, z: {}".format(hand_position[0], hand_position[1], hand_position[2]))

            data = agent.collect_robot_data(role='puppeteer')
            agent.take_action(action=data, role='performer')

            print("[Normal Mode]: Step {} finished".format(current_step + 1))
            current_step += 1

            # sleep(step_delta_t)
            sleep(0.01)

    agent.say("Okay, we finished the first greeting", role='performer')
    sleep(1.0)
    agent.set_stiffness(1.0, role='puppeteer')
    agent.set_stiffness(1.0, role='performer')
    print("Finished set stiffness to 1.0")
    agent.say("Wait a second, I need to save the data.", role='performer')
    recorder.save_demo_data()
    print("[Normal Mode]: Demo data saved")
    agent.go_home_pose(role='performer')
    agent.go_home_pose(role='puppeteer')

    # sleep(1.0)

    print('#########################################')

    """ Start "Diversion Mode" Experiment """
    agent.say("Okay, data saved! Let's continue!", role='performer')
    sleep(1.0)
    agent.say("Please do greeting with me again", role='performer')
    sleep(1.0)
    agent.say("Ready?", role='performer')
    sleep(1.0)
    agent.say("Start!", role='performer')

    agent.set_stiffness(0.0, role='puppeteer')
    agent.set_stiffness(1.0, role='performer')

    current_step = 0
    robot_arm_depth_last = None
    robot_arm_depth = None
    arm_depth_init_thres = 0.1
    back_steps_total = 0
    back_steps_thres = 5
    whether_stop_ori_vary = False
    recorder.reset()

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
            performer_data = agent.collect_robot_data(role='performer')
            robot_hand_pose = agent.get_hand_pose(role='performer')
            robot_hand_position = np.array([robot_hand_pose[0], robot_hand_pose[1], robot_hand_pose[2]])
            robot_hand_ori = np.array([robot_hand_pose[3], robot_hand_pose[4], robot_hand_pose[5]])
            recorder.collect_human_hand_position(human_hand_position=human_hand_position)
            recorder.collect_robot_data(joint_values=performer_data)
            recorder.collect_robot_hand_position(robot_hand_position=robot_hand_position)
            recorder.collect_robot_hand_ori(robot_hand_ori=robot_hand_ori)

            robot_arm_depth = robot_hand_position[0]
            if not whether_stop_ori_vary:
                # for initiation
                if robot_arm_depth_last is None:
                    robot_arm_depth_last = robot_arm_depth
                # check whether robot arm is withdrawing, with x direction pointing from robot to human
                if robot_arm_depth - robot_arm_depth_last < 0:
                    back_steps_total += 1
                # stop vary arm velocity if robot is withdrawing hands, instead of reaching hands
                if back_steps_total >= back_steps_thres:
                    whether_stop_ori_vary = True
                # update arm depth
                robot_arm_depth_last = robot_arm_depth

            # allow whole body movement of puppeteer
            data = agent.collect_robot_data(role='puppeteer')
            agent.take_action(action=data, role='performer')
            if not whether_stop_ori_vary and robot_arm_depth >= arm_depth_init_thres:
                agent.take_action_wrist(action=90.0/180.0*np.pi, role='performer')
            else:
                agent.take_action_wrist(action=0.0, role='performer')

            print("[Diversion Mode]: Step {} finished".format(current_step + 1))
            current_step += 1

            # sleep(step_delta_t)
            sleep(0.01)

    agent.set_stiffness(1.0, role='puppeteer')
    agent.set_stiffness(1.0, role='performer')
    print("Finished set stiffness to 1.0")

    agent.say("Great! We finished this section of experiment!", role='performer')
    sleep(1.0)
    # agent.set_stiffness(1.0, role='puppeteer')
    # agent.set_stiffness(1.0, role='performer')
    # print("Finished set stiffness to 1.0")
    agent.say("Please take a rest and fill out the questionnaire", role='performer')
    recorder.save_demo_data()
    print("[Diversion Mode]: Demo data saved")
    agent.go_home_pose(role='performer')
    agent.go_home_pose(role='puppeteer')

    # sleep(1.0)


def vary_withdraw_delay_old(agent, recorder, episode_total_steps, cam, step_delta_t, section_id=6):
    current_step = 0
    recorder.reset()

    agent.say("We are going to start a new section of experiments", role='performer')
    sleep(1.0)
    agent.say("Section {}".format(section_id), role='performer')
    sleep(0.5)

    """ Start "Normal Mode" experiment """
    agent.say("Please do greeting with me for the first time", role='performer')
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
            performer_data = agent.collect_robot_data(role='performer')
            robot_hand_pose = agent.get_hand_pose(role='performer')
            robot_hand_position = np.array([robot_hand_pose[0], robot_hand_pose[1], robot_hand_pose[2]])
            robot_hand_ori = np.array([robot_hand_pose[3], robot_hand_pose[4], robot_hand_pose[5]])
            recorder.collect_human_hand_position(human_hand_position=human_hand_position)
            recorder.collect_robot_data(joint_values=performer_data)
            recorder.collect_robot_hand_position(robot_hand_position=robot_hand_position)
            recorder.collect_robot_hand_ori(robot_hand_ori=robot_hand_ori)

            # hand_position = agent.get_hand_position(role='performer')
            # print("hand position: x: {}, y: {}, z: {}".format(hand_position[0], hand_position[1], hand_position[2]))

            data = agent.collect_robot_data(role='puppeteer')
            agent.take_action(action=data, role='performer')

            print("[Normal Mode]: Step {} finished".format(current_step + 1))
            current_step += 1

            # sleep(step_delta_t)
            sleep(0.01)

    agent.say("Okay, we finished the first greeting", role='performer')
    sleep(1.0)
    agent.set_stiffness(1.0, role='puppeteer')
    agent.set_stiffness(1.0, role='performer')
    print("Finished set stiffness to 1.0")
    agent.say("Wait a second, I need to save the data.", role='performer')
    recorder.save_demo_data()
    print("[Normal Mode]: Demo data saved")
    agent.go_home_pose(role='performer')
    agent.go_home_pose(role='puppeteer')

    # sleep(1.0)

    print('#########################################')

    """ Start "Diversion Mode" Experiment """
    agent.say("Okay, data saved! Let's continue!", role='performer')
    sleep(1.0)
    agent.say("Please do greeting with me again", role='performer')
    sleep(1.0)
    agent.say("Ready?", role='performer')
    sleep(1.0)
    agent.say("Start!", role='performer')

    agent.set_stiffness(0.0, role='puppeteer')
    agent.set_stiffness(1.0, role='performer')

    current_step = 0
    old_data = None
    delay_started = False
    delay_finished = False
    delay_steps = 0
    still_steps = 0
    delay_thres = 10  # total number of steps for reaching delay
    still_steps_thres = 3
    delay_init_thres = 0.02  # joint value threshold to trigger the delay
    depth_thred = 0.2
    recorder.reset()

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
            performer_data = agent.collect_robot_data(role='performer')
            robot_hand_pose = agent.get_hand_pose(role='performer')
            robot_hand_position = np.array([robot_hand_pose[0], robot_hand_pose[1], robot_hand_pose[2]])
            robot_hand_ori = np.array([robot_hand_pose[3], robot_hand_pose[4], robot_hand_pose[5]])
            recorder.collect_human_hand_position(human_hand_position=human_hand_position)
            recorder.collect_robot_data(joint_values=performer_data)
            recorder.collect_robot_hand_position(robot_hand_position=robot_hand_position)
            recorder.collect_robot_hand_ori(robot_hand_ori=robot_hand_ori)

            robot_arm_depth = robot_hand_position[0]

            # add time delay to response
            data = robot_arm_depth
            if not delay_started:
                if old_data is None:
                    old_data = data

                if robot_arm_depth >= depth_thred and abs(data - old_data) <= delay_init_thres:
                    still_steps += 1
                    if still_steps >= still_steps_thres:
                        delay_steps += 1
                        delay_started = True
                        agent.set_stiffness(1.0, role='puppeteer')
                        agent.set_led(1.0, role='puppeteer', led_name='ear')  # turn on right ear led
                        print("[Diversion Mode]: Delay started")
                    else:
                        old_data = data
                else:
                    old_data = data
            else:
                if not delay_finished:
                    if delay_steps < delay_thres:
                        delay_steps += 1
                        print("[Diversion Mode]: Delay step {}".format(delay_steps))
                    else:
                        agent.set_stiffness(0.0, role='puppeteer')
                        agent.set_led(0.0, role='puppeteer', led_name='ear')  # turn off right ear led
                        print("[Diversion Mode]: Delay finished")
                        delay_finished = True
                else:
                    pass

            # allow whole body movement of puppeteer
            data = agent.collect_robot_data(role='puppeteer')
            agent.take_action(action=data, role='performer')

            print("[Diversion Mode]: Step {} finished".format(current_step + 1))
            current_step += 1

            # sleep(step_delta_t)
            sleep(0.01)

    agent.set_stiffness(1.0, role='puppeteer')
    agent.set_stiffness(1.0, role='performer')
    print("Finished set stiffness to 1.0")

    agent.say("Great! We finished this section of experiment!", role='performer')
    sleep(1.0)
    # agent.set_stiffness(1.0, role='puppeteer')
    # agent.set_stiffness(1.0, role='performer')
    # print("Finished set stiffness to 1.0")
    agent.say("Please take a rest and fill out the questionnaire", role='performer')
    recorder.save_demo_data()
    print("[Diversion Mode]: Demo data saved")
    agent.go_home_pose(role='performer')
    agent.go_home_pose(role='puppeteer')

    # sleep(1.0)


def main():
    # prepare episode-related parameters
    step_delta_t = 0.05
    episode_time_length = 3.0
    episode_total_steps = int(episode_time_length / step_delta_t)
    total_fac_num = 1

    # prepare camera-related variables
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    # get human subject id for data saving
    args = argparser() # --sub_id
    experiment_data_dir = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/experiment_data/robot_factor_exp/'
    factor_orders = np.genfromtxt(experiment_data_dir + 'factor_orders_list.csv').astype(int) # 2d np array in the shape of (total_subject_num, total_factor_num)
    mode_orders = np.genfromtxt(experiment_data_dir + 'mode_orders_list.csv').astype(int) # 2d np array in the shape of (total_subject_num, total_factor_num * total_mode_num)
    mode_orders = np.reshape(mode_orders, newshape=[20, -1, 2]) # 3d np array in the shape of (total_subject_num, total_factor_num, total_mode_num)

    """ Start Experiments """
    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        agent = Agent(puppeteer_mode=True)
        recorder = demo_data_recorder(pose, mp_pose=mp_pose, mp_drawing=mp_drawing, mp_drawing_styles=mp_drawing_styles, subject_id=args.sub_id)
        # recorder.reset_data_saving_path(experiment_data_dir)

        cam_port = 0
        cam = cv2.VideoCapture(cam_port)
        print("[Connector]: Succeeded to connect to camera")

        """ Start experiments """
        subject_id = args.sub_id
        factor_order = factor_orders[subject_id - 1]
        # factor_order = list(range(1))
        # factor_order = [5]
        mode_order = mode_orders[subject_id - 1]
        error_section_id = 0
        # mode_order = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]

        agent.set_stiffness_whole_body(0.0, role='puppeteer')

        for section_num in range(len(factor_order)):
            if (section_num + 1) < error_section_id:
                continue

            factor_id = factor_order[section_num]
            # factor_id = 5
            recorder.reset_data_saving_path(experiment_data_dir, factor_id=factor_id)
            first_mode_id = mode_order[factor_id][0]
            second_mode_id = mode_order[factor_id][1]
            print("[Section {} Experiment]: Start".format(section_num + 1))
            print("factor id: {}, first mode id: {}, second mode id: {}".format(factor_id, first_mode_id, second_mode_id))

            """ START FIRST SECTION """

            if first_mode_id == 1:
                # agent.set_led(1.0, role='puppeteer', led_name='shoulder')
                agent.set_led(1.0, role='puppeteer', led_name='ear_back')
                """if factor_id == 3:
                    agent.set_stiffness_whole_body(0.0, role='puppeteer')"""

            agent.say("We are going to start a new section of experiments", role='performer')
            sleep(1.0)
            agent.say("Section {}".format(section_num + 1), role='performer')
            sleep(0.5)
            agent.say("Please do greeting with me for the first time", role='performer')
            sleep(1.0)
            agent.say("Ready?", role='performer')
            sleep(1.0)
            agent.say("Start!", role='performer')

            if first_mode_id == 0:
                start_normal_mode(agent=agent, recorder=recorder, episode_total_steps=episode_total_steps, cam=cam, step_delta_t=step_delta_t, section_id=factor_id)
            else:
                # agent.set_led(1.0, role='puppeteer', led_name='shoulder')
                start_diversion_mode(agent=agent, recorder=recorder, episode_total_steps=episode_total_steps, cam=cam, step_delta_t=step_delta_t, section_id=factor_id)
                # agent.set_led(0.0, role='puppeteer', led_name='shoulder')
                agent.set_led(0.0, role='puppeteer', led_name='ear_back')

            agent.say("Great, we finished the first greeting", role='performer')
            sleep(1.0)

            # for hip movement experiment, it needs more time to set whole body stiffness
            if factor_id == 3:
                agent.say("Take a breath!", role='performer')
                sleep(0.5)
                agent.say("I need some time to save the data", role='performer')
                recorder.save_demo_data()
                print("[First Mode]: Demo data saved")

                """if first_mode_id == 1:
                    agent.set_stiffness_whole_body(1.0, role='puppeteer')
                    print("Finished set puppeteer stiffness to 1.0")"""
            else:
                agent.say("Take a rest! I need to save the data", role='performer')
                recorder.save_demo_data()
                print("[First Mode]: Demo data saved")

            agent.go_home_pose(role='performer')
            """agent.go_home_pose(role='puppeteer')"""

            print('#########################################')

            """ START SECOND SECTION """
            if second_mode_id == 1:
                # agent.set_led(1.0, role='puppeteer', led_name='shoulder')
                agent.set_led(1.0, role='puppeteer', led_name='ear_back')
                """if factor_id == 3:
                    agent.set_stiffness_whole_body(0.0, role='puppeteer')"""


            agent.say("Okay, Data saved! Let's continue!", role='performer')
            sleep(1.0)
            agent.say("Please do greeting with me again", role='performer')
            sleep(1.0)
            agent.say("Ready?", role='performer')
            sleep(1.0)
            agent.say("Start!", role='performer')

            if second_mode_id == 0:
                start_normal_mode(agent=agent, recorder=recorder, episode_total_steps=episode_total_steps, cam=cam,
                                  step_delta_t=step_delta_t, section_id=factor_id)
            else:
                start_diversion_mode(agent=agent, recorder=recorder, episode_total_steps=episode_total_steps, cam=cam,
                                     step_delta_t=step_delta_t, section_id=factor_id)
                # agent.set_led(0.0, role='puppeteer', led_name='shoulder')
                agent.set_led(0.0, role='puppeteer', led_name='ear_back')

            agent.say("Great! We finished this section of experiment!", role='performer')
            sleep(1.0)

            agent.say("Please take a rest and fill out the questionnaire", role='performer')
            recorder.save_demo_data()
            print("[Second Mode]: Demo data saved")

            agent.go_home_pose(role='performer')

            """if factor_id == 3 and second_mode_id == 1:
                agent.set_stiffness_whole_body(1.0, role='puppeteer')
                print("Finished set puppeteer stiffness to 1.0")"""

            """agent.go_home_pose(role='puppeteer')"""

            print("[Section {} Experiment]: Finished".format(section_num + 1))
            print('*****************************')
            print('******************************')


            input("Press Enter to continue next section ...")

        # """ Start experiments """
        # for factor_id in range(total_fac_num):
        #     print('[Section {}]: Going to start ...'.format(factor_id + 1))
        #     if factor_id == 5:
        #         # start experiment section for hand-reaching delay
        #         print("[Reaching Delay Experiment]: Start")
        #         vary_reaching_delay(agent=agent, recorder=recorder, episode_total_steps=episode_total_steps, cam=cam, step_delta_t=step_delta_t, section_id=factor_id+1)
        #         print('[Reaching Delay Experiment]: Finished')
        #     elif factor_id == 1:
        #         # start experiment section for robot arm velocity
        #         print("[Arm Velocity Experiment]: Start")
        #         vary_arm_velocity(agent=agent, recorder=recorder, episode_total_steps=episode_total_steps, cam=cam, step_delta_t=step_delta_t, section_id=factor_id+1)
        #         print('[Arm Velocity Experiment]: Finished')
        #     elif factor_id == 2:
        #         # start experiment section for robot hand offset
        #         print("[Hand Offset Experiment]: Start")
        #         vary_hand_position(agent=agent, recorder=recorder, episode_total_steps=episode_total_steps, cam=cam, step_delta_t=step_delta_t, section_id=factor_id+1)
        #         print('[Hand Offset Experiment]: Finished')
        #     elif factor_id == 3:
        #         # start experiment section for robot hand offset
        #         print("[Whole Body Experiment]: Start")
        #         vary_hip_movement(agent=agent, recorder=recorder, episode_total_steps=episode_total_steps, cam=cam, step_delta_t=step_delta_t, section_id=factor_id+1)
        #         print('[Whole Body Experiment]: Finished')
        #     elif factor_id == 4:
        #         # start experiment section for robot hand offset
        #         print("[Wrist Orientation Experiment]: Start")
        #         vary_wrist_orientation(agent=agent, recorder=recorder, episode_total_steps=episode_total_steps, cam=cam, step_delta_t=step_delta_t, section_id=factor_id+1)
        #         print('[Wrist Orientation Experiment]: Finished')
        #     elif factor_id == 0:
        #         # start experiment section for robot hand offset
        #         print("[Withdraw Delay Experiment]: Start")
        #         vary_withdraw_delay(agent=agent, recorder=recorder, episode_total_steps=episode_total_steps, cam=cam, step_delta_t=step_delta_t, section_id=factor_id+1)
        #         print('[Withdraw Delay Experiment]: Finished')
        #
        #     print("[Section {}]: Finished".format(factor_id + 1))
        #     print('*****************************')
        #     print('******************************')

    agent.say("Fabulous! This part of experiment is finished!", role='performer')
    sleep(0.5)
    agent.say("Please take a rest. Then we are going to conduct next part of experiments.", role='performer')
    agent.stop()
    cam.release()
    print("Robot-factor experiments finished")


if __name__ == '__main__':
    main()