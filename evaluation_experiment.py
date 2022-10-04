import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from time import sleep
import os

from policy_net import Policy_net
from environment import GreetingEnv
from agent import Agent
from run_test import prepare_state_seq, prepare_action
from robot_factor_experiment_wizard import get_human_hand_position
from run_behavior_clone import save_demo

from simulation_robot_client import simulation_robot_client
from run_behavior_clone import prepare_expert_data_seq, load_multiple_subject_data
from run_test import argparser


def run_bc_baseline(sess, args, saver, data_collection_mode, agent, env, Policy, episode_total_steps,
                    state_max, state_min, state_with_robot_state, action_max, action_min, action_with_robot_vel, subject_id):
    human_pose_data = [np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])]  # obs of expert demonstration
    human_hand_position_data = [np.array([np.inf, np.inf, np.inf])]  # trajectory of 3d position of human hand
    robot_action_data = [np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])]  # act of expert demonstration
    robot_hand_position_data = [np.array([np.inf, np.inf, np.inf])]  # trajectory of 3d position of robot hand
    robot_hand_orientation_data = [np.array([np.inf, np.inf, np.inf])]  # trajectory of orientation of robot hand

    evaluation_data_dir = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/experiment_data/evaluation_exp/'
    human_pose_data_dir = evaluation_data_dir + 'subject_' + str(subject_id) + '/' + data_collection_mode + '/expert_observations.csv'
    human_hand_position_data_dir = evaluation_data_dir + 'subject_' + str(subject_id) + '/' + data_collection_mode + '/human_hand_position_trajectories.csv'
    robot_action_data_dir = evaluation_data_dir + 'subject_' + str(subject_id) + '/' + data_collection_mode + '/expert_actions.csv'
    robot_hand_position_data_dir = evaluation_data_dir + 'subject_' + str(subject_id) + '/' + data_collection_mode + '/robot_hand_position_trajectories.csv'
    robot_hand_orientation_data_dir = evaluation_data_dir + 'subject_' + str(subject_id) + '/' + data_collection_mode + '/robot_hand_ori_trajectories.csv'
    new_path = evaluation_data_dir + 'subject_' + str(subject_id) + '/' + data_collection_mode
    if not os.path.exists(new_path):
        os.makedirs(new_path)
        print("Created a new saving path for subject {}".format(subject_id))

    sess.run(tf.global_variables_initializer())

    # load the trained model
    model_dir = 'pepper/bc/experiments/' + data_collection_mode + '/'
    if args.model == '':
        saver.restore(sess, args.modeldir + '/' + args.alg + '/' + model_dir + 'model.ckpt')
    else:
        saver.restore(sess, args.modeldir + '/' + args.alg + '/' + model_dir + 'model.ckpt-' + args.model)

    current_step = 0
    latest_obs_seq = []

    agent.say("Going to start new trial.", role='performer')
    sleep(1.0)
    agent.say("Please prepare to do greeting with me", role='performer')
    sleep(1.0)
    agent.say("Ready?", role='performer')
    sleep(1.0)
    agent.say("Start!", role='performer')

    obs = env.reset()
    last_obs = obs

    while current_step < episode_total_steps:
        if obs is None:
            agent.say("Oh no, I can't see you", role='performer')
            print("Step {}: No human detected or fail to connect to the camera".format(current_step + 1))
            obs = env.reset()
            last_obs = obs
            continue

        human_pose = obs
        human_hand_position = get_human_hand_position(hand_x_in_hip_frame=human_pose[0],
                                                      hand_y_in_hip_frame=human_pose[1],
                                                      hand_z_in_hip_frame=human_pose[2])
        performer_data = agent.collect_robot_data_whole_body(role='performer')
        robot_hand_pose = agent.get_hand_pose(role='performer')
        robot_hand_position = np.array([robot_hand_pose[0], robot_hand_pose[1], robot_hand_pose[2]])
        robot_hand_ori = np.array([robot_hand_pose[3], robot_hand_pose[4], robot_hand_pose[5]])

        human_pose_data.append(human_pose)
        human_hand_position_data.append(human_hand_position)
        robot_action_data.append(performer_data)
        robot_hand_position_data.append(robot_hand_position)
        robot_hand_orientation_data.append(robot_hand_ori)


        # reshape the obs for Policy input placeholder
        # reshaped_obs would be 2d np.array of shape (batch_size, seq_length * state_dimension)
        # latest_obs_seq would a list containing 1d arrays of shape (state_dimension,)
        state, latest_obs_seq = prepare_state_seq(current_human_state=obs, last_human_state=last_obs,
                                                  current_robot_state=None,
                                                  obs_seq=latest_obs_seq, state_max=state_max, state_min=state_min,
                                                  include_robot_state=state_with_robot_state)

        act, _ = Policy.act(obs=state, stochastic=False)  # in the form of normalized angular velocity of each joint
        reshaped_act = prepare_action(current_robot_state=None, normalized_action=act,
                                      action_max=action_max, action_min=action_min,
                                      use_robot_vel=action_with_robot_vel)  # in the form of target joint value
        agent.take_action(reshaped_act, role='performer')

        # sleep(step_delta_t)
        # sleep(0.01)
        print("Step {}: Finish taking action".format(current_step + 1))
        # sleep(0.05)

        # update the state for next step
        next_obs, _, _, _ = env.step(act)
        last_obs = obs
        obs = next_obs
        current_step += 1

    agent.say("Great! We finished this trial.", role='performer')
    sleep(1.0)
    agent.say("Take a rest, I need to save the data.", role='performer')
    agent.go_home_pose(role='performer')
    save_demo(demo_data=human_pose_data, demo_path=human_pose_data_dir)
    save_demo(demo_data=human_hand_position_data, demo_path=human_hand_position_data_dir)
    save_demo(demo_data=robot_action_data, demo_path=robot_action_data_dir)
    save_demo(demo_data=robot_hand_position_data, demo_path=robot_hand_position_data_dir)
    save_demo(demo_data=robot_hand_orientation_data, demo_path=robot_hand_orientation_data_dir)
    print("Data saved")

    print("Finished go home pose")


def run_bc_baseline_simulation(sess, args, saver, data_collection_mode, agent, env, Policy, episode_total_steps,
                    state_max, state_min, state_with_robot_state, action_max, action_min, action_with_robot_vel, subject_id):
    subject_ids = [1, 2]
    testing_episodes_id = [0]
    seq_length = 10
    delta_t = 0.05
    num_states = Policy.ob_space.shape[0]
    expert_data_dir = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/experiment_data/human_feature_exp/' + 'guided_collect' + '/subject_'
    expert_observations, expert_actions = load_multiple_subject_data(sub_ids=subject_ids,
                                                                     act_shape=(6,),
                                                                     demo_data_dir=expert_data_dir)
    _, _, testing_expert_obs, _ = prepare_expert_data_seq(expert_observations, expert_actions,
                                                          state_max, state_min, action_max, action_min,
                                                          num_states, testing_episodes_id,
                                                          state_with_robot_state, action_with_robot_vel,
                                                          seq_length, delta_t)

    sess.run(tf.global_variables_initializer())

    # load the trained model
    model_dir = 'pepper/bc/experiments/' + data_collection_mode + '/'
    if args.model == '':
        saver.restore(sess, args.modeldir + '/' + args.alg + '/' + model_dir + 'model.ckpt')
    else:
        saver.restore(sess, args.modeldir + '/' + args.alg + '/' + model_dir + 'model.ckpt-' + args.model)

    current_step = 0

    while current_step < episode_total_steps:


        state = testing_expert_obs[current_step]
        state = np.reshape(state, newshape=[-1, seq_length * num_states])

        act, _ = Policy.act(obs=state, stochastic=False)  # in the form of normalized angular velocity of each joint
        reshaped_act = prepare_action(current_robot_state=None, normalized_action=act,
                                      action_max=action_max, action_min=action_min,
                                      use_robot_vel=action_with_robot_vel)  # in the form of target joint value
        agent.take_action(reshaped_act)

        # sleep(step_delta_t)
        # sleep(0.01)
        print("[Step {}]: Finish taking action".format(current_step + 1))
        print("action: {}".format(reshaped_act))
        print("state: {}".format(state[0, (seq_length -1) * num_states : seq_length * num_states]))

        sleep(0.05)

        current_step += 1


    sleep(2.0)

    print("Finished go home pose")


def run_wizard_of_oz(env, agent, episode_total_steps, subject_id):
    human_pose_data = [np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])] # obs of expert demonstration
    human_hand_position_data = [np.array([np.inf, np.inf, np.inf])] # trajectory of 3d position of human hand
    robot_action_data = [np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])] # act of expert demonstration
    robot_hand_position_data = [np.array([np.inf, np.inf, np.inf])] # trajectory of 3d position of robot hand
    robot_hand_orientation_data = [np.array([np.inf, np.inf, np.inf])] # trajectory of orientation of robot hand

    evaluation_data_dir = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/experiment_data/evaluation_exp/'
    human_pose_data_dir = evaluation_data_dir + 'subject_' + str(subject_id) + '/wizard_of_oz' + '/expert_observations.csv'
    human_hand_position_data_dir = evaluation_data_dir + 'subject_' + str(subject_id) + '/wizard_of_oz' + '/human_hand_position_trajectories.csv'
    robot_action_data_dir = evaluation_data_dir + 'subject_' + str(subject_id) + '/wizard_of_oz' + '/expert_actions.csv'
    robot_hand_position_data_dir = evaluation_data_dir + 'subject_' + str(subject_id) + '/wizard_of_oz' + '/robot_hand_position_trajectories.csv'
    robot_hand_orientation_data_dir = evaluation_data_dir + 'subject_' + str(subject_id) + '/wizard_of_oz' + '/robot_hand_ori_trajectories.csv'
    new_path = evaluation_data_dir + 'subject_' + str(subject_id) + '/' + 'wizard_of_oz'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
        print("Created a new saving path for subject {}".format(subject_id))

    # agent.set_stiffness(1.0, role='performer')

    agent.say("Going to start new trial.", role='performer')
    sleep(1.0)
    agent.say("Please prepare to do greeting with me", role='performer')
    sleep(1.0)
    agent.say("Ready?", role='performer')
    sleep(1.0)
    agent.say("Start!", role='performer')

    obs = env.reset()
    current_step = 0

    while current_step < episode_total_steps:
        if obs is None:
            agent.say("Oh no, I can't see you", role='performer')
            print("Step {}: No human detected or fail to connect to the camera".format(current_step + 1))
            obs = env.reset()
            continue

        human_pose = obs
        human_hand_position = get_human_hand_position(hand_x_in_hip_frame=human_pose[0],
                                                      hand_y_in_hip_frame=human_pose[1],
                                                      hand_z_in_hip_frame=human_pose[2])
        performer_data = agent.collect_robot_data_whole_body(role='performer')
        robot_hand_pose = agent.get_hand_pose(role='performer')
        robot_hand_position = np.array([robot_hand_pose[0], robot_hand_pose[1], robot_hand_pose[2]])
        robot_hand_ori = np.array([robot_hand_pose[3], robot_hand_pose[4], robot_hand_pose[5]])

        human_pose_data.append(human_pose)
        human_hand_position_data.append(human_hand_position)
        robot_action_data.append(performer_data)
        robot_hand_position_data.append(robot_hand_position)
        robot_hand_orientation_data.append(robot_hand_ori)

        data = agent.collect_robot_data_whole_body(role='puppeteer')
        agent.take_action(action=data, role='performer')

        print("Step {} finished".format(current_step + 1))

        next_obs, _, _, _ = env.step(data)
        obs = next_obs
        current_step += 1

        # sleep(step_delta_t)
        sleep(0.01)

    # agent.set_stiffness(1.0, role='performer')
    # print("Finished set stiffness to 1.0")

    agent.say("Great! We finished this trial.", role='performer')
    sleep(1.0)
    agent.say("Take a rest, I need to save the data.", role='performer')
    agent.go_home_pose(role='performer')
    save_demo(demo_data=human_pose_data, demo_path=human_pose_data_dir)
    save_demo(demo_data=human_hand_position_data, demo_path=human_hand_position_data_dir)
    save_demo(demo_data=robot_action_data, demo_path=robot_action_data_dir)
    save_demo(demo_data=robot_hand_position_data, demo_path=robot_hand_position_data_dir)
    save_demo(demo_data=robot_hand_orientation_data, demo_path=robot_hand_orientation_data_dir)
    print("Data saved")

    print("Finished go home pose")


def main(args):
    evaluation_experiment_dir = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/experiment_data/evaluation_exp/'
    baseline_orders = np.genfromtxt(evaluation_experiment_dir + 'baseline_orders_list.csv').astype(int)  # 2d np array in the shape of (total_subject_num, total_baseline_num)
    subject_id = args.sub_id # index from 1 to 20
    baseline_order = baseline_orders[subject_id - 1]
    # baseline_order = [0]

    using_vae = False
    bc_using_lstm = False
    Policy = Policy_net('policy', using_latent_state=using_vae, using_lstm=bc_using_lstm)

    # parameters related to episode length
    step_delta_t = 0.05
    episode_time_length = 3.0
    episode_total_steps = int(episode_time_length / step_delta_t)

    joint_values_max = np.array([2.0857, -0.0087, 2.0857, 1.5620, 1.8239, 0.5149])
    joint_values_min = np.array([-2.0857, -1.5620, -2.0857, 0.0087, -1.8239, -0.5149])
    delta_t = 0.05
    state_with_robot_state = False
    action_with_robot_vel = False

    if not action_with_robot_vel:
        action_max = joint_values_max
        action_min = joint_values_min
    else:
        action_max = (joint_values_max - joint_values_min) / delta_t
        action_min = (joint_values_min - joint_values_max) / delta_t

    state_max = Policy.ob_space.high
    state_min = Policy.ob_space.low

    saver = tf.train.Saver()
    env = GreetingEnv()
    agent = Agent(puppeteer_mode=True)
    # agent = simulation_robot_client(using_gui=True, auto_step=True)
    agent.set_stiffness(1.0, role='performer')
    agent.set_stiffness_whole_body(0.0, role='puppeteer')
    print("[Robot Client]: Finish setting stiffness to 1.0")

    with tf.Session() as sess:
        for trial in range(len(baseline_order)):
            baseline_id = baseline_order[trial]

            if baseline_id == 0:
                print("[Baseline id: 0]: Going to run wizard-of-oz")
                agent.set_led(1.0, role='puppeteer', led_name='ear_back')
                run_wizard_of_oz(env=env, agent=agent, episode_total_steps=episode_total_steps, subject_id=subject_id)
                agent.set_led(0.0, role='puppeteer', led_name='ear_back')
            elif baseline_id == 1:
                data_collection_mode = 'random_collect'
                print("[Baseline id: 1]: Going to run bc model with random collection")
                run_bc_baseline(sess=sess, args=args, saver=saver, data_collection_mode=data_collection_mode,
                                agent=agent, env=env, Policy=Policy, episode_total_steps=episode_total_steps,
                                state_max=state_max, state_min=state_min, state_with_robot_state=state_with_robot_state,
                                action_max=action_max, action_min=action_min, action_with_robot_vel=action_with_robot_vel,
                                subject_id=subject_id)
            else:
                data_collection_mode = 'guided_collect'
                print("[Baseline id: 2]: Going to run bc model with guided collection")
                run_bc_baseline(sess=sess, args=args, saver=saver, data_collection_mode=data_collection_mode,
                                agent=agent, env=env, Policy=Policy, episode_total_steps=episode_total_steps,
                                state_max=state_max, state_min=state_min, state_with_robot_state=state_with_robot_state,
                                action_max=action_max, action_min=action_min, action_with_robot_vel=action_with_robot_vel,
                                subject_id=subject_id)

            print("[Trial {}]: Finished".format(trial + 1))
            print("********************************************")
            if trial < len(baseline_order) - 1:
                input("Press Enter to continue next section ...")

        agent.say("Great! All trials are finished! Thank you so much for your patience!", role='performer')
        agent.stop()
        print("All episodes finished")


if __name__ == '__main__':
    args = argparser()
    main(args)