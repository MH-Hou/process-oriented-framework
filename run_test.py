import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()   #禁止TensorFlow2默认的即时执行模式
from time import sleep

from policy_net import Policy_net
from environment import GreetingEnv
from agent import Agent

from sklearn.preprocessing import MinMaxScaler

from run_behavior_clone import prepare_expert_data_seq, load_multiple_subject_data
from vae_tf1 import Vae


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='directory of model', default='trained_models')
    parser.add_argument('--alg', help='chose algorithm one of gail, ppo, bc', default='bc')
    parser.add_argument('--model', help='number of model to test. model.ckpt-number', default='1000')
    parser.add_argument('--logdir', help='log directory', default='log/test')
    parser.add_argument('--iteration', default=int(6))
    parser.add_argument('--stochastic', action='store_false')
    parser.add_argument('--sub_id', help='id of human subject', default=0, type=int)
    return parser.parse_args()


def reshape_obs(original_obs, obs_seq, seq_length=10):
    num_states = original_obs.shape[0]

    if not obs_seq:
        latest_obs_seq = []
        for i in range(seq_length):
            obs_copy = original_obs.copy()
            latest_obs_seq.append(obs_copy)
    else:
        latest_obs_seq = obs_seq.copy()
        latest_obs_seq.pop(0)
        obs_copy = original_obs.copy()
        latest_obs_seq.append(obs_copy)

    # latest_obs_seq would be a list of 1d np.array with shape (state_dimension,)
    # reshaped_obs would be reshaped np.array of latest_obs_seq
    reshaped_obs = np.array(latest_obs_seq).astype(dtype=np.float32) # in the shape of (seq_length, state_dimension)
    # in the form of (batch_size, seq_length * state_dimension)
    reshaped_obs = np.reshape(reshaped_obs, newshape=[-1, seq_length * num_states])

    scaler = MinMaxScaler()
    scaled_obs = scaler.fit_transform(np.transpose(reshaped_obs))
    scaled_obs = np.transpose(scaled_obs)

    return scaled_obs, latest_obs_seq

# get sequence of normalized states including human state, human velocity, and(or) robot state
def prepare_state_seq(current_human_state, last_human_state, current_robot_state, obs_seq,
                      state_max, state_min, include_robot_state=False, seq_length=10, delta_t=0.05):
    current_human_vel = (current_human_state - last_human_state) / delta_t
    if not include_robot_state:
        state = np.concatenate([current_human_state, current_human_vel])
    else:
        state = np.concatenate([current_human_state, current_human_vel, current_robot_state])

    state = (state - state_min) / (state_max - state_min)
    num_states = state.shape[0]

    # construct sequential states
    if not obs_seq:
        latest_obs_seq = []
        for i in range(seq_length):
            obs_copy = state.copy()
            latest_obs_seq.append(obs_copy)
    else:
        latest_obs_seq = obs_seq.copy()
        latest_obs_seq.pop(0)
        obs_copy = state.copy()
        latest_obs_seq.append(obs_copy)

    # latest_obs_seq would be a list of 1d np.array with shape (state_dimension,)
    # reshaped_obs would be reshaped np.array of latest_obs_seq
    reshaped_obs = np.array(latest_obs_seq).astype(dtype=np.float32)  # in the shape of (seq_length, state_dimension)
    # in the form of (batch_size, seq_length * state_dimension)
    reshaped_obs = np.reshape(reshaped_obs, newshape=[-1, seq_length * num_states])

    return reshaped_obs, latest_obs_seq



# get the normalized state that includes both human state, human velocity, and robot state
def prepare_state(current_human_state, last_human_state, current_robot_state, state_max, state_min, include_robot_state=False, delta_t=0.05):
    current_human_vel = (current_human_state - last_human_state) / delta_t

    if not include_robot_state:
        state = np.concatenate([current_human_state, current_human_vel])
    else:
        state = np.concatenate([current_human_state, current_human_vel, current_robot_state])

    state = (state - state_min) / (state_max - state_min)
    state = np.reshape(state, newshape=[-1] + list(state.shape)) # reshape the state into (batch_size, state_dimension) for policy_net input

    # print("current human state:")
    # print(current_human_state)
    # print("current human vel:")
    # print(current_human_vel)
    # print("current robot state:")
    # print(current_robot_state)
    '''
    scaler = MinMaxScaler()
    state = scaler.fit_transform(np.transpose(state))
    state = np.transpose(state)
    '''

    return state


# turn the joint velocities into target joint values to send command to the real robot
def prepare_action(current_robot_state, normalized_action,
                   action_max,
                   action_min,
                   use_robot_vel=False,
                   delta_t=0.05):
    # turn the tensor form into np.array in the shape of (action_dimension,), which is in the range of [0, 1]
    normalized_action = np.array(normalized_action)

    # de-normalize to real action range
    if not use_robot_vel:
        # target_joint_values = normalized_action * (action_max - action_min) + action_min
        target_joint_values = normalized_action
    else:
        # real_joint_vel = normalized_action * (action_max - action_min) + action_min
        real_joint_vel = normalized_action
        target_joint_values = current_robot_state + real_joint_vel * delta_t

    # print("target joint values:")
    # print(target_joint_values)

    return target_joint_values

def save_demo(demo_data, demo_path):
    try:
        with open(demo_path, 'ab') as f_handle:
            np.savetxt(f_handle, demo_data, fmt='%s')
    except FileNotFoundError:
        with open(demo_path, 'wb') as f_handle:
            np.savetxt(f_handle, demo_data, fmt='%s')


def main(args):
    # subject_ids = [1]
    # training_subjects_num = len(subject_ids)
    # training_traj_num = 8
    # trial_id = 2

    using_vae = False
    vae_using_lstm = False
    bc_using_lstm = False

    data_collection_mode = 'guided_collect'

    if using_vae:
        vae = Vae('vae', latent_dim=7, using_lstm=vae_using_lstm)
        Policy = Policy_net('policy', using_latent_state=using_vae, latent_dim=vae.latent_dim)
        model_dir = 'pepper/vae_bc/experiments/' + data_collection_mode + '/'
        # model_dir = 'pepper/vae_bc/' + str(training_subjects_num) + '_subject/' + str(training_traj_num) + '_training_traj/'
        # model_dir = 'pepper/vae_bc/'
    else:
        Policy = Policy_net('policy', using_latent_state=using_vae, using_lstm=bc_using_lstm)
        model_dir = 'pepper/bc/experiments/' + data_collection_mode + '/'
        # model_dir = 'pepper/bc/' + str(training_subjects_num) + '_subject/' + str(training_traj_num) + '_training_traj/'
        # model_dir = 'pepper/bc/'

    saver = tf.train.Saver()
    env = GreetingEnv()
    agent = Agent()
    agent.set_stiffness(1.0)
    print("[Robot Client]: Finish setting stiffness to 1.0")


    """# latency test with given actions
    subject_ids = [5]
    expert_observations, expert_actions = load_multiple_subject_data(subject_ids)
    testing_episodes_id = [0]
    seq_length = 10
    num_states = Policy.ob_space.shape[0]"""


    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logdir + '/pepper/' + args.alg, sess.graph)
        sess.run(tf.global_variables_initializer())

        # load the trained model
        # model_dir = 'pepper/vae_bc/'
        # dir_note = 'side_view/include_back/new_state_design_seq/'
        if args.model == '':
            saver.restore(sess, args.modeldir+'/'+args.alg+'/'+ model_dir + 'model.ckpt')
        else:
            saver.restore(sess, args.modeldir+'/'+args.alg+'/'+ model_dir + 'model.ckpt-' + args.model)

        # parameters related to episode length
        step_delta_t = 0.05
        episode_time_length = 3.0
        episode_total_steps = int(episode_time_length / step_delta_t)

        joint_values_max = np.array([2.0857,  -0.0087, 2.0857, 1.5620, 1.8239, 0.5149])
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

        # for debug
        states_record = []
        actions_record = []
        demo_path_real_obs = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/demo_data/side_view/multi_pos/2/real_test_obs.csv'
        demo_path_real_act = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/demo_data/side_view/multi_pos/2/real_test_act.csv'

        """_, _, _, testing_expert_act = prepare_expert_data_seq(expert_observations, expert_actions,
                                                              state_max, state_min, action_max, action_min,
                                                              num_states, testing_episodes_id,
                                                              state_with_robot_state, action_with_robot_vel,
                                                              seq_length, delta_t)"""

        for iteration in range(args.iteration):
            current_step = 0
            latest_obs_seq = []
            agent.say("Going to start new episode " + str(iteration + 1))
            print("****************************************************")
            print("Episode {}: Going to start a new episode".format(iteration + 1))
            sleep(1.0)
            agent.say("Please prepare to do greeting with me")
            sleep(1.0)
            agent.say("Start")
            obs = env.reset()
            last_obs = obs
            states_record = [np.ones(Policy.ob_space.shape[0]) * np.inf]
            actions_record = [np.ones(Policy.act_space.shape[0]) * np.inf]

            while current_step < episode_total_steps:
                # if current_step == 0:
                #     obs = env.reset()  # 1d np.array in the form of (state_dimension,)

                if obs is None:
                    agent.say("Failed to detect human")
                    print("Episode {} step {}: No human detected or fail to connect to the camera".format(iteration + 1, current_step))
                    obs = env.reset()
                    last_obs = obs
                    continue

                # reshape the obs for Policy input placeholder
                # reshaped_obs would be 2d np.array of shape (batch_size, seq_length * state_dimension)
                # latest_obs_seq would a list containing 1d arrays of shape (state_dimension,)
                # reshaped_obs, latest_obs_seq = reshape_obs(obs, latest_obs_seq)
                # robot_state = agent.collect_robot_data() # in the form of 1d np.array for all the joint values
                # state = prepare_state(obs, last_obs, robot_state, state_max, state_min, state_with_robot_state)
                state, latest_obs_seq = prepare_state_seq(current_human_state=obs, last_human_state=last_obs, current_robot_state=None,
                                                          obs_seq=latest_obs_seq, state_max=state_max, state_min=state_min, include_robot_state=state_with_robot_state)
                # act = testing_expert_act[current_step]
                if using_vae:
                    latent_state = vae.get_latent_state(state)
                    act, _ = Policy.act(obs=latent_state, stochastic=False)
                else:
                    act, _ = Policy.act(obs=state, stochastic=False) # in the form of normalized angular velocity of each joint
                # act, _ = Policy.act(obs=reshaped_obs, stochastic=False)
                # reshaped_act = np.array(act) # in the shape of (action_dimension,)

                reshaped_act = prepare_action(current_robot_state=None, normalized_action=act,
                                              action_max=action_max, action_min=action_min, use_robot_vel=action_with_robot_vel) # in the form of target joint value
                agent.take_action(reshaped_act)
                # env.render()

                # sleep(step_delta_t)
                # sleep(0.01)
                print("Episode {} step {}: Finish taking action".format(iteration + 1, current_step))

                # states_record.append(np.reshape(state, newshape=list(Policy.ob_space.shape)))
                # actions_record.append(np.array(act))

                # print("################################")

                # update the state for next step
                next_obs, _, _, _ = env.step(act)
                last_obs = obs
                obs = next_obs
                current_step += 1

            agent.say("Finished episode " + str(iteration + 1))
            sleep(1.0)
            agent.say("I am going to home position")
            agent.go_home_pose()
            print("Episode {}: Episode finished".format(iteration + 1))

            # save_demo(states_record, demo_path_real_obs)
            # save_demo(actions_record, demo_path_real_act)



        agent.say("Great! Our testing is all finished! Thank you so much for your patience!")
        agent.stop()
        print("All episodes finished")



if __name__ == '__main__':
    args = argparser()
    main(args)






