import argparse
import gym
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()   #禁止TensorFlow2默认的即时执行模式
from policy_net import Policy_net
from behavior_clone import BehavioralCloning
from datetime import datetime
import os

from sklearn.preprocessing import MinMaxScaler
from vae_tf1 import Vae

from utils.utils import data_set_partition, save_data


TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())


def demo_data_partition(total_trial_num, trial_per_human_for_training=8, trial_per_human_for_testing=2, trial_per_human=10):
    training_trial_ids = []
    testing_trial_ids = []
    for i in range(total_trial_num):
        if i % trial_per_human > (trial_per_human_for_testing - 1) and \
                i % trial_per_human <= (trial_per_human_for_training + trial_per_human_for_testing - 1):
            training_trial_ids.append(i)

        if i % trial_per_human <= (trial_per_human_for_testing - 1):
            testing_trial_ids.append(i)

    return training_trial_ids, testing_trial_ids

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', help='name of directory to save model', default='trained_models/bc/')
    # parser.add_argument('--savedir', help='name of directory to save model', default='trained_models/bc/pendulum')
    parser.add_argument('--max_to_keep', help='number of models to save', default=10, type=int)
    # parser.add_argument('--logdir', help='log directory', default='log/train/bc/pendulum/')
    parser.add_argument('--logdir', help='log directory', default='log/train/bc/')
    parser.add_argument('--iteration', default=int(1e3), type=int)
    parser.add_argument('--interval', help='save interval', default=int(1e2), type=int)
    parser.add_argument('--minibatch_size', default=60, type=int)
    parser.add_argument('--epoch_num', default=60, type=int)
    return parser.parse_args()

# prepare sequential observations from expert demo data
def prepare_expert_seq(original_obs_data, seq_length=10, test_episodes_id=[8, 9]):
    training_seq_data = []
    testing_seq_data = []
    num_data = original_obs_data.shape[0]
    num_states = original_obs_data.shape[1]
    new_episode_start_id = -1
    episode_id = -1

    # start to shift the window with a length of seq_length
    for i in range(num_data):
        data_copy = original_obs_data[i].copy()
        # initialize the current_seq_data whenever start a new episode
        if data_copy[0] == np.inf:
            new_episode_start_id = i + 1
            episode_id += 1
            continue

        if i == new_episode_start_id:
            current_seq_data = []
            for _ in range(seq_length):
                first_data_of_new_episode = original_obs_data[i].copy()
                current_seq_data.append(first_data_of_new_episode)  # in the shape of (seq_length, num_states)
        else:
            current_seq_data.pop(0)
            current_seq_data.append(data_copy)

        seq_data_copy = current_seq_data.copy()

        # divide into training data-set and testing data-set
        # in the shape of (batch_size, seq_length, num_states)
        if episode_id in test_episodes_id:
            testing_seq_data.append(seq_data_copy)
        else:
            training_seq_data.append(seq_data_copy)

    # reshape the data-set into the shape of (batch_size, seq_length * num_states)
    training_seq_data = np.array(training_seq_data).astype(dtype=np.float32)
    training_seq_data = np.reshape(training_seq_data, newshape=[-1, seq_length * num_states])
    testing_seq_data = np.array(testing_seq_data).astype(dtype=np.float32)
    testing_seq_data = np.reshape(testing_seq_data, newshape=[-1, seq_length * num_states])

    # feature rescaling (i.e., normalization) for better training performance
    scaler = MinMaxScaler()
    scaled_training_seq_data = scaler.fit_transform(np.transpose(training_seq_data))
    scaled_training_seq_data = np.transpose(scaled_training_seq_data)
    scaled_testing_seq_data = scaler.fit_transform(np.transpose(testing_seq_data))
    scaled_testing_seq_data = np.transpose(scaled_testing_seq_data)

    # for i in range(all_seq_data.shape[0]):
    #     print("row: {}".format(i))
    #     print(all_seq_data[i])

    return scaled_training_seq_data, scaled_testing_seq_data

# prepare action data from expert demo
def prepare_action_data(original_action_data, test_episodes_id=[8, 9]):
    # inds_to_delete =[]
    training_action_data =[]
    testing_action_data = []
    num_data = original_action_data.shape[0]
    episode_id = -1

    for i in range(num_data):
        # the beginning of every new episode is labeled with np.inf for easy recognition
        if original_action_data[i][0] == np.inf:
            episode_id += 1
            # inds_to_delete.append(i)
            continue

        data_copy = original_action_data[i].copy() # in the form of (action_dimension,)

        # divide into training set and testing set, in the shape of (batch_size, action_dimension)
        if episode_id in test_episodes_id:
            testing_action_data.append(data_copy)
        else:
            training_action_data.append(data_copy)

    # cleared_action_data = np.delete(original_action_data, inds_to_delete, axis=0)
    training_action_data = np.array(training_action_data).astype(dtype=np.float32)
    testing_action_data = np.array(testing_action_data).astype(dtype=np.float32)

    return training_action_data, testing_action_data

# prepare non-sequential states and actions from expert demo with new state-action design
def prepare_expert_data(original_expert_obs, original_expert_act, state_max, state_min, action_max, action_min, test_episodes_id, include_robot_state=False, use_robot_vel=False, delta_t=0.05):
    # prepare training sets and testing sets
    training_states = []
    training_actions = []
    testing_states = []
    testing_actions = []

    num_data = original_expert_obs.shape[0]
    new_episode_start_id = -1
    episode_id = -1

    for i in range(num_data):
        if original_expert_obs[i][0] == np.inf:
            new_episode_start_id = i + 1
            episode_id += 1
            continue

        current_human_state = original_expert_obs[i].copy()
        current_robot_state = original_expert_act[i].copy()

        # prepare current human velocity at time step t
        if i == new_episode_start_id:
            last_human_state = current_human_state
            current_human_vel = (current_human_state - last_human_state) / delta_t
        else:
            last_human_state = original_expert_obs[i-1].copy()
            current_human_vel = (current_human_state - last_human_state) / delta_t

        # prepare next robot velocity right after time step t
        if (i + 1 == num_data) or original_expert_obs[i+1][0] == np.inf:
            next_robot_state = current_robot_state
            next_robot_vel = (next_robot_state - current_robot_state) / delta_t
        else:
            next_robot_state = original_expert_act[i+1].copy()
            next_robot_vel = (next_robot_state - current_robot_state) / delta_t

        # print("size of current_human_state: {}".format(current_human_state.shape))
        if not include_robot_state:
            state = np.concatenate([current_human_state, current_human_vel])
        else:
            state = np.concatenate([current_human_state, current_human_vel, current_robot_state])

        if not use_robot_vel:
            action = next_robot_state
        else:
            action = next_robot_vel

        # normalize the states and actions to [0, 1] for each dimension
        state = (state - state_min) / (state_max - state_min) # rescale the feature (i.e., normalization)
        # action = (action - action_min) / (action_max - action_min)

        # print("episode [{}], batch [{}]:".format(episode_id, i))
        # # print("current_robot_state: {}".format(current_robot_state))
        # # print("next_robot_state: {}".format(next_robot_state))
        # print("action: {}".format(action))
        # print("*********************************")


        # divide into training data-set and testing data-set
        # in the form of (batch_size, feature_dimension)
        if episode_id in test_episodes_id:
            testing_states.append(state)
            testing_actions.append(action)
            # print("episode [{}], batch [{}]:".format(episode_id, i))
            # print("current_robot_state: {}".format(current_robot_state))
            # print("next_robot_state: {}".format(next_robot_state))
            # print("action: {}".format(action))
            # print("*********************************")
        else:
            training_states.append(state)
            training_actions.append(action)

    training_states = np.array(training_states)
    training_actions = np.array(training_actions)
    testing_states = np.array(testing_states)
    testing_actions = np.array(testing_actions)

    '''
    scaler = MinMaxScaler()
    training_states = scaler.fit_transform(np.transpose(training_states))
    training_states = np.transpose(training_states)
    testing_states = scaler.fit_transform(np.transpose(testing_states))
    testing_states = np.transpose(testing_states)

    training_actions = scaler.fit_transform(np.transpose(training_actions))
    training_actions = np.transpose(training_actions)
    testing_actions = scaler.fit_transform(np.transpose(testing_actions))
    testing_actions = np.transpose(testing_actions)
    '''

    # print("training states:")
    # print(training_states)
    # print("*******************************")
    # print("training actions:")
    # print(training_actions)
    # print("*******************************")
    # print("testing states:")
    # print(testing_states)
    # print("********************************")
    # print("testing actions")
    # print(testing_actions)

    return training_states, training_actions, testing_states, testing_actions

# prepare sequential states and actions from expert demo with new state-action design
def prepare_expert_data_seq(original_expert_obs, original_expert_act,
                            state_max, state_min, action_max, action_min, num_states,
                            test_episodes_id, include_robot_state=False, use_robot_vel=False, seq_length=10, delta_t=0.05, train_episodes_id=None):
    # prepare training sets and testing sets
    training_states_seq = []
    training_actions = []
    testing_states_seq = []
    testing_actions = []

    num_data = original_expert_obs.shape[0]
    current_seq_data = []
    new_episode_start_id = -1
    episode_id = -1

    for i in range(num_data):
        if original_expert_obs[i][0] == np.inf:
            new_episode_start_id = i + 1
            episode_id += 1
            continue

        ''' First to prepare current state and action '''
        current_human_state = original_expert_obs[i].copy()
        current_robot_state = original_expert_act[i].copy()

        # prepare current human velocity at time step t
        if i == new_episode_start_id:
            last_human_state = current_human_state
            current_human_vel = (current_human_state - last_human_state) / delta_t
        else:
            last_human_state = original_expert_obs[i - 1].copy()
            current_human_vel = (current_human_state - last_human_state) / delta_t

        # prepare next robot velocity right after time step t
        if (i + 1 == num_data) or original_expert_obs[i + 1][0] == np.inf:
            next_robot_state = current_robot_state
            next_robot_vel = (next_robot_state - current_robot_state) / delta_t
        else:
            next_robot_state = original_expert_act[i + 1].copy()
            next_robot_vel = (next_robot_state - current_robot_state) / delta_t

        # print("sample [{}]:".format(i))
        # print("next robot state:")
        # print(next_robot_state)
        # print("current robot state:")
        # print(current_robot_state)
        # print("next robot velocity:")
        # print(next_robot_vel)

        # construct the state
        if not include_robot_state:
            state = np.concatenate([current_human_state, current_human_vel])
        else:
            state = np.concatenate([current_human_state, current_human_vel, current_robot_state])

        # construct the action
        if not use_robot_vel:
            action = next_robot_state
        else:
            action = next_robot_vel

        # print("action:")
        # print(action)
        # print("state before normalized:")
        # print(state)
        # print("**********************************************************")


        # normalize the states and actions to [0, 1] for each dimension
        state = (state - state_min) / (state_max - state_min) # rescale the feature (i.e., normalization)
        # action = (action - action_min) / (action_max - action_min)

        ''' Then to prepare sequential state data '''
        if i == new_episode_start_id:
            current_seq_data = []
            for _ in range(seq_length):
                first_data_of_new_episode = state.copy()
                current_seq_data.append(first_data_of_new_episode)  # in the shape of (seq_length, num_states)
        else:
            current_seq_data.pop(0)
            current_seq_data.append(state.copy())

        seq_data_copy = current_seq_data.copy()

        if episode_id in test_episodes_id:
            testing_states_seq.append(seq_data_copy) # in the shape of (batch_size, seq_length, num_states)
            testing_actions.append(action)
        else:
            if train_episodes_id is None:
                training_states_seq.append(seq_data_copy)
                training_actions.append(action)
            else:
                if episode_id in train_episodes_id:
                    training_states_seq.append(seq_data_copy)
                    training_actions.append(action)

    ''' Turn the list into np.array and reshape for Policy_net input '''
    training_states_seq = np.array(training_states_seq).astype(dtype=np.float32) # in the shape of (batch_size, seq_length, num_states)
    training_states_seq = np.reshape(training_states_seq, newshape=[-1, seq_length * num_states])
    testing_states_seq = np.array(testing_states_seq).astype(dtype=np.float32)
    testing_states_seq = np.reshape(testing_states_seq, newshape=[-1, seq_length * num_states])
    training_actions = np.array(training_actions).astype(dtype=np.float32) # in the shape of (batch_size, num_actions)
    testing_actions = np.array(testing_actions).astype(dtype=np.float32)

    return training_states_seq, training_actions, testing_states_seq, testing_actions


def load_multiple_subject_data(sub_ids=[1], act_shape=(4,), demo_data_dir='/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/demo_data/pepper/'):
    all_obs_data = []
    all_act_data = []
    for sub_id in sub_ids:
        demo_observations = np.genfromtxt(demo_data_dir + str(sub_id) + '/expert_observations.csv')
        demo_actions = np.genfromtxt(demo_data_dir + str(sub_id) + '/expert_actions.csv')
        demo_actions = np.reshape(demo_actions, newshape=[-1] + list(act_shape))  # in the shape of (batch_size, action_dimension)

        # if all_obs_data is None:
        #     all_obs_data = demo_observations
        #     print(all_obs_data)
        # else:
        #     all_obs_data = np.concatenate((all_obs_data, demo_observations), axis=0)
        all_obs_data.append(demo_observations)

        # if all_act_data is None:
        #     all_act_data = demo_actions
        # else:
        #     all_act_data = np.concatenate((all_act_data, demo_actions), axis=0)
        all_act_data.append(demo_actions)

    all_obs_data = np.array(all_obs_data).astype(dtype=np.float32) # in the shape of (num_subs, batch_size, state_dim)
    all_act_data = np.array(all_act_data).astype(dtype=np.float32) # in the shape of (num_subs, batch_size, action_dim)

    state_dim = all_obs_data.shape[2]
    all_obs_data = np.reshape(all_obs_data, newshape=[-1, state_dim])
    action_dim = all_act_data.shape[2]
    all_act_data = np.reshape(all_act_data, newshape=[-1, action_dim])

    return all_obs_data, all_act_data




def save_demo(demo_data, demo_path):
    try:
        with open(demo_path, 'ab') as f_handle:
            np.savetxt(f_handle, demo_data, fmt='%s')
    except FileNotFoundError:
        with open(demo_path, 'wb') as f_handle:
            np.savetxt(f_handle, demo_data, fmt='%s')


def main(args):
    total_subject_num = 15
    subject_ids = list(range(1, total_subject_num + 1))
    traj_per_human = 5
    """training_subjects_num = len(subject_ids)"""

    # training_traj_num_list = list(range(2, 8 * len(subject_ids) + 1, 2)) # [2, 4, 6, 8, ..., 14, 16]
    """training_traj_num_list = list(range(4, 8 * len(subject_ids) + 1, 4))
    mse_results = []"""

    """for training_traj_num in training_traj_num_list:
        trials_num = 3
        total_mse = 0.0"""

    """print("###################################")
    print("[BC: {} subject, {} training traj]: Start".format(len(subject_ids), training_traj_num))"""
    """for trial_id in range(trials_num):"""
    # training_traj_num = 6

    data_collection_mode = 'guided_collect'
    summary_sub_dir = 'pepper/bc/experiments/' + data_collection_mode + '/'
    """summary_sub_dir = 'pepper/bc/' + str(training_subjects_num) + '_subject/' + str(training_traj_num) + '_training_traj/' + 'trial_id_' + str(trial_id + 1) + '/'"""
    """data_sub_dir = 'pepper/2/'"""
    # data_sub_dir = 'side_view/multi_pos/'
    model_sub_dir = 'pepper/bc/experiments/' + data_collection_mode + '/'
    """model_sub_dir = 'pepper/bc/' + str(training_subjects_num) + '_subject/' + str(training_traj_num) + '_training_traj/' + 'trial_id_' + str(trial_id + 1) + '/'"""

    if data_collection_mode == 'guided_collect':
        reg_factor = 5e-4
    else:
        reg_factor = 1e-3
    # dir_note = 'side_view/include_back/'

    '''
    seq_length = 10
    testing_episodes_id = [39, 38, 37, 36, 35, 34, 33, 32]
    # testing_episodes_id = [8, 9]

    expert_observations = np.genfromtxt('demo_data/' + data_sub_dir + 'expert_observations.csv')
    # expert_observations = np.reshape(expert_observations, newshape=[-1] + list(Policy.ob_space.shape))  # in the shape of (batch_size, num_states)
    expert_actions = np.genfromtxt('demo_data/' + data_sub_dir + 'expert_actions.csv')
    expert_actions = np.reshape(expert_actions, newshape=[-1] + list(Policy.act_space.shape))  # in the shape of (batch_size, action_dimension)

    # prepare the range of state and action
    joint_values_max = np.array([2.0857, 0.3142, 2.0857, 1.5446])
    joint_values_min = np.array([-2.0857, -1.3265, -2.0857, 0.0349])
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
    state_dimension = Policy.ob_space.shape[0]

    # prepare the state and action data of expert demo
    expert_observations_training, expert_actions_training, \
    expert_observations_testing, expert_actions_testing = prepare_expert_data_seq(expert_observations, expert_actions,
                                                                                  state_max, state_min,
                                                                                  action_max, action_min,
                                                                                  state_dimension, testing_episodes_id,
                                                                                  state_with_robot_state,
                                                                                  action_with_robot_vel,
                                                                                  seq_length)
    # expert_observations_training, expert_actions_training, \
    # expert_observations_testing, expert_actions_testing = prepare_expert_data(expert_observations, expert_actions,
    #                                                                           state_max, state_min,
    #                                                                           action_max, action_min,
    #                                                                           testing_episodes_id,
    #                                                                           state_with_robot_state,
    #                                                                           action_with_robot_vel)
    '''

    """
    # expert_observations_training, expert_observations_testing = prepare_expert_seq(expert_observations, seq_length, testing_episodes_id) # in the shape of (batch_size, seq_length * state_dimension)
    # expert_observations = (expert_observations - expert_observations.min()) / (expert_observations.max() - expert_observations.min())
    demo_path_obs_training = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/demo_data/side_view/multi_pos/expert_observations_training.csv'
    # demo_path_obs_training = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/demo_data/side_view/include_back/expert_observations_training.csv'
    save_demo(expert_observations_training, demo_path_obs_training)
    demo_path_obs_testing = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/demo_data/side_view/multi_pos/expert_observations_testing.csv'
    # demo_path_obs_testing = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/demo_data/side_view/include_back/expert_observations_testing.csv'
    save_demo(expert_observations_training, demo_path_obs_testing)
    """

    '''
    print("Finish processing observation data")
    print("size of training observation data: {}".format(np.shape(expert_observations_training)))
    print("size of testing observation data: {}".format(np.shape(expert_observations_testing)))
    '''
    '''
    # expert_actions_training, expert_actions_testing = prepare_action_data(expert_actions, testing_episodes_id)
    demo_path_act_training = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/demo_data/side_view/multi_pos/expert_actions_training.csv'
    # demo_path_act_training = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/demo_data/side_view/include_back/expert_actions_training.csv'
    save_demo(expert_actions_training, demo_path_act_training)
    demo_path_act_testing = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/demo_data/side_view/multi_pos/expert_actions_testing.csv'
    # demo_path_act_testing = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/demo_data/side_view/include_back/expert_actions_testing.csv'
    save_demo(expert_actions_testing, demo_path_act_testing)
    '''

    '''
    print("Finish processing action data")
    print("size of training action data: {}".format(np.shape(expert_actions_training)))
    print("size of testing action data: {}".format(np.shape(expert_actions_testing)))
    '''

    # To-do:
    # expert_actions and expert_observations may need to be truncated because of synchronization problem
    # options:
    # 1) Manually revise the csv file beforehand
    # 2) Delete the extra batches
    # 3) Don't use SIC motion-record and write new recording functions
    # batch_action = expert_actions.shape[0]
    # batch_obs = expert_observations.shape[0]
    # min_batch = min(batch_obs, batch_action)
    # expert_observations = expert_observations[:min_batch, :]
    # expert_actions = expert_actions[:min_batch, :]

    """print("**************************")
    print("[BC: {} subject, {} training traj, trial id {}]: Start training...".format(len(subject_ids), training_traj_num, trial_id + 1))"""
    tf.reset_default_graph()
    with tf.Session() as sess:
        # writer = tf.summary.FileWriter(args.logdir + dir_note + 'new_state_design_seq/' + TIMESTAMP, sess.graph)
        writer = tf.summary.FileWriter(args.logdir + summary_sub_dir + TIMESTAMP, sess.graph)

        Policy = Policy_net('policy', using_latent_state=False, using_lstm=False, reg_factor=reg_factor)
        BC = BehavioralCloning(Policy)

        saver = tf.train.Saver(max_to_keep=100)
        sess.run(tf.global_variables_initializer())
        print('Finish Initializing')

        ''' Prepare Demo Data '''
        seq_length = 10

        # testing_episodes_id = [39, 38, 37, 36, 35, 34, 33, 32]
        # testing_episodes_id = [0, 1, 10, 11, 20, 21]
        # training_episodes_id, testing_episodes_id = demo_data_partition(total_trial_num=len(subject_ids) * 10,
        #                                                                 trial_per_human_for_training=8,
        #                                                                 trial_per_human_for_testing=2,
        #                                                                 trial_per_human=10)
        if data_collection_mode == 'guided_collect':
            testing_ids_per_human = [3]
        else:
            testing_ids_per_human = []
        training_episodes_id, testing_episodes_id = data_set_partition(sub_ids=subject_ids,
                                                                       training_traj_num=None,
                                                                       random_training_set=False,
                                                                       testing_ids_per_human=testing_ids_per_human,
                                                                       traj_per_human=traj_per_human
                                                                       )

        print("Training episodes id:")
        print(training_episodes_id)
        print("Testing episodes id:")
        print(testing_episodes_id)

        # testing_episodes_id = [8, 9]

        '''
        expert_observations = np.genfromtxt('demo_data/' + data_sub_dir + 'expert_observations.csv')
        # expert_observations = np.reshape(expert_observations, newshape=[-1] + list(Policy.ob_space.shape))  # in the shape of (batch_size, num_states)
        expert_actions = np.genfromtxt('demo_data/' + data_sub_dir + 'expert_actions.csv')
        expert_actions = np.reshape(expert_actions, newshape=[-1] + list(Policy.act_space.shape))  # in the shape of (batch_size, action_dimension)
        '''
        expert_data_dir = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/experiment_data/human_feature_exp/' + data_collection_mode + '/subject_'
        expert_observations, expert_actions = load_multiple_subject_data(sub_ids=subject_ids,
                                                                         act_shape=(6,),
                                                                         demo_data_dir=expert_data_dir)

        # prepare the range of state and action
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
        state_dimension = Policy.ob_space.shape[0]

        # prepare the state and action data of expert demo
        expert_observations_training, expert_actions_training, \
        expert_observations_testing, expert_actions_testing = prepare_expert_data_seq(expert_observations,
                                                                                      expert_actions,
                                                                                      state_max, state_min,
                                                                                      action_max, action_min,
                                                                                      state_dimension,
                                                                                      testing_episodes_id,
                                                                                      state_with_robot_state,
                                                                                      action_with_robot_vel,
                                                                                      seq_length,
                                                                                      train_episodes_id=training_episodes_id)

        print("Finish processing observation data")
        print("size of training observation data: {}".format(np.shape(expert_observations_training)))
        print("size of testing observation data: {}".format(np.shape(expert_observations_testing)))

        print("Finish processing action data")
        print("size of training action data: {}".format(np.shape(expert_actions_training)))
        print("size of testing action data: {}".format(np.shape(expert_actions_testing)))

        # writer = tf.summary.FileWriter(args.logdir + summary_sub_dir + TIMESTAMP, sess.graph)
        # sess.run(tf.global_variables_initializer())
        # print("Finish Initializing")

        inp = [expert_observations_training, expert_actions_training]
        """inp_testing = [expert_observations_testing, expert_actions_testing]"""
        # lr = 1e-4

        # data_size = expert_observations_training.shape[0]
        # batch_size = args.minibatch_size
        # epoch_num = int(data_size / batch_size)
        for iteration in range(args.iteration):  # episode
            # print("***************************")
            # print("[iteration {}]: Start training".format(iteration))

            # train
            for epoch in range(args.epoch_num):
                # select sample indices in [low, high)
                sample_indices = np.random.randint(low=0, high=expert_observations_training.shape[0], size=args.minibatch_size)
                # sample_indices = list(range(epoch * batch_size, (epoch + 1) * batch_size))
                #
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                # print("[iteration {}, epoch {}]: Finish sampling inputs".format(iteration, epoch))

                # BC.train(obs=sampled_inp[0], actions=sampled_inp[1])
                BC.train(obs=sampled_inp[0], actions=sampled_inp[1])
                # print("[iteration {}, epoch {}]: Finish training".format(iteration + 1, epoch + 1))
                # BC.train(obs=expert_observations, actions=expert_actions)

            """action_mse = tf.get_default_session().run(BC.mse_actions_testing,
                                                      feed_dict={BC.Policy.obs: inp_testing[0],
                                                                 BC.expert_actions_testing: inp_testing[1]})

            print("[iteration {}]: Finished training, testing MSE is: {}".format(iteration + 1, action_mse ))"""
            print("[iteration {}]: Finished training".format(iteration + 1))

            summary = BC.get_summary(obs=inp[0], actions=inp[1])
            """summary_testing = BC.get_summary_testing(obs=inp_testing[0], actions=inp_testing[1])"""
            # print("[iteration {}]: Finish retrieving summary".format(iteration))

            if (iteration+1) % args.interval == 0:
                # lr *= 0.1
                # saver.save(sess, args.savedir + dir_note + 'new_state_design_seq/' + 'model.ckpt', global_step=iteration+1)
                if not os.path.exists(args.savedir + model_sub_dir):
                    os.makedirs(args.savedir + model_sub_dir)
                saver.save(sess, args.savedir + model_sub_dir + 'model.ckpt', global_step=iteration + 1)
                print("[BC: iteration {}]: save model!".format(iteration + 1))

            writer.add_summary(summary, iteration)
            """writer.add_summary(summary_testing, iteration)"""
            # print("[iteration {}]: Saved summary".format(iteration))

        # action_mse = tf.get_default_session().run(BC.mse_actions_testing,
        #                                           feed_dict={BC.Policy.obs: inp_testing[0],
        #                                                      BC.expert_actions_testing: inp_testing[1]})
        # # total_mse += action_mse
        # print("Action MSE for testing set: {}".format(action_mse))

        writer.close()
        print("Finished behavior cloning!")

    """average_action_mse = total_mse / trials_num
    mse_results.append([training_traj_num, average_action_mse])
    print("[BC: {} subject, {} training traj]: Finished training, Average MSE is: {}".format(len(subject_ids), training_traj_num, average_action_mse))"""

    """print("[BC: {} subject]: All training finished!".format(len(subject_ids)))
    mse_results = np.array(mse_results)
    save_data(data=mse_results, path='/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/trained_models/bc/pepper/bc/'
                                     + str(len(subject_ids)) + '_subject/mse_results.csv')"""


if __name__ == '__main__':
    args = argparser()
    main(args)
