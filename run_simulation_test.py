import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from time import sleep

from simulation_robot_client import simulation_robot_client
from policy_net import Policy_net
from environment import GreetingEnv

from run_test import argparser
from run_behavior_clone import prepare_expert_data_seq, load_multiple_subject_data
from vae_tf1 import Vae
from run_vae_bc import demo_data_partition


def main(args):
    subject_ids = [1, 2]
    using_vae = False
    vae_using_lstm = False
    bc_using_lstm = False
    training_subjects_num = len(subject_ids)
    data_collection_mode = 'guided_collect'
    # training_traj_num = 16

    robot_client = simulation_robot_client(using_gui=True, auto_step=True)

    if using_vae:
        vae = Vae('vae', latent_dim=7, using_lstm=vae_using_lstm)
        Policy = Policy_net('policy', using_latent_state=using_vae, latent_dim=vae.latent_dim)
        model_dir = 'pepper/vae_bc/experiments/' + data_collection_mode + '/'
        # model_dir = 'pepper/vae_bc/' + str(training_subjects_num) + '_subject/' + str(training_traj_num) + '_training_traj/trial_id_1/'
    else:
        Policy = Policy_net('policy', using_latent_state=using_vae, using_lstm=bc_using_lstm)
        model_dir = 'pepper/bc/experiments/' + data_collection_mode + '/'
        # model_dir = 'pepper/bc/' + str(training_subjects_num) + '_subject/' + str(training_traj_num) + '_training_traj/trial_id_1/'

    saver = tf.train.Saver()

    # model_dir = 'pepper/vae_bc/'
    # model_dir = 'side_view/multi_pos/new_state_design_seq/'
    # data_sub_dir = 'side_view/multi_pos/'
    # data_sub_dir = 'pepper/2/'
    '''
    expert_observations = np.genfromtxt('demo_data/' + data_sub_dir + 'expert_observations.csv')
    expert_actions = np.genfromtxt('demo_data/' + data_sub_dir + 'expert_actions.csv')
    expert_actions = np.reshape(expert_actions, newshape=[-1] + list(Policy.act_space.shape))  # in the shape of (batch_size, action_dimension)
    '''

    expert_data_dir = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/experiment_data/human_feature_exp/' + data_collection_mode + '/subject_'
    expert_observations, expert_actions = load_multiple_subject_data(sub_ids=subject_ids,
                                                                     act_shape=(6,),
                                                                     demo_data_dir=expert_data_dir)
    # expert_observations, expert_actions = load_multiple_subject_data(subject_ids)
    testing_episodes_id = [0, 5]

    seq_length = 10
    num_states = Policy.ob_space.shape[0]

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logdir + '/simulation/nao/' + args.alg, sess.graph)
        sess.run(tf.global_variables_initializer())

        # load the trained model
        if args.model == '':
            saver.restore(sess, args.modeldir+'/'+args.alg+'/'+ model_dir + 'model.ckpt')
        else:
            saver.restore(sess, args.modeldir+'/'+args.alg+'/'+ model_dir + 'model.ckpt-' + args.model)

        # parameters related to episode length
        step_delta_t = 0.05
        episode_time_length = 3.0
        episode_total_steps = int(episode_time_length / step_delta_t)

        # prepare
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

        _, _, testing_expert_obs, _ = prepare_expert_data_seq(expert_observations, expert_actions,
                                                              state_max, state_min, action_max, action_min,
                                                              num_states, testing_episodes_id,
                                                              state_with_robot_state, action_with_robot_vel,
                                                              seq_length, delta_t)

        num_iterations = int(testing_expert_obs.shape[0] / episode_total_steps)
        if using_vae:
            testing_latent_expert_obs = vae.get_latent_state(testing_expert_obs)


        for iteration in range(num_iterations):
            current_step = 0
            print("****************************************************")
            print("Episode {}: Going to start a new episode".format(iteration + 1))
            sleep(1.0)

            while current_step < episode_total_steps:
                if using_vae:
                    state = testing_latent_expert_obs[iteration * episode_total_steps + current_step]
                    state = np.reshape(state, newshape=[-1, vae.latent_dim])
                else:
                    state = testing_expert_obs[iteration * episode_total_steps + current_step]
                    state = np.reshape(state, newshape=[-1, seq_length * num_states])
                # state = testing_latent_expert_obs[iteration * episode_total_steps + current_step]
                # state = np.reshape(state, newshape=[-1, vae.latent_dim])
                act, _ = Policy.act(obs=state, stochastic=False) # in the form of normalized angular velocity of each joint
                reshaped_act = np.array(act) # in the shape of (action_dimension,)
                robot_client.take_action(reshaped_act)

                sleep(step_delta_t)
                print("Episode {} step {}: Finish taking action".format(iteration + 1, current_step))

                # update the state for next step
                current_step += 1

            robot_client.go_home_pose()
            sleep(2.0)
            print("Episode {}: Episode finished".format(iteration + 1))

        print("All episodes finished")

if __name__=='__main__':
    args = argparser()
    main(args)