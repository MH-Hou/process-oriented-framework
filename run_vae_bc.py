import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from sklearn.preprocessing import MinMaxScaler
import os

from datetime import datetime

from policy_net import Policy_net
from behavior_clone import BehavioralCloning
from run_behavior_clone import argparser, prepare_expert_data_seq, load_multiple_subject_data
from vae_tf1 import Vae, train_vae
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



def main(args):
    subject_ids = [1, 2, 3]
    training_subjects_num = len(subject_ids)

    # training_traj_num_list = list(range(2, 8 * len(subject_ids) + 1, 2))
    training_traj_num_list = list(range(4, 8 * len(subject_ids) + 1, 4)) # [2, 4, 6, 8, ..., 14, 16]
    mse_results = []

    for training_traj_num in training_traj_num_list:
        trials_num = 3
        total_mse = 0.0

        print("###################################")
        print("[VAE-BC: {} subject, {} training traj]: Start".format(len(subject_ids), training_traj_num))
        for trial_id in range(trials_num):
            # training_traj_num = 6
            summary_sub_dir = 'pepper/vae_bc/' + str(training_subjects_num) + '_subject/' + str(training_traj_num) + '_training_traj/' + 'trial_id_' + str(trial_id + 1) + '/'
            # data_sub_dir = 'pepper/2/'
            # data_sub_dir = 'side_view/multi_pos/'
            model_sub_dir = 'pepper/vae_bc/' + str(training_subjects_num) + '_subject/' + str(training_traj_num) + '_training_traj/' + 'trial_id_' + str(trial_id + 1) + '/'
            whether_trained_vae = False

            print("**************************")
            print("[VAE-BC: {} subject, {} training traj, trial id {}]: Start training...".format(len(subject_ids), training_traj_num, trial_id + 1))
            tf.reset_default_graph()
            with tf.Session() as sess:
                vae = Vae('vae', latent_dim=7, reg_factor=1e-1, using_lstm=False, kl_factor=0.5)
                Policy = Policy_net('policy', using_latent_state=True, latent_dim=vae.latent_dim, reg_factor=1e-3)
                BC = BehavioralCloning(Policy)

                saver = tf.train.Saver(max_to_keep=100)
                writer = tf.summary.FileWriter(args.logdir + summary_sub_dir + TIMESTAMP, sess.graph)

                sess.run(tf.global_variables_initializer())
                print('Finish Initializing')

                ''' Prepare Demo Data '''
                seq_length = 10
                # testing_episodes_id = [39, 38, 37, 36, 35, 34, 33, 32]
                # testing_episodes_id = [0, 1, 10, 11, 20, 21]
                # testing_episodes_id = [8, 9]
                # training_episodes_id, testing_episodes_id = demo_data_partition(total_trial_num=len(subject_ids)*10,
                #                                                                 trial_per_human_for_training=8,
                #                                                                 trial_per_human_for_testing=2,
                #                                                                 trial_per_human=10)

                training_episodes_id, testing_episodes_id = data_set_partition(sub_ids=subject_ids,
                                                                               random_training_set=True,
                                                                               testing_ids_per_human=[0, 9],
                                                                               training_traj_num=training_traj_num)

                print("Training episodes id:")
                print(training_episodes_id)
                print("Testing episodes id:")
                print(testing_episodes_id)

                '''
                expert_observations = np.genfromtxt('demo_data/' + data_sub_dir + 'expert_observations.csv')
                # expert_observations = np.reshape(expert_observations, newshape=[-1] + list(Policy.ob_space.shape))  # in the shape of (batch_size, num_states)
                expert_actions = np.genfromtxt('demo_data/' + data_sub_dir + 'expert_actions.csv')
                expert_actions = np.reshape(expert_actions, newshape=[-1] + list(Policy.act_space.shape))  # in the shape of (batch_size, action_dimension)
                '''
                expert_observations, expert_actions = load_multiple_subject_data(subject_ids)

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

                ''' Train Vae '''
                if not whether_trained_vae:
                    train_vae(vae=vae,
                              training_data=expert_observations_training,
                              testing_data=expert_observations_testing,
                              sess=sess,
                              saver=saver,
                              writer=writer,
                              fig_saving_dir='/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/trained_models/bc/' + model_sub_dir
                              )

                    print("Finish training vae")
                else:
                    saver.restore(sess, 'vae_trained_models/no_lstm/' + 'model.ckpt-' + '3000')
                    print("Finish loading vae model")

                expert_latent_observations_training = vae.get_latent_state(expert_observations_training) # in the shape of (batch_size, latent_dim)
                expert_latent_observations_testing = vae.get_latent_state(expert_observations_testing)
                inp = [expert_latent_observations_training, expert_actions_training]
                inp_testing = [expert_latent_observations_testing, expert_actions_testing]

                for iteration in range(args.iteration):  # episode
                    # print("***************************")
                    # print("[iteration {}]: Start training".format(iteration))

                    # train
                    for epoch in range(args.epoch_num):
                        # select sample indices in [low, high)
                        sample_indices = np.random.randint(low=0, high=expert_observations_training.shape[0], size=args.minibatch_size)
                        #
                        sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                        # print("[iteration {}, epoch {}]: Finish sampling inputs".format(iteration, epoch))

                        # BC.train(obs=sampled_inp[0], actions=sampled_inp[1])
                        BC.train(obs=sampled_inp[0], actions=sampled_inp[1])
                        # print("[iteration {}, epoch {}]: Finish training".format(iteration, epoch))
                        # BC.train(obs=expert_observations, actions=expert_actions)

                    # print("[iteration {}]: Finished training".format(iteration))

                    summary = BC.get_summary(obs=inp[0], actions=inp[1])
                    summary_testing = BC.get_summary_testing(obs=inp_testing[0], actions=inp_testing[1])
                    # print("[iteration {}]: Finish retrieving summary".format(iteration))

                    if (iteration+1) % args.interval == 0:
                        # lr *= 0.1
                        # saver.save(sess, args.savedir + dir_note + 'new_state_design_seq/' + 'model.ckpt', global_step=iteration+1)
                        if not os.path.exists(args.savedir + model_sub_dir):
                            os.makedirs(args.savedir + model_sub_dir)
                        saver.save(sess, args.savedir + model_sub_dir + 'model.ckpt', global_step=iteration + 1)
                        print("[VAE-BC: {} subject, {} training traj, trial {}, iteration {}]: save model!".format(
                            len(subject_ids),
                            training_traj_num,
                            trial_id + 1,
                            iteration + 1))

                    writer.add_summary(summary, iteration)
                    writer.add_summary(summary_testing, iteration)
                    # print("[iteration {}]: Saved summary".format(iteration))

                action_mse = tf.get_default_session().run(BC.mse_actions_testing,
                                                          feed_dict={BC.Policy.obs: inp_testing[0],
                                                                     BC.expert_actions_testing: inp_testing[1]})
                total_mse += action_mse

                writer.close()
                print("Finished behavior cloning!")

        average_action_mse = total_mse / trials_num
        mse_results.append([training_traj_num, average_action_mse])
        print("[VAE-BC: {} subject, {} training traj]: Finished training, Average MSE is: {}".format(len(subject_ids),
                                                                                                     training_traj_num,
                                                                                                     average_action_mse))

    print("[VAE-BC: {} subject]: All training finished!".format(len(subject_ids)))
    mse_results = np.array(mse_results)
    save_data(data=mse_results, path='/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/trained_models/bc/pepper/vae_bc/'
                                     + str(len(subject_ids)) + '_subject/mse_results.csv')


if __name__ == '__main__':
    args = argparser()
    main(args)