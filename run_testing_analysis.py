import tensorflow.compat.v1 as tf
tf.disable_eager_execution()   #禁止TensorFlow2默认的即时执行模式
import numpy as np

from run_test import argparser
from vae_tf1 import Vae
from policy_net import Policy_net
from behavior_clone import BehavioralCloning
from run_behavior_clone import prepare_expert_data_seq
from utils.utils import load_multiple_subject_data, data_set_partition, save_data



def main(args):
    using_vae = True
    vae_using_lstm = False
    bc_using_lstm = False

    testing_subject_ids = [3]

    # directory parameters for loading trained model
    training_subject_ids = [1]
    training_subjects_num = len(training_subject_ids)

    training_traj_num_list = list(range(2, 8 * len(training_subject_ids) + 1, 2))
    mse_results = []
    # training_traj_num = 4

    for training_traj_num in training_traj_num_list:
        trials_num = 3
        total_mse = 0.0

        for trial_id in range(trials_num):
            tf.reset_default_graph()

            if using_vae:
                vae = Vae('vae', latent_dim=7, using_lstm=vae_using_lstm)
                Policy = Policy_net('policy', using_latent_state=using_vae, latent_dim=vae.latent_dim)
                BC = BehavioralCloning(Policy)
                model_dir = 'pepper/vae_bc/' + str(training_subjects_num) + '_subject/' + str(training_traj_num) + '_training_traj/' + 'trial_id_' + str(trial_id + 1) + '/'
            else:
                Policy = Policy_net('policy', using_latent_state=using_vae, using_lstm=bc_using_lstm)
                BC = BehavioralCloning(Policy)
                model_dir = 'pepper/bc/' + str(training_subjects_num) + '_subject/' + str(training_traj_num) + '_training_traj/' + 'trial_id_' + str(trial_id + 1) + '/'

            saver = tf.train.Saver()

            expert_observations, expert_actions = load_multiple_subject_data(testing_subject_ids)
            seq_length = 10
            num_states = Policy.ob_space.shape[0]

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                # load the trained model
                if args.model == '':
                    saver.restore(sess, args.modeldir + '/' + args.alg + '/' + model_dir + 'model.ckpt')
                else:
                    saver.restore(sess, args.modeldir + '/' + args.alg + '/' + model_dir + 'model.ckpt-' + args.model)

                # parameters related to episode length
                step_delta_t = 0.05
                episode_time_length = 3.0
                episode_total_steps = int(episode_time_length / step_delta_t)

                # prepare
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

                testing_expert_obs, testing_expert_act, _, _ = prepare_expert_data_seq(original_expert_obs=expert_observations,
                                                                              original_expert_act=expert_actions,
                                                                              state_max=state_max,
                                                                              state_min=state_min,
                                                                              action_max=action_max,
                                                                              action_min=action_min,
                                                                              num_states=num_states,
                                                                              test_episodes_id=[],
                                                                              include_robot_state=state_with_robot_state,
                                                                              use_robot_vel=action_with_robot_vel,
                                                                              seq_length=seq_length,
                                                                              delta_t=delta_t,
                                                                              train_episodes_id=None)

                print("Finish processing data")
                print("size of testing observation data: {}".format(np.shape(testing_expert_obs)))
                print("size of testing action data: {}".format(np.shape(testing_expert_act)))

                if using_vae:
                    testing_latent_expert_obs = vae.get_latent_state(testing_expert_obs)
                    action_mse = tf.get_default_session().run(BC.mse_actions_testing,
                                                                       feed_dict={BC.Policy.obs: testing_latent_expert_obs,
                                                                            BC.expert_actions_testing: testing_expert_act})
                    total_mse += action_mse

                    # average_mse_actions = total_mse / trials_num
                    # print("[VAE-BC: {} training subjects, {} training traj]: Average MSE is {}".format(training_subjects_num,
                    #                                                                            training_traj_num,
                    #                                                                            average_mse_actions))
                else:
                    action_mse = tf.get_default_session().run(BC.mse_actions_testing,
                                                                       feed_dict={BC.Policy.obs: testing_expert_obs,
                                                                                  BC.expert_actions_testing: testing_expert_act})
                    total_mse += action_mse
                    # print("[BC: {} subjects, {} training traj]: Average MSE is {}".format(training_subjects_num,
                    #                                                                            training_traj_num,
                    #                                                                            average_mse_actions))

        average_action_mse = total_mse / trials_num
        mse_results.append([training_traj_num, average_action_mse])

        if using_vae:
            print("[VAE-BC: {} training traj]: Average MSE is: {}".format(training_traj_num, average_action_mse))
        else:
            print("[BC: {} training traj]: Average MSE is: {}".format(training_traj_num, average_action_mse))

    mse_results = np.array(mse_results)
    if using_vae:
        save_data(data=mse_results,
                  path='/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/trained_models/bc/pepper/vae_bc/'
                       + str(len(training_subject_ids)) + '_subject/mse_results_testing.csv')
        print("[VAE-BC: {} subject]: All analysis finished!".format(len(training_subject_ids)))
    else:
        save_data(data=mse_results,
                  path='/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/trained_models/bc/pepper/bc/'
                       + str(len(training_subject_ids)) + '_subject/mse_results_testing.csv')
        print("[BC: {} subject]: All analysis finished!".format(len(training_subject_ids)))



if __name__=='__main__':
    args = argparser()
    main(args)