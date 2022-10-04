import tensorflow as tf2
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import numpy as np
import os

import tensorflow_probability as tfp

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

from datetime import datetime

tfd = tfp.distributions

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())


class Vae:
    def __init__(self, name, latent_dim=2, using_lstm=False, seq_length=10, state_dim=12, reg_factor=1e-1, dropout=0.3, lstm_size=64, kl_factor=0.5, lr=1e-3, lstm_layers=1):
        self.name = name
        self.latent_dim = latent_dim
        self.using_lstm = using_lstm
        self.seq_length = seq_length
        self.state_dim = state_dim
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.regularizer = tf2.keras.regularizers.l2(reg_factor)
        self.dropout = dropout
        self.kl_factor = kl_factor
        self.learning_rate = lr

        self.construct_networks()
        self.build_loss_function()
        self.build_summary()

    def construct_networks(self):
        self.z_posterior, self.z_mean, self.z_sd = self.build_encoder()
        # self.sampled_z = self.z_posterior.sample() # in the shape of (batch_size, z_dim)
        self.sampled_z = self.z_mean + self.z_sd * tf.random.normal(shape=[self.latent_dim]) # reparametrized trick
        self.new_x_dist, self.new_x_mean, self.new_x_sd = self.build_decoder(self.sampled_z)

    def build_encoder(self):
        with tf.variable_scope(self.name):
            # input shape is in the form of (batch_size, seq_length * state_dim)
            self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.seq_length * self.state_dim])

            with tf.variable_scope('encoder'):
                if not self.using_lstm:
                    # construct dense layers
                    dense_l1 = tf.layers.dense(self.x, 200, tf.nn.tanh, kernel_regularizer=self.regularizer)
                    dense_l2 = tf.layers.dense(dense_l1, 200, tf.nn.tanh, kernel_regularizer=self.regularizer)

                    # construct dense layers for mean and standard deviation for latent space normal distribution
                    loc = tf.layers.dense(dense_l2, self.latent_dim, kernel_regularizer=self.regularizer)
                    scale = tf.layers.dense(dense_l2, self.latent_dim, tf.nn.softplus, kernel_regularizer=self.regularizer)
                else:
                    # reshape x into the shape of (batch_size, seq_length, state_dim) for LSTM input
                    reshaped_x = tf.reshape(self.x, shape=[-1, self.seq_length, self.state_dim])
                    # last_hidden_state = tf.keras.layers.LSTM(self.lstm_size, dropout=self.dropout)(reshaped_x)  # in the shape of (batch_size, lstm_dim)
                    # construct the LSTM layers to extract features of sequential states
                    lstm_cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=self.lstm_size, forget_bias=1, activation=tf.nn.tanh) for _ in range(self.lstm_layers)]
                    cells = tf.nn.rnn_cell.MultiRNNCell(lstm_cells, state_is_tuple=True)
                    rnn_outputs, _ = tf.nn.dynamic_rnn(cell=cells, inputs=reshaped_x, dtype="float32")
                    last_hidden_state = rnn_outputs[:, -1, :]

                    dense_l1 = tf.layers.dense(last_hidden_state, 200, tf.nn.tanh, kernel_regularizer=self.regularizer)
                    dense_l2 = tf.layers.dense(dense_l1, 200, tf.nn.tanh, kernel_regularizer=self.regularizer)

                    # construct dense layers for mean and standard deviation for latent space nomral distribution
                    loc = tf.layers.dense(dense_l2, self.latent_dim, kernel_regularizer=self.regularizer)
                    scale = tf.layers.dense(dense_l2, self.latent_dim, tf.nn.softplus, kernel_regularizer=self.regularizer)

                # in the shape of (batch_size, z_dim)
                return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale), loc, scale

    def build_decoder(self, code):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope('decoder'):
                if not self.using_lstm:
                    z = code  # in the shape of (batch_size, z_dim)
                    dense_l1 = tf.layers.dense(z, 200, tf.nn.tanh, kernel_regularizer=self.regularizer)
                    dense_l2 = tf.layers.dense(dense_l1, 200, tf.nn.tanh, kernel_regularizer=self.regularizer)

                    new_x_loc = tf.layers.dense(dense_l2, self.seq_length * self.state_dim, kernel_regularizer=self.regularizer)
                    new_x_scale = tf.layers.dense(dense_l2, self.seq_length * self.state_dim, tf.nn.softplus, kernel_regularizer=self.regularizer)
                else:
                    z = code
                    dense_l1 = tf.layers.dense(z, 200, tf.nn.tanh, kernel_regularizer=self.regularizer)
                    dense_l2 = tf.layers.dense(dense_l1, 200, tf.nn.tanh, kernel_regularizer=self.regularizer)  # in the shape of (batch_size, layer_2_dim)

                    repeated_l2_output = tf.keras.layers.RepeatVector(self.seq_length)(dense_l2)  # in the shape of (batch_size, seq_length, layer_2_dim)
                    # outputs = tf.keras.layers.LSTM(self.lstm_size, return_sequences=True, dropout=self.dropout)(repeated_l2_output)  # in the shape of (batch_size, seq_length, lstm_dim)
                    lstm_cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=self.lstm_size, forget_bias=1, activation=tf.nn.tanh) for _ in range(self.lstm_layers)]
                    cells = tf.nn.rnn_cell.MultiRNNCell(lstm_cells, state_is_tuple=True)
                    outputs, _ = tf.nn.dynamic_rnn(cell=cells, inputs=repeated_l2_output, dtype="float32")

                    new_x_loc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.state_dim, kernel_regularizer=self.regularizer))(outputs)  # in the shape of (batch_size, seq_length, state_dim)
                    new_x_loc = tf.reshape(new_x_loc, shape=[-1, self.seq_length * self.state_dim])
                    new_x_scale = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.state_dim, activation=tf.nn.softplus, kernel_regularizer=self.regularizer))(outputs)
                    new_x_scale = tf.reshape(new_x_scale, shape=[-1, self.seq_length * self.state_dim])

                # in the shape of (batch_size, x_dim)
                return tfd.MultivariateNormalDiag(new_x_loc, new_x_scale), new_x_loc, new_x_scale

    def build_loss_function(self):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.likelihood = self.new_x_dist.log_prob(self.x)  # in the shape of (batch_size,)
            self.divergence = - 0.5 * tf.math.reduce_sum(1 + tf.math.log(tf.math.square(self.z_sd))
                                                         - tf.math.square(self.z_mean) - tf.math.square(self.z_sd),
                                                         axis=1)

            self.elbo = tf.reduce_mean(self.likelihood - self.kl_factor * self.divergence)
            self.reg_loss = tf.losses.get_regularization_loss(scope=self.name)

            self.trainable_vars = self.get_trainable_variables()
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(-self.elbo + self.reg_loss, var_list=self.trainable_vars)

    def build_summary(self):
        # Mean square error between original x and the mean of reconstructed new_x
        self.mse_x = tf.reduce_mean(tf.reduce_sum(tf.math.square(self.x - self.new_x_mean), axis=1))
        self.average_likelihood = tf.reduce_mean(self.likelihood)
        self.average_kl = tf.reduce_mean(self.divergence)
        self.average_new_x_sd = tf.reduce_mean(self.new_x_sd)

        # create summary to track training process
        average_q_mu = tf.summary.scalar('average_z_mu', tf.reduce_mean(self.z_mean))
        average_q_sigma = tf.summary.scalar('average_z_sigma', tf.reduce_mean(self.z_sd))
        average_new_x_mu = tf.summary.scalar('average_new_x_mu', tf.reduce_mean(self.new_x_mean))
        average_new_x_sigma = tf.summary.scalar('average_new_x_sigma', tf.reduce_mean(self.new_x_sd))
        likelihood_summary = tf.summary.scalar('average_log_likelihood', tf.reduce_mean(self.likelihood))
        kl_summary = tf.summary.scalar('average_kl', tf.reduce_mean(self.divergence))
        elbo_summary = tf.summary.scalar('negative_elbo', -self.elbo)
        mse_x_summary = tf.summary.scalar('MSE_x', self.mse_x)

        # create summary to track testing result
        average_q_mu_test = tf.summary.scalar('average_z_mu_test', tf.reduce_mean(self.z_mean))
        average_q_sigma_test = tf.summary.scalar('average_z_sigma_test', tf.reduce_mean(self.z_sd))
        average_new_x_mu_test = tf.summary.scalar('average_new_x_mu_test', tf.reduce_mean(self.new_x_mean))
        average_new_x_sigma_test = tf.summary.scalar('average_new_x_sigma_test', tf.reduce_mean(self.new_x_sd))
        likelihood_summary_test = tf.summary.scalar('average_log_likelihood_test', tf.reduce_mean(self.likelihood))
        kl_summary_test = tf.summary.scalar('average_kl_test', tf.reduce_mean(self.divergence))
        elbo_summary_test = tf.summary.scalar('negative_elbo_test', -self.elbo)
        mse_x_summary_test = tf.summary.scalar('MSE_x_test', self.mse_x)

        self.summary_op = tf.summary.merge([likelihood_summary, kl_summary, elbo_summary, mse_x_summary])
        self.summary_op_test = tf.summary.merge([likelihood_summary_test, kl_summary_test, elbo_summary_test, mse_x_summary_test])

    def get_training_summary(self, x):
        return tf.get_default_session().run(self.summary_op, feed_dict={self.x: x})

    def get_testing_summary(self, x):
        return tf.get_default_session().run(self.summary_op_test, feed_dict={self.x: x})

    def train(self, x):
        return tf.get_default_session().run(self.optimize, feed_dict={self.x: x})

    def get_latent_state(self, x):
        # return tf.get_default_session().run(self.z_mean, feed_dict={self.x: x})
        return tf.get_default_session().run(self.sampled_z, feed_dict={self.x: x})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

# prepare sequential states and actions from expert demo with new state-action design
def prepare_expert_data_seq(original_expert_obs, original_expert_act,
                            state_max, state_min, action_max, action_min, num_states,
                            test_episodes_id, include_robot_state=False, use_robot_vel=False, seq_length=10, delta_t=0.05):
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

def prepare_input_data():
  dir_note = 'side_view/multi_pos/'
  seq_length = 10
  # testing_episodes_id = [39, 38, 37, 36, 35, 34, 33, 32]
  # testing_episodes_id = [0, 1, 2, 3, 4, 5, 6, 7]
  testing_episodes_id = list(range(8))
  expert_observations = np.genfromtxt('/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/demo_data/' + dir_note + 'expert_observations.csv')
  expert_actions = np.genfromtxt('/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/demo_data/' + dir_note + 'expert_actions.csv')
  expert_actions = np.reshape(expert_actions, newshape=[-1, 4])  # in the shape of (batch_size, action_dimension)

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

  state_max = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0 / 0.05, 1.0 / 0.05, 2.0 / 0.05, 1.0 / 0.05, 1.0 / 0.05, 2.0 / 0.05
                        ])
  state_min = np.array([0.0, 0.0, -1.0, 0.0, 0.0, -1.0,
                        -1.0 / 0.05, -1.0 / 0.05, -2.0 / 0.05, -1.0 / 0.05, -1.0 / 0.05, -2.0 / 0.05
                        ])
  state_dimension = state_max.shape[0]

  # prepare the state and action data of expert demo
  expert_observations_training, expert_actions_training, \
  expert_observations_testing, expert_actions_testing = prepare_expert_data_seq(expert_observations, expert_actions,
                                                                                state_max, state_min,
                                                                                action_max, action_min,
                                                                                state_dimension, testing_episodes_id,
                                                                                state_with_robot_state,
                                                                                action_with_robot_vel,
                                                                                seq_length)

  print("Finish processing observation data")
  print("size of training observation data: {}".format(np.shape(expert_observations_training)))
  print("size of testing observation data: {}".format(np.shape(expert_observations_testing)))

  print("Finish processing action data")
  print("size of training action data: {}".format(np.shape(expert_actions_training)))
  print("size of testing action data: {}".format(np.shape(expert_actions_testing)))

  return expert_observations_training, expert_actions_training, expert_observations_testing, expert_actions_testing


def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


# Interface to LineCollection:
def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0, ax=None):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

    if ax is None:
        ax = plt.gca()

    ax.add_collection(lc)

    return lc


def plot_codes(ax, codes, labels, cmap, fig):
  # set up the colormap range to avoid to be too faint for the beginning color
  # cmap = mpl.cm.Greys(np.linspace(0, 1, 20))
  cmap = mpl.colors.ListedColormap(cmap[5:, :-1])
  data_num = codes.shape[0]

  # draw the scatter points and a line to connect them with fading color
  colorline(codes[:, 0], codes[:, 1], cmap=cmap, linewidth=5, ax=ax)
  im = ax.scatter(codes[:, 0], codes[:, 1], s=100, c=range(data_num), cmap=cmap, alpha=1.0)
  # im = ax.scatter(codes[:, 0], codes[:, 1], s=10, c=labels, alpha=0.8)
  ax.set_aspect('equal')
  ax.set_xlim(codes.min() - .1, codes.max() + .1)
  ax.set_ylim(codes.min() - .1, codes.max() + .1)
  ax.tick_params(
      axis='both', which='both', left=True, bottom=True,
      labelleft=True, labelbottom=True, labelsize=20)

  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="10%", pad=0.3)
  cbar = fig.colorbar(im, ax=ax, cax=cax)
  cbar.ax.tick_params(labelsize=20)


def train_vae(vae, training_data, testing_data, sess, saver, writer, fig_saving_dir, model_save_dir='vae_trained_models/no_lstm/', model_saving_interval=100):
    # prepare training parameters
    epoch_nums = int(1e3)
    plot_per_epoch = 500
    steps_per_trial = 60

    # prepare plot parameters
    # fig, ax = plt.subplots(nrows=3, ncols=int(epoch_nums / plot_per_epoch), figsize=(70, 70))
    fig, ax = plt.subplots(nrows=2, ncols=int(epoch_nums / plot_per_epoch), figsize=(70, 70))
    plot_axe_min = 0
    plot_axe_max = 0
    # for i in range(3):
    #     ax[i, 0].set_ylabel('Height Situation {}'.format(i + 1), fontsize=25, fontweight='black', labelpad=20)

    ''' Training '''
    # divide training set into multiple batches
    samples_num = training_data.shape[0]
    batch_size = 128
    batches = [training_data[i:i + batch_size] for i in range(0, samples_num, batch_size)]  # a list of nparray in the form of (batch_size, seq_length * state_dim)

    for epoch in range(epoch_nums):
        feed = {vae.x: testing_data}
        test_elbo, test_likelihood, test_kl, test_mse_x, test_new_x_sd = sess.run([vae.elbo,
                                                                                   vae.average_likelihood,
                                                                                   vae.average_kl,
                                                                                   vae.mse_x,
                                                                                   vae.average_new_x_sd],
                                                                                   feed)
        summary_test = vae.get_testing_summary(testing_data)
        writer.add_summary(summary_test, epoch)

        # plot the latent space of testing data
        if (epoch + 1) % plot_per_epoch == 0:
            cmap_list = [mpl.cm.Greys(np.linspace(0, 1, 20)),
                         mpl.cm.Blues(np.linspace(0, 1, 20)),
                         mpl.cm.Reds(np.linspace(0, 1, 20)), ]
            for i in range(2):
                # each demo trial has 60 time steps, plot the first test demo
                trial_start_id = i * steps_per_trial
                trial_end_id = (i + 1) * steps_per_trial
                training_observation_to_plot = training_data[trial_start_id:trial_end_id, :]  # in the shape of (trial_length, dim_state)

                feed = {vae.x: training_observation_to_plot}
                code_to_plot = sess.run(vae.sampled_z, feed)
                plot_axe_min = min(plot_axe_min, code_to_plot.min() - .1)
                plot_axe_max = max(plot_axe_max, code_to_plot.max() + .1)
                plot_codes(ax[i, int((epoch + 1) / plot_per_epoch) - 1], code_to_plot, range(steps_per_trial),
                           cmap=cmap_list[i], fig=fig)
                # ax[int((epoch + 1)/plot_per_epoch) - 1, i].title.set_text('Height Situation {}'.format(i+1))

            # ax[int((epoch + 1)/plot_per_epoch) - 1, 0].set_ylabel('Epoch {}'.format(epoch + 1))
            ax[0, int((epoch + 1) / plot_per_epoch) - 1].set_title('Epoch {}'.format(epoch + 1),
                                                                   fontdict={'fontsize': 25, 'fontweight': 'black'},
                                                                   pad=20)
            # print("Finish plotting at epoch {}".format(epoch))

        # print('Epoch', epoch, 'test_negative_elbo', -test_elbo, 'test_likelihood', test_likelihood, 'test_kl', test_kl,
        #       'test_mse_x', test_mse_x, 'test_new_x_sd', test_new_x_sd)

        for batch in batches:
            '''
            sample_indices = np.random.randint(low=0, high=expert_observations_training.shape[0], size=100)
            sampled_train_data = np.take(a=expert_observations_training, indices=sample_indices, axis=0)  # sample training data
            feed = {data: sampled_train_data}
            '''
            vae.train(batch)

        # saving model every 100 epochs
        if (epoch + 1) % model_saving_interval == 0:
            saver.save(sess, model_save_dir + 'model.ckpt', global_step=epoch + 1)
            print('Epoch', epoch + 1, 'test_negative_elbo', -test_elbo, 'test_likelihood', test_likelihood, 'test_kl', test_kl, 'test_mse_x', test_mse_x, 'test_new_x_sd', test_new_x_sd)
            print("Saved model at epoch {}".format(epoch + 1))

        summary_train = vae.get_training_summary(training_data)
        writer.add_summary(summary_train, epoch)
        # print("Finish training epoch {}".format(epoch))
        # print('*****************************************')

    # some format adjustment for subplots
    plt.setp(ax, xlim=(plot_axe_min, plot_axe_max), ylim=(plot_axe_min, plot_axe_max))
    # fig.tight_layout()
    # fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.85,
                        bottom=0.1, top=0.35,
                        wspace=0.25,
                        hspace=0.25)

    if not os.path.exists(fig_saving_dir):
        os.makedirs(fig_saving_dir)
    file_path = os.path.join(fig_saving_dir, 'latent_state_training.png')
    plt.savefig(file_path, dpi=fig.dpi, transparent=False, bbox_inches='tight')
    # plt.savefig('vae-greeting-training.png', dpi=fig.dpi, transparent=False, bbox_inches='tight')

    plt.cla()
    plt.close(fig)


if __name__ == '__main__':
    latent_dim = 2
    using_lstm = True
    seq_length = 10
    state_dim = 12
    reg_factor = 1e-1
    dropout = 0.3
    lstm_size = 64
    kl_factor = 0.5
    lr = 1e-3
    model_saving_interval = 100
    model_save_dir = 'vae_trained_models/no_lstm/'

    vae = Vae('vae', latent_dim=latent_dim, using_lstm=using_lstm, seq_length=seq_length, state_dim=state_dim,
              reg_factor=reg_factor, dropout=dropout, lstm_size=lstm_size, kl_factor=kl_factor, lr=lr)

    # prepare input data
    expert_observations_training, expert_actions_training, \
    expert_observations_testing, expert_actions_testing = prepare_input_data()

    # prepare training parameters
    epoch_nums = int(3e3)
    plot_per_epoch = 500
    steps_per_trial = 60

    # prepare plot parameters
    fig, ax = plt.subplots(nrows=3, ncols=int(epoch_nums / plot_per_epoch), figsize=(70, 70))
    plot_axe_min = 0
    plot_axe_max = 0
    for i in range(3):
        ax[i, 0].set_ylabel('Height Situation {}'.format(i + 1), fontsize=25, fontweight='black', labelpad=20)

    # training
    # with tf.train.MonitoredSession() as sess:
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('tmp/log/' + TIMESTAMP, sess.graph)
        saver = tf.train.Saver(max_to_keep=10)
        sess.run(tf.global_variables_initializer())

        # divide training set into multiple batches
        samples_num = expert_observations_training.shape[0]
        batch_size = 128
        batches = [expert_observations_training[i:i + batch_size] for i in range(0, samples_num,batch_size)]  # a list of nparray in the form of (batch_size, seq_length * state_dim)

        for epoch in range(epoch_nums):
            feed = {vae.x: expert_observations_testing}
            test_elbo, test_likelihood, test_kl, test_mse_x, test_new_x_sd = sess.run([vae.elbo, vae.average_likelihood, vae.average_kl, vae.mse_x, vae.average_new_x_sd], feed)
            summary_test = vae.get_testing_summary(expert_observations_testing)
            writer.add_summary(summary_test, epoch)

            # plot the latent space of testing data
            if (epoch + 1) % plot_per_epoch == 0:
                cmap_list = [mpl.cm.Greys(np.linspace(0, 1, 20)),
                             mpl.cm.Blues(np.linspace(0, 1, 20)),
                             mpl.cm.Reds(np.linspace(0, 1, 20)), ]
                for i in range(3):
                    # each demo trial has 60 time steps, plot the first test demo
                    trial_start_id = i * steps_per_trial
                    trial_end_id = (i + 1) * steps_per_trial
                    training_observation_to_plot = expert_observations_training[trial_start_id:trial_end_id, :]  # in the shape of (trial_length, dim_state)
                    feed = {vae.x: training_observation_to_plot}
                    code_to_plot = sess.run(vae.sampled_z, feed)
                    plot_axe_min = min(plot_axe_min, code_to_plot.min() - .1)
                    plot_axe_max = max(plot_axe_max, code_to_plot.max() + .1)
                    plot_codes(ax[i, int((epoch + 1) / plot_per_epoch) - 1], code_to_plot, range(steps_per_trial), cmap=cmap_list[i], fig=fig)
                    # ax[int((epoch + 1)/plot_per_epoch) - 1, i].title.set_text('Height Situation {}'.format(i+1))

                # ax[int((epoch + 1)/plot_per_epoch) - 1, 0].set_ylabel('Epoch {}'.format(epoch + 1))
                ax[0, int((epoch + 1) / plot_per_epoch) - 1].set_title('Epoch {}'.format(epoch + 1),
                                                                       fontdict={'fontsize': 25, 'fontweight': 'black'},
                                                                       pad=20)
                print("Finish plotting at epoch {}".format(epoch))

            print('Epoch', epoch, 'test_negative_elbo', -test_elbo, 'test_likelihood', test_likelihood, 'test_kl', test_kl, 'test_mse_x', test_mse_x, 'test_new_x_sd', test_new_x_sd)

            for batch in batches:
                '''
                sample_indices = np.random.randint(low=0, high=expert_observations_training.shape[0], size=100)
                sampled_train_data = np.take(a=expert_observations_training, indices=sample_indices, axis=0)  # sample training data
                feed = {data: sampled_train_data}
                '''
                vae.train(batch)

            # saving model every 100 epochs
            if (epoch + 1) % model_saving_interval == 0:
                saver.save(sess, model_save_dir + 'model.ckpt', global_step=epoch + 1)
                print("Saved model at epoch {}".format(epoch))


            summary_train = vae.get_training_summary(expert_observations_training)
            writer.add_summary(summary_train, epoch)
            print("Finish training epoch {}".format(epoch))
            print('*****************************************')

    # some format adjustment for subplots
    plt.setp(ax, xlim=(plot_axe_min, plot_axe_max), ylim=(plot_axe_min, plot_axe_max))
    # fig.tight_layout()
    # fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.85,
                        bottom=0.1, top=0.35,
                        wspace=0.25,
                        hspace=0.25)

    plt.savefig('vae-greeting-training.png', dpi=fig.dpi, transparent=False, bbox_inches='tight')