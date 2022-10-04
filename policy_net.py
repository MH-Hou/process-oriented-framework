import tensorflow as tf2
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()   #禁止TensorFlow2默认的即时执行模式
import numpy as np
import gym
from gym import spaces


class Policy_net:
    def __init__(self, name: str, using_latent_state=False, using_lstm=True, latent_dim=2, reg_factor = 1e-3, seq_length=10, lstm_layers=1):
        self.using_latent_state = using_latent_state
        self.using_lstm = using_lstm
        self.latent_dim = latent_dim

        self.construct_env()

        self.num_actions = self.act_space.shape[0]  # action_space.shape is a tuple, i.e., (num_action, )
        self.action_min = self.act_space.low  # action_space.low is a np.array in the shape of (num_action, )
        self.action_max = self.act_space.high

        with tf.variable_scope(name):
            self.expert_actions = tf.placeholder(tf.float32, shape=[None] + list(self.act_space.shape), name='actions_expert')
            self.regularizer = tf2.keras.regularizers.l2(reg_factor)

            # self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.ob_space.shape), name='obs')
            if self.using_latent_state:
                self.obs = tf.placeholder(dtype=tf.float32, shape=[None, self.latent_dim], name='latent_obs')
                outputs = self.obs
            else:
                self.obs = tf.placeholder(dtype=tf.float32, shape=[None, seq_length * self.ob_space.shape[0]], name='obs')

                # # preprocessing of input sequential obs
                # self.processed_obs = tf.layers.dense(inputs=self.obs, units=seq_length * self.ob_space.shape[0], activation=tf.nn.tanh)

                if self.using_lstm:
                    # construct LSTM inputs
                    self.rnn_inputs = tf.reshape(self.obs, shape=[-1, seq_length, self.ob_space.shape[0]])

                    # construct the LSTM layers to extract features of sequential states
                    lstm_cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=64, forget_bias=1) for _ in range(lstm_layers)]
                    cells = tf.nn.rnn_cell.MultiRNNCell(lstm_cells, state_is_tuple=True)
                    rnn_outputs, _ = tf.nn.dynamic_rnn(cell=cells, inputs=self.rnn_inputs, dtype="float32")
                    # lstm_cells = [tf2.keras.layers.LSTMCell(64, kernel_regularizer= self.regularizer) for _ in range(lstm_layers)]
                    # rnn_layer = tf2.keras.layers.RNN(lstm_cells,
                    #                                 return_sequences=False,
                    #                                 return_state=False)
                    # rnn_outputs = rnn_layer(inputs=self.rnn_inputs)

                    outputs = rnn_outputs[:, -1, :]
                    # rnn_outputs = outputs[:,-1,:]
                    # rnn_outputs = tf.layers.dense(inputs=outputs[:,-1,:], units=128, activation=tf.tanh)
                else:
                    outputs = self.obs


            with tf.variable_scope('policy_net'):
                layer_1 = tf.layers.dense(inputs=outputs, units=256, activation=tf.nn.relu, kernel_regularizer=self.regularizer)
                layer_2 = tf.layers.dense(inputs=layer_1, units=256, activation=tf.nn.relu, kernel_regularizer=self.regularizer)
                layer_3 = tf.layers.dense(inputs=layer_2, units=256, activation=tf.nn.relu,kernel_regularizer=self.regularizer)
                # layer_4 = tf.layers.dense(inputs=layer_3, units=256, activation=tf.nn.relu,kernel_regularizer=self.regularizer)

                self.action_mean = ((self.action_max - self.action_min) / 2) * \
                                    tf.layers.dense(inputs=layer_3,
                                                    units=self.num_actions,
                                                    activation=tf.nn.tanh,
                                                    name='mean',
                                                    kernel_regularizer=self.regularizer
                                                    ) + \
                                    (self.action_max + self.action_min) / 2

                self.action_variance = tf.layers.dense(inputs=layer_3,
                                                       units=self.num_actions,
                                                       activation=tf.nn.softplus,
                                                       kernel_initializer=tf.initializers.random_uniform(0.1, 1),
                                                       kernel_regularizer=self.regularizer,
                                                       bias_initializer=tf.constant_initializer(0.1),
                                                       name='variance'
                                                      )

                self.action_norm_dist = tf.distributions.Normal(loc=self.action_mean, scale=self.action_variance, name='normal')
                # self.action_norm_dist = tf.distributions.Normal(loc=self.action_mean, scale=0.15, name='normal')

            with tf.variable_scope('value_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=256, activation=tf.nn.relu)
                # layer_2 = tf.layers.dense(inputs=layer_1, units=256, activation=tf.nn.relu)
                self.v_preds = tf.layers.dense(inputs=layer_1, units=1, activation=None)

            self.act_stochastic = tf.squeeze(self.action_norm_dist.sample(1), axis=0) # in the shape of (batch_size, num_actions) -> (1, num_actions)
            self.act_deterministic = tf.squeeze(self.action_mean, axis=0) # in the shape of (num_actions,)

            expert_action_prob = self.action_norm_dist.prob(self.expert_actions)
            self.KL_divergence = tf.reduce_mean(-tf.log(tf.clip_by_value(expert_action_prob, 1e-10, 1.0)))

            self.scope = tf.get_variable_scope().name

    def construct_env(self):
        # every single state (i.e., position of human hand) is in the form of (x, y, z)
        # state = (human_state, human_vel, robot_state)
        # self.ob_space = spaces.Box(low=np.array([0.0, 0.0, -1.0, 0.0, 0.0, -1.0,
        #                                          -1.0/0.05, -1.0/0.05, -2.0/0.05, -1.0/0.05, -1.0/0.05, -2.0/0.05,
        #                                          -2.0857, -1.3265, -2.0857, 0.0349]),
        #                            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        #                                           1.0/0.05, 1.0/0.05, 2.0/0.05, 1.0/0.05, 1.0/0.05, 2.0/0.05,
        #                                           2.0857, 0.3142, 2.0857, 1.5446]),
        #                            dtype=np.float32)

        if self.using_latent_state:
            self.latent_ob_space = spaces.Box(low=np.array([-10.0, -10.0]),
                                       high=np.array([10.0, 10.0]),
                                       dtype=np.float32)

        self.ob_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                                                 -2.0 / 0.05, -2.0 / 0.05, -2.0 / 0.05, -2.0 / 0.05, -2.0 / 0.05,
                                                 -2.0 / 0.05,
                                                 ]),
                                   high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                  2.0 / 0.05, 2.0 / 0.05, 2.0 / 0.05, 2.0 / 0.05, 2.0 / 0.05,
                                                  2.0 / 0.05,
                                                  ]),
                                   dtype=np.float32)

        # self.ob_space = spaces.Box(low=np.array([0.0, 0.0, -1.0, 0.0, 0.0, -1.0,
        #                                          -1.0 / 0.05, -1.0 / 0.05, -2.0 / 0.05, -1.0 / 0.05, -1.0 / 0.05, -2.0 / 0.05,
        #                                          ]),
        #                            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        #                                           1.0 / 0.05, 1.0 / 0.05, 2.0 / 0.05, 1.0 / 0.05, 1.0 / 0.05, 2.0 / 0.05,
        #                                           ]),
        #                            dtype=np.float32)

        # every single action (i.e., joint values of robot arms) is in the form of (j1,j2,j3,j4,j5)
        # new form of action would be the angular velocity for each joint
        # self.act_space = spaces.Box(low=np.array([-(2.0857 - (-2.0857))/0.05, -(0.3142 - (-1.3265))/0.05, -(2.0857 - (-2.0857))/0.05, -(1.5446 - 0.0349)/0.05]),
        #                             high=np.array([(2.0857 - (-2.0857))/0.05, (0.3142 - (-1.3265))/0.05, (2.0857 - (-2.0857))/0.05, (1.5446 - 0.0349)/0.05]),
        #                             dtype=np.float32)

        self.act_space = spaces.Box(low=np.array([-2.0857, -1.5620, -2.0857, 0.0087, -1.8239, -0.5149]),
                                    high=np.array([2.0857,  -0.0087, 2.0857, 1.5620, 1.8239, 0.5149]),
                                    dtype=np.float32)

        # self.act_space = spaces.Box(low=np.array([-2.0857, -1.3265, -2.0857, 0.0349]),
        #                             high=np.array([2.0857, 0.3142, 2.0857, 1.5446]),
        #                             dtype=np.float32)

        # self.act_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0]),
        #                             high=np.array([1.0, 1.0, 1.0, 1.0]),
        #                             dtype=np.float32)

    def act(self, obs, stochastic=True):
        if stochastic:
            v_preds = tf.get_default_session().run(self.v_preds, feed_dict={self.obs: obs})
            action = tf.get_default_session().run(self.act_stochastic, feed_dict={self.obs: obs})[0]
            action = np.clip(action, self.action_min, self.action_max) # in the shape of (num_actions,)
            # print("action mean: {}, action variance: {}".format(tf.get_default_session().run(self.action_mean, feed_dict={self.obs: obs}),
            #                                                     tf.get_default_session().run(self.action_variance, feed_dict={self.obs: obs})))

            return action, v_preds
        else:
            v_preds = tf.get_default_session().run(self.v_preds, feed_dict={self.obs: obs})
            action = tf.get_default_session().run(self.act_deterministic, feed_dict={self.obs: obs})
            action = np.clip(action, self.action_min, self.action_max)
            # print("action mean: {}, action variance: {}".format(tf.get_default_session().run(self.action_mean, feed_dict={self.obs: obs}),
            #                                                     tf.get_default_session().run(self.action_variance, feed_dict={self.obs: obs})))

            return action, v_preds

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.action_norm_dist, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def calculate_KL_divergence(self, expert_obs, expert_acts):
        # agent_prob_dist = tf.get_default_session().run(tf.convert_to_tensor(self.action_norm_dist), feed_dict={self.obs: expert_obs})
        # agent_action_probs = agent_prob_dist.prob(expert_acts)
        # KL_divergence = tf.reduce_mean(-tf.log(tf.clip_by_value(agent_action_probs, 1e-10, 1.0)))
        return tf.get_default_session().run(self.KL_divergence, feed_dict={self.obs: expert_obs, self.expert_actions: expert_acts})