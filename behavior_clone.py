import tensorflow as tf2
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()   #禁止TensorFlow2默认的即时执行模式


class BehavioralCloning:
    def __init__(self, Policy):
        self.Policy = Policy

        # self.lr = tf.placeholder(tf.float32)

        self.expert_actions = tf.placeholder(tf.float32, shape=[None] + list(self.Policy.act_space.shape), name='actions_expert')
        self.expert_actions_testing = tf.placeholder(tf.float32, shape=[None] + list(self.Policy.act_space.shape), name='actions_expert_testing')

        # actions_vec = tf.one_hot(self.actions_expert, depth=self.Policy.act_probs.shape[1], dtype=tf.float32)
        # self.action_log_probs = tf.clip_by_value(self.Policy.action_norm_dist.log_prob(self.expert_actions), 1e-10, 1.0)
        self.action_log_probs = tf.log(tf.clip_by_value(self.Policy.action_norm_dist.prob(self.expert_actions), 1e-10, 1.0))
        self.action_log_probs_testing = tf.log(tf.clip_by_value(self.Policy.action_norm_dist.prob(self.expert_actions_testing), 1e-10, 1.0))

        # loss = tf.reduce_sum(actions_vec * tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), 1)
        # loss = - tf.reduce_mean(loss)
        # loss = tf.reduce_mean(self.expert_actions * self.action_log_probs, axis=1)
        # loss = -tf.reduce_mean(loss)
        # loss = -tf.reduce_mean(tf.reduce_sum(self.action_log_probs, axis=1))

        # reg_term = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # reg_loss = sum(reg_term)

        reg_loss = tf.losses.get_regularization_loss(scope=self.Policy.scope)
        entropy_loss = -tf.reduce_mean(self.action_log_probs)
        loss = entropy_loss + reg_loss
        self.mse_actions = tf.reduce_mean(tf.reduce_sum(tf.math.square(self.expert_actions - self.Policy.action_mean), axis=1))

        reg_loss_summary = tf.summary.scalar('training/BH_loss/reg_loss', reg_loss)
        entropy_loss_summary = tf.summary.scalar('training/BH_loss/cross_entropy', entropy_loss)
        loss_summary = tf.summary.scalar('training/BH_loss/total_loss', loss)
        mse_actions_summary = tf.summary.scalar('training/mse_actions', self.mse_actions)

        entropy_loss_testing = -tf.reduce_mean(self.action_log_probs_testing)
        loss_testing = entropy_loss_testing + reg_loss
        self.mse_actions_testing = tf.reduce_mean(tf.reduce_sum(tf.math.square(self.expert_actions_testing - self.Policy.action_mean), axis=1))

        reg_loss_testing_summary = tf.summary.scalar('testing/BH_loss/reg_loss', reg_loss)
        entropy_loss_testing_summary = tf.summary.scalar('testing/BH_loss/cross_entropy', entropy_loss_testing)
        loss_testing_summary = tf.summary.scalar('testing/BH_loss/total_loss', loss_testing)
        mse_actions_testing_summary = tf.summary.scalar('testing/mse_actions', self.mse_actions_testing)

        average_act_variance = tf.reduce_mean(self.Policy.action_variance)
        variance_summary = tf.summary.scalar('average_action_variance', average_act_variance)

        self.optimizer = tf.train.AdamOptimizer(1e-3)
        # self.optimizer = tf.train.GradientDescentOptimizer(1e-3)
        self.trainable_vars = self.Policy.get_trainable_variables()
        self.train_op = self.optimizer.minimize(loss, var_list=self.trainable_vars)

        # self.merged = tf.summary.merge_all()
        self.merged = tf.summary.merge([reg_loss_summary, entropy_loss_summary, loss_summary, variance_summary, mse_actions_summary])
        self.merged_testing = tf.summary.merge([reg_loss_testing_summary, entropy_loss_testing_summary, loss_testing_summary, mse_actions_testing_summary])


    def train(self, obs, actions):
        return tf.get_default_session().run(self.train_op, feed_dict={self.Policy.obs: obs,
                                                                      self.expert_actions: actions})

    def get_summary(self, obs, actions):
        return tf.get_default_session().run(self.merged, feed_dict={self.Policy.obs: obs,
                                                                    self.expert_actions: actions})

    def get_summary_testing(self, obs, actions):
        return tf.get_default_session().run(self.merged_testing, feed_dict={self.Policy.obs: obs,
                                                                    self.expert_actions_testing: actions})