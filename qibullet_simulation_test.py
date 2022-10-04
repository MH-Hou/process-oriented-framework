import numpy as np
import redis
from time import sleep

from run_behavior_clone import prepare_action_data
from simulation_robot_client import simulation_robot_client

if __name__ == '__main__':
    # data_sub_dir = 'pepper/'
    data_sub_dir = 'side_view/multi_pos/'
    simulation_robot_client = simulation_robot_client(using_gui=True, auto_step=True)

    # load and process demo action data
    expert_actions = np.genfromtxt('demo_data/' + data_sub_dir + 'expert_actions.csv')
    expert_actions = np.reshape(expert_actions, newshape=[-1, 4])  # in the shape of (batch_size, action_dimension)
    training_actions, testing_actions = prepare_action_data(expert_actions, [0])

    print("Finish loading action data")
    print("Training action data shape: {}".format(training_actions.shape))
    print("Testing action data shape: {}".format(testing_actions.shape))

    ''' send command to move joints '''
    num_trials = 10
    for trial in range(num_trials):
        num_actions = testing_actions.shape[0]
        for i in range(num_actions):
            joint_values = testing_actions[i]
            simulation_robot_client.take_action(joint_values)
            print("Trial [{}], action [{}]: Finish taking action".format(trial, i))
            sleep(0.05)

        simulation_robot_client.go_home_pose()
        sleep(5.0)
        print("***********************************************************")