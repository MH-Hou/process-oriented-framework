import numpy as np


def load_robot_factor_trajs():
    experiment_data_dir = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/experiment_data/robot_factor_exp/'
    robot_hand_trajs = []
    human_hand_trajs = []
    total_subject_num = 3
    factor_id = 0 # from 0 to 5
    mode_id = 0 # 0 is normal mode, 1 is diversion mode
    mode_orders = np.genfromtxt(experiment_data_dir + 'mode_orders_list.csv').astype(int)
    mode_orders = np.reshape(mode_orders, newshape=[20, -1, 2])
    steps_per_trial = 61

    for subject_id in range(total_subject_num):
        mode_order = mode_orders[subject_id][factor_id].tolist()
        mode_index = mode_order.index(mode_id) # the index of mode_id value in the mode order list, index 0 corresponds to first 60 steps in trajectory data
        print("[Subject {}, Factor {}]: Mode index for normal mode is {}".format(subject_id+1, factor_id, mode_index))

        robot_hand_traj = np.genfromtxt(experiment_data_dir + 'subject_' + str(subject_id + 1) + '/factor_' + str(factor_id)  + '/robot_hand_position_trajectories.csv')
        robot_hand_traj = robot_hand_traj[steps_per_trial * mode_index : steps_per_trial * (mode_index + 1)] # 2d np array in the form of (steps_per_trial, 3)
        human_hand_traj = np.genfromtxt(experiment_data_dir + 'subject_' + str(subject_id + 1) + '/factor_' + str(factor_id) + '/human_hand_position_trajectories.csv')
        human_hand_traj = human_hand_traj[steps_per_trial * mode_index : steps_per_trial * (mode_index + 1)] # 2d np array in the form of (steps_per_trial, 3)

        robot_hand_trajs.append(robot_hand_traj)
        human_hand_trajs.append(human_hand_traj)

    robot_hand_trajs = np.array(robot_hand_trajs)
    robot_hand_trajs = np.reshape(robot_hand_trajs, newshape=(-1, 3))
    human_hand_trajs = np.array(human_hand_trajs)
    human_hand_trajs = np.reshape(human_hand_trajs, newshape=(-1, 3))

    return robot_hand_trajs, human_hand_trajs


if __name__ == '__main__':
    robot_trajs, human_trajs = load_robot_factor_trajs()

    print("robot trajs:")
    print(human_trajs)
    print("size of robot trajs")
    print(human_trajs.shape)



