import numpy as np
from time import sleep

from finite_state_machine import Fsn_policy
from simulation_robot_client import simulation_robot_client
from utils.utils import load_multiple_subject_data


def prepare_testing_obs_data(subject_ids, testing_episode_ids):
    expert_obs, _ = load_multiple_subject_data(subject_ids) # in the shape of (batch_size, state_dim)
    num_batch = expert_obs.shape[0]
    episode_id = -1
    testing_obs_data = []

    for i in range(num_batch):
        if expert_obs[i][0] == np.inf:
            episode_id += 1
            continue

        if episode_id in testing_episode_ids:
            obs = expert_obs[i].copy() # in the shape of (state_dim,)
            testing_obs_data.append(obs)

    testing_obs_data = np.array(testing_obs_data).astype(dtype=np.float32) # in the shape of (batch_size, state_dim)

    return testing_obs_data




def main():
    robot_client = simulation_robot_client(using_gui=True, auto_step=True)
    subject_ids = [1, 2]
    testing_episode_ids = [0]
    testing_obs = prepare_testing_obs_data(subject_ids=subject_ids, testing_episode_ids=testing_episode_ids)
    fsn_policy = Fsn_policy(bodyUniqueId=robot_client.bodyUniqueId,
                            endEffectorLinkIndex=robot_client.endEffectorLinkIndex)

    step_delta_t = 0.05
    episode_time_length = 3.0
    episode_total_steps = int(episode_time_length / step_delta_t)
    num_iterations = int(testing_obs.shape[0] / episode_total_steps)

    for iteration in range(num_iterations):
        current_step = 0
        print("****************************************************")
        print("Episode {}: Going to start a new episode".format(iteration))
        sleep(1.0)

        while current_step < episode_total_steps:
            state = testing_obs[iteration * episode_total_steps + current_step]
            act = fsn_policy.act(state)

            if act is None:
                pass
            elif act[0] == np.inf:
                robot_client.go_home_pose()
            else:
                robot_client.take_action(act)

            sleep(step_delta_t)
            print("Episode {} step {}: Finish taking action".format(iteration, current_step))
            current_step += 1

        robot_client.go_home_pose()
        fsn_policy.reset()
        sleep(5.0)
        print("Episode {}: Episode finished".format(iteration))

    print("All episodes finished")


if __name__ == '__main__':
    main()