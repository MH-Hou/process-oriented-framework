import numpy as np
from time import sleep

from environment import GreetingEnv
from agent import Agent
from finite_state_machine import Fsn_policy
from simulation_robot_client import simulation_robot_client
from run_test import argparser

def main(args):
    subject_ids = [1]

    step_delta_t = 0.05
    episode_time_length = 3.0
    episode_total_steps = int(episode_time_length / step_delta_t)

    env = GreetingEnv()
    agent = Agent()
    robot_client = simulation_robot_client(using_gui=False, auto_step=True)
    fsn_policy = Fsn_policy(bodyUniqueId=robot_client.bodyUniqueId,
                            endEffectorLinkIndex=robot_client.endEffectorLinkIndex)
    agent.set_stiffness(1.0)
    print("[Robot Client]: Finish setting stiffness to 1.0")

    for iteration in range(args.iteration):
        current_step = 0
        latest_obs_seq = []
        agent.say("Going to start new episode " + str(iteration + 1))
        print("****************************************************")
        print("Episode {}: Going to start a new episode".format(iteration + 1))
        sleep(1.0)
        agent.say("Please prepare to do greeting with me")
        sleep(1.0)
        agent.say("Start")
        obs = env.reset()
        last_obs = obs

        while current_step < episode_total_steps:
            if obs is None:
                agent.say("Failed to detect human")
                print("Episode {} step {}: No human detected or fail to connect to the camera".format(iteration + 1, current_step))
                obs = env.reset()
                last_obs = obs
                continue

            act, _ = fsn_policy.act(obs)

            if act is None:
                pass
            elif act[0] == np.inf:
                agent.go_home_pose(blocking=False)
            else:
                agent.take_action(act)


            sleep(step_delta_t)
            # sleep(0.01)
            print("Episode {} step {}: Finish taking action".format(iteration + 1, current_step))

            # states_record.append(np.reshape(state, newshape=list(Policy.ob_space.shape)))
            # actions_record.append(np.array(act))

            # print("################################")

            # update the state for next step
            next_obs, _, _, _ = env.step(act)
            last_obs = obs
            obs = next_obs
            current_step += 1

        agent.say("Finished episode " + str(iteration + 1))
        sleep(1.0)
        agent.say("I am going to home position")
        agent.go_home_pose()
        print("Episode {}: Episode finished".format(iteration + 1))
        fsn_policy.reset()

        # save_demo(states_record, demo_path_real_obs)
        # save_demo(actions_record, demo_path_real_act)



    agent.say("Great! Our testing is all finished! Thank you so much for your patience!")
    agent.stop()
    print("All episodes finished")



if __name__ == '__main__':
    args = argparser()
    main(args)

