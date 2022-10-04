import numpy as np
from time import sleep

from environment import GreetingEnv
from agent import Agent
from finite_state_machine import Fsn_policy
from simulation_robot_client import simulation_robot_client


def vary_reaching_delay(agent, env, episode_total_steps, fsn_policy, step_delta_t):
    current_step = 0

    agent.say("We are going to start a new experiment")
    sleep(1.0)

    """ Start "Normal Mode" experiment """
    agent.say("Please do greeting with me for the first time")
    sleep(1.0)
    agent.say("Ready?")
    sleep(1.0)
    agent.say("Start!")
    obs = env.reset()
    last_obs = obs

    while current_step < episode_total_steps:
        if obs is None:
            agent.say("Oh no, I can't see you!")
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
        print("[Normal Mode]: step {}: Finish taking action".format(current_step + 1))

        # states_record.append(np.reshape(state, newshape=list(Policy.ob_space.shape)))
        # actions_record.append(np.array(act))

        # print("################################")

        # update the state for next step
        next_obs, _, _, _ = env.step(act)
        last_obs = obs
        obs = next_obs
        current_step += 1

    agent.say("Okay, we finished the first greeting")
    sleep(1.0)
    agent.say("I am going to home position")
    agent.go_home_pose()
    fsn_policy.reset()
    sleep(1.0)

    print('#########################################')

    """ Start "Diversion Mode" Experiment """
    sleep(1.0)
    agent.say("Okay, Let's continue!")
    sleep(1.0)
    agent.say("Please do greeting with me again")
    sleep(1.0)
    agent.say("Ready?")
    sleep(1.0)
    agent.say("Start!")
    obs = env.reset()
    last_obs = obs
    current_step = 0

    sleep(2.0)
    while current_step < episode_total_steps:
        if obs is None:
            agent.say("Oh no, I can't see you!")
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
        print("[Diversion Mode]: step {}: Finish taking action".format(current_step + 1))

        # states_record.append(np.reshape(state, newshape=list(Policy.ob_space.shape)))
        # actions_record.append(np.array(act))

        # print("################################")

        # update the state for next step
        next_obs, _, _, _ = env.step(act)
        last_obs = obs
        obs = next_obs
        current_step += 1

    agent.say("Great! We finished this section of experiment!")
    sleep(1.0)
    agent.say("Please take a rest and fill out the questionnaire")
    agent.go_home_pose()
    fsn_policy.reset()
    sleep(1.0)


def vary_arm_velocity(agent, env, episode_total_steps, fsn_policy, step_delta_t):
    current_step = 0

    agent.say("We are going to start a new experiment")
    sleep(1.0)

    """ Start "Normal Mode" experiment """
    agent.say("Please do greeting with me for the first time")
    sleep(1.0)
    agent.say("Ready?")
    sleep(1.0)
    agent.say("Start!")
    obs = env.reset()
    last_obs = obs

    while current_step < episode_total_steps:
        if obs is None:
            agent.say("Oh no, I can't see you!")
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
        print("[Normal Mode]: step {}: Finish taking action".format(current_step + 1))

        # states_record.append(np.reshape(state, newshape=list(Policy.ob_space.shape)))
        # actions_record.append(np.array(act))

        # print("################################")

        # update the state for next step
        next_obs, _, _, _ = env.step(act)
        last_obs = obs
        obs = next_obs
        current_step += 1

    agent.say("Okay, we finished the first greeting")
    sleep(1.0)
    agent.say("I am going to home position")
    agent.go_home_pose()
    fsn_policy.reset()
    sleep(1.0)

    print('#########################################')

    """ Start "Diversion Mode" Experiment """
    sleep(1.0)
    agent.say("Okay, Let's continue!")
    sleep(1.0)
    agent.say("Please do greeting with me again")
    sleep(1.0)
    agent.say("Ready?")
    sleep(1.0)
    agent.say("Start!")
    obs = env.reset()
    last_obs = obs
    current_step = 0

    while current_step < episode_total_steps:
        if obs is None:
            agent.say("Oh no, I can't see you!")
            obs = env.reset()
            last_obs = obs
            continue

        act, _ = fsn_policy.act(obs)

        if act is None:
            pass
        elif act[0] == np.inf:
            agent.go_home_pose(blocking=False)
        else:
            agent.take_action(action=act, vel=0.1)

        sleep(step_delta_t)
        # sleep(0.01)
        print("[Diversion Mode]: step {}: Finish taking action".format(current_step + 1))

        # states_record.append(np.reshape(state, newshape=list(Policy.ob_space.shape)))
        # actions_record.append(np.array(act))

        # print("################################")

        # update the state for next step
        next_obs, _, _, _ = env.step(act)
        last_obs = obs
        obs = next_obs
        current_step += 1

    agent.say("Great! We finished this section of experiment!")
    sleep(1.0)
    agent.say("Please take a rest and fill out the questionnaire")
    agent.go_home_pose()
    fsn_policy.reset()
    sleep(1.0)


def vary_arm_jerk(agent, env, episode_total_steps, fsn_policy, step_delta_t):
    current_step = 0

    agent.say("We are going to start a new experiment")
    sleep(1.0)

    """ Start "Normal Mode" experiment """
    agent.say("Please do greeting with me for the first time")
    sleep(1.0)
    agent.say("Ready?")
    sleep(1.0)
    agent.say("Start!")
    obs = env.reset()
    last_obs = obs

    while current_step < episode_total_steps:
        if obs is None:
            agent.say("Oh no, I can't see you!")
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
        print("[Normal Mode]: step {}: Finish taking action".format(current_step + 1))

        # states_record.append(np.reshape(state, newshape=list(Policy.ob_space.shape)))
        # actions_record.append(np.array(act))

        # print("################################")

        # update the state for next step
        next_obs, _, _, _ = env.step(act)
        last_obs = obs
        obs = next_obs
        current_step += 1

    agent.say("Okay, we finished the first greeting")
    sleep(1.0)
    agent.say("I am going to home position")
    agent.go_home_pose()
    fsn_policy.reset()
    sleep(1.0)

    print('#########################################')

    """ Start "Diversion Mode" Experiment """
    sleep(1.0)
    agent.say("Okay, Let's continue!")
    sleep(1.0)
    agent.say("Please do greeting with me again")
    sleep(1.0)
    agent.say("Ready?")
    sleep(1.0)
    agent.say("Start!")
    obs = env.reset()
    last_obs = obs
    current_step = 0

    while current_step < episode_total_steps:
        if obs is None:
            agent.say("Oh no, I can't see you!")
            obs = env.reset()
            last_obs = obs
            continue

        act, _ = fsn_policy.act(obs)

        if act is None:
            pass
        elif act[0] == np.inf:
            agent.go_home_pose(blocking=False)
        else:
            for i in range(4):
                act[i] = act[i] + np.random.uniform(-0.5, 0.5)
            agent.take_action(action=act)

        sleep(step_delta_t)
        # sleep(0.01)
        print("[Diversion Mode]: step {}: Finish taking action".format(current_step + 1))

        # states_record.append(np.reshape(state, newshape=list(Policy.ob_space.shape)))
        # actions_record.append(np.array(act))

        # print("################################")

        # update the state for next step
        next_obs, _, _, _ = env.step(act)
        last_obs = obs
        obs = next_obs
        current_step += 1

    agent.say("Great! We finished this section of experiment!")
    sleep(1.0)
    agent.say("Please take a rest and fill out the questionnaire")
    agent.go_home_pose()
    fsn_policy.reset()
    sleep(1.0)


def main():
    step_delta_t = 0.05
    episode_time_length = 3.0
    episode_total_steps = int(episode_time_length / step_delta_t)
    total_fac_num = 3

    env = GreetingEnv()
    agent = Agent()
    robot_client = simulation_robot_client(using_gui=False, auto_step=True)
    fsn_policy = Fsn_policy(bodyUniqueId=robot_client.bodyUniqueId,
                            endEffectorLinkIndex=robot_client.endEffectorLinkIndex)
    agent.set_stiffness(1.0)
    print("[Robot Client]: Finish setting stiffness to 1.0")

    for factor_id in range(total_fac_num):
        print('[Section {}]: Going to start ...'.format(factor_id + 1))
        if factor_id == 0:
            # start experiment section for hand-reaching delay
            print("[Reaching Delay Experiment]: Start")
            vary_reaching_delay(agent=agent, env=env, episode_total_steps=episode_total_steps, fsn_policy=fsn_policy, step_delta_t=step_delta_t)
            print('[Reaching Delay Experiment]: Finished')
        elif factor_id == 1:
            # start experiment section for robot arm velocity
            print("[Arm Velocity Experiment]: Start")
            vary_arm_velocity(agent=agent, env=env, episode_total_steps=episode_total_steps, fsn_policy=fsn_policy, step_delta_t=step_delta_t)
            print('[Arm Velocity Experiment]: Finished')
        elif factor_id == 2:
            # start experiment section for robot arm jerk
            print("[Arm Jerk Experiment]: Start")
            vary_arm_jerk(agent=agent, env=env, episode_total_steps=episode_total_steps, fsn_policy=fsn_policy, step_delta_t=step_delta_t)
            print('[Arm Jerk Experiment]: Finished')

        print("[Section {}]: Finished".format(factor_id + 1))
        print('*****************************')
        print('******************************')

    agent.say("Fabulous! All experiments are finished! Thanks for your patience and have a nice day!")
    agent.stop()
    print("All experiments finished")


if __name__ == '__main__':
    main()