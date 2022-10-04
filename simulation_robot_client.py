import pybullet as p
from qibullet import SimulationManager
from time import sleep

import redis


class simulation_robot_client:
    def __init__(self, using_gui=False, auto_step=False):
        self.using_gui = using_gui
        self.auto_step = auto_step
        self.build_simulation_env(self.using_gui, self.auto_step)
        # self.joint_names = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "KneePitch", "HipPitch", "HipRoll"]
        self.joint_names = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", "HipRoll"]
        self.fractionMaxSpeed = 1.0

        self.r = redis.Redis(host='localhost', port=6379, db=0)
        self.sub_joint = self.r.pubsub()
        self.listening_thread_joint = None

        # self.initialize_subscribers()
        # self.start_to_listen()

    def build_simulation_env(self, using_gui, auto_step):
        self.simulation_manager = SimulationManager()
        self.client = self.simulation_manager.launchSimulation(gui=using_gui, auto_step=auto_step)
        self.pepper = self.simulation_manager.spawnPepper(self.client, translation=[0.0, 0.0, 0.0], spawn_ground_plane=True)

        self.pepper.goToPosture("Stand", 1.0)
        if not auto_step:
            self.simulation_manager.stepSimulation(self.client)

        # Get Robot Info
        self.bodyUniqueId = self.pepper.getRobotModel()
        # self.endEffectorLinkIndex = self.pepper.link_dict["r_wrist"].getIndex()
        self.endEffectorLinkIndex = self.pepper.link_dict["r_hand"].getIndex()

        print("Finish building simulation environments")

    def take_action(self, joint_values):
        for i in range(len(joint_values)):
            joint_name = self.joint_names[i]
            joint_value = joint_values[i].item()
            self.pepper.setAngles(joint_name, joint_value, self.fractionMaxSpeed)

        if not self.auto_step:
            self.simulation_manager.stepSimulation(self.client)
        # print('[Simulation robot client]: Finish taking action')

    def go_home_pose(self):
        self.pepper.goToPosture("Stand", 1.0)

    def initialize_subscribers(self):
        self.sub_joint.subscribe(**{'target_joint_values': self.joint_message_handler})
        print("Finish initializing subscribers")

    def start_to_listen(self):
        self.listening_thread_joint = self.sub_joint.run_in_thread(sleep_time=0.01)
        print("Start to listen to joint messages...")

    def stop(self):
        self.listening_thread_joint.stop()

    def joint_message_handler(self, message):
        print("[Robot Client]: Receive joint command data: {}".format(message['data']))

        # Send joint command
        joint_values = [float(i) for i in message['data'].split()]
        print("type of joint values: {}".format(type(joint_values)))
        print("joint values: {}".format(joint_values))
        print("length of joint values: {}".format(len(joint_values)))
        for i in range(len(joint_values)):
            joint_name = self.joint_names[i]
            joint_value = joint_values[i]
            print("joint name: {}".format(joint_name))
            print("joint value: {}".format(joint_value))
            self.pepper.setAngles(joint_name, joint_value, 1.0)
            print("Finish set joint [{}] as value [{}]".format(joint_name, joint_value))

        self.simulation_manager.stepSimulation(self.client)

        print("[Robot Client]: Finish sending joint command")

