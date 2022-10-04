import redis
import numpy as np
from time import sleep

class Agent():
    def __init__(self, puppeteer_mode=False):
        # Redis-based client to send command to and retrieve data from the robot side
        self.r = redis.Redis(host='localhost', port=6379, db=0)

        self.puppeteer_mode = puppeteer_mode

        self.whether_finished_say = False
        self.whether_finished_go_home_pose = False
        self.whether_finished_set_stiffness = False
        self.whether_received_robot_data = False
        self.whether_finished_take_action = False
        self.whether_finished_set_led = False
        self.whether_received_hand_pose = False
        self.whether_finished_set_stiffness_whole_body = False
        self.whether_received_robot_data_whole_body = False
        self.whether_finished_take_action_wrist = False

        self.latest_robot_data = None
        self.latest_robot_data_whole_body = None
        self.latest_hand_pose = None

        self.sub_finished_say = self.r.pubsub()
        self.sub_finished_go_home_pose = self.r.pubsub()
        self.sub_finished_set_stiffness = self.r.pubsub()
        self.sub_robot_data = self.r.pubsub()
        self.sub_finished_set_led = self.r.pubsub()
        self.sub_received_hand_pose = self.r.pubsub()
        self.sub_finished_set_stiffness_whole_body = self.r.pubsub()
        self.sub_robot_data_whole_body = self.r.pubsub()
        # self.sub_finished_take_action = self.r.pubsub()

        # subscribers for puppeteer
        self.sub_finished_go_home_pose_puppeteer = self.r.pubsub()
        self.sub_finished_set_stiffness_puppeteer = self.r.pubsub()
        self.sub_robot_data_puppeteer = self.r.pubsub()
        self.sub_finished_set_led_puppeteer = self.r.pubsub()
        self.sub_received_hand_pose_puppeteer = self.r.pubsub()
        self.sub_finished_set_stiffness_whole_body_puppeteer = self.r.pubsub()
        self.sub_robot_data_whole_body_puppeteer = self.r.pubsub()

        # subscribers for performer
        self.sub_finished_say_performer = self.r.pubsub()
        self.sub_finished_go_home_pose_performer = self.r.pubsub()
        self.sub_finished_set_stiffness_performer = self.r.pubsub()
        self.sub_robot_data_performer = self.r.pubsub()
        self.sub_received_hand_pose_performer = self.r.pubsub()
        self.sub_robot_data_whole_body_performer = self.r.pubsub()

        self.listening_thread_finished_say = None
        self.listening_thread_finished_go_home_pose = None
        self.listening_thread_finished_set_stiffness = None
        self.listening_thread_robot_data = None
        self.listening_thread_finished_take_action = None
        self.listening_thread_finished_set_led = None
        self.listening_thread_received_hand_pose = None
        self.listening_thread_finished_set_stiffness_whole_body = None
        self.listening_thread_robot_data_whole_body = None

        # flags for puppeteer
        self.listening_thread_finished_go_home_pose_puppeteer = None
        self.listening_thread_finished_set_stiffness_puppeteer = None
        self.listening_thread_robot_data_puppeteer = None
        self.listening_thread_finished_set_led_puppeteer = None
        self.listening_thread_received_hand_pose_puppeteer = None
        self.listening_thread_finished_set_stiffness_whole_body_puppeteer = None
        self.listening_thread_robot_data_whole_body_puppeteer = None

        # flags for performer
        self.listening_thread_finished_say_performer = None
        self.listening_thread_finished_go_home_pose_performer = None
        self.listening_thread_finished_set_stiffness_performer = None
        self.listening_thread_robot_data_performer = None
        self.listening_thread_received_hand_pose_performer = None
        self.listening_thread_robot_data_whole_body_performer = None

        self.start_to_listen()

    def initialize_subscribers(self):
        if not self.puppeteer_mode:
            self.sub_finished_say.subscribe(**{'finished_say': self.finished_say_message_handler})
            self.sub_finished_go_home_pose.subscribe(**{'finished_go_home_pose': self.finished_go_home_pose_message_handler})
            self.sub_finished_set_stiffness.subscribe(**{'finished_set_stiffness': self.finished_set_stiffness_message_handler})
            self.sub_robot_data.subscribe(**{'latest_joint_values': self.robot_data_message_handler})
            self.sub_finished_set_led.subscribe(**{'finished_set_led': self.finished_set_led_message_handler})
            self.sub_received_hand_pose.subscribe(**{'latest_hand_pose': self.received_hand_pose_message_handler})
            self.sub_finished_set_stiffness_whole_body.subscribe(**{'finished_set_stiffness_whole_body': self.finished_set_stiffness_whole_body_message_handler})
            self.sub_robot_data_whole_body.subscribe(**{'latest_joint_values_whole_body': self.robot_data_whole_body_message_handler})
            # self.sub_finished_take_action.subscribe(**{'finished_take_action': self.finished_take_action_message_handler})
        else:
            # subscribers for puppeteer
            self.sub_finished_go_home_pose_puppeteer.subscribe(**{'puppeteer_finished_go_home_pose': self.finished_go_home_pose_message_handler})
            self.sub_finished_set_stiffness_puppeteer.subscribe(**{'puppeteer_finished_set_stiffness': self.finished_set_stiffness_message_handler})
            self.sub_robot_data_puppeteer.subscribe(**{'puppeteer_latest_joint_values': self.robot_data_message_handler})
            self.sub_finished_set_led_puppeteer.subscribe(**{'puppeteer_finished_set_led': self.finished_set_led_message_handler})
            self.sub_received_hand_pose_puppeteer.subscribe(**{'puppeteer_latest_hand_pose': self.received_hand_pose_message_handler})
            self.sub_finished_set_stiffness_whole_body_puppeteer.subscribe(**{'puppeteer_finished_set_stiffness_whole_body': self.finished_set_stiffness_whole_body_message_handler})
            self.sub_robot_data_whole_body_puppeteer.subscribe(**{'puppeteer_latest_joint_values_whole_body': self.robot_data_whole_body_message_handler})

            # subscribers for performer
            self.sub_finished_say_performer.subscribe(**{'performer_finished_say': self.finished_say_message_handler})
            self.sub_finished_go_home_pose_performer.subscribe(**{'performer_finished_go_home_pose': self.finished_go_home_pose_message_handler})
            self.sub_finished_set_stiffness_performer.subscribe(**{'performer_finished_set_stiffness': self.finished_set_stiffness_message_handler})
            self.sub_robot_data_performer.subscribe(**{'performer_latest_joint_values': self.robot_data_message_handler})
            self.sub_received_hand_pose_performer.subscribe(**{'performer_latest_hand_pose': self.received_hand_pose_message_handler})
            self.sub_robot_data_whole_body_performer.subscribe(**{'performer_latest_joint_values_whole_body': self.robot_data_whole_body_message_handler})

    def start_to_listen(self):
        self.initialize_subscribers()

        if not self.puppeteer_mode:
            self.listening_thread_finished_say = self.sub_finished_say.run_in_thread(sleep_time=0.01)
            self.listening_thread_finished_go_home_pose = self.sub_finished_go_home_pose.run_in_thread(sleep_time=0.01)
            self.listening_thread_finished_set_stiffness = self.sub_finished_set_stiffness.run_in_thread(sleep_time=0.01)
            self.listening_thread_robot_data = self.sub_robot_data.run_in_thread(sleep_time=0.01)
            self.listening_thread_finished_set_led = self.sub_finished_set_led.run_in_thread(sleep_time=0.01)
            self.listening_thread_received_hand_pose = self.sub_received_hand_pose.run_in_thread(sleep_time=0.01)
            self.listening_thread_finished_set_stiffness_whole_body = self.sub_finished_set_stiffness_whole_body.run_in_thread(sleep_time=0.01)
            self.listening_thread_robot_data_whole_body = self.sub_robot_data_whole_body.run_in_thread(sleep_time=0.01)
            # self.listening_thread_finished_take_action = self.sub_finished_take_action.run_in_thread(sleep_time=0.01)
        else:
            # puppeteer threads
            self.listening_thread_finished_go_home_pose_puppeteer = self.sub_finished_go_home_pose_puppeteer.run_in_thread(sleep_time=0.01)
            self.listening_thread_finished_set_stiffness_puppeteer = self.sub_finished_set_stiffness_puppeteer.run_in_thread(sleep_time=0.01)
            self.listening_thread_robot_data_puppeteer = self.sub_robot_data_puppeteer.run_in_thread(sleep_time=0.01)
            self.listening_thread_finished_set_led_puppeteer = self.sub_finished_set_led_puppeteer.run_in_thread(sleep_time=0.01)
            self.listening_thread_received_hand_pose_puppeteer = self.sub_received_hand_pose_puppeteer.run_in_thread(sleep_time=0.01)
            self.listening_thread_finished_set_stiffness_whole_body_puppeteer = self.sub_finished_set_stiffness_whole_body_puppeteer.run_in_thread(sleep_time=0.01)
            self.listening_thread_robot_data_whole_body_puppeteer = self.sub_robot_data_whole_body_puppeteer.run_in_thread(sleep_time=0.01)

            # performer threads
            self.listening_thread_finished_say_performer = self.sub_finished_say_performer.run_in_thread(sleep_time=0.01)
            self.listening_thread_finished_go_home_pose_performer = self.sub_finished_go_home_pose_performer.run_in_thread(sleep_time=0.01)
            self.listening_thread_finished_set_stiffness_performer = self.sub_finished_set_stiffness_performer.run_in_thread(sleep_time=0.01)
            self.listening_thread_robot_data_performer = self.sub_robot_data_performer.run_in_thread(sleep_time=0.01)
            self.listening_thread_received_hand_pose_performer = self.sub_received_hand_pose_performer.run_in_thread(sleep_time=0.01)
            self.listening_thread_robot_data_whole_body_performer = self.sub_robot_data_whole_body_performer.run_in_thread(sleep_time=0.01)

    def stop(self):
        if not self.puppeteer_mode:
            self.listening_thread_finished_say.stop()
            self.listening_thread_finished_go_home_pose.stop()
            self.listening_thread_finished_set_stiffness.stop()
            self.listening_thread_robot_data.stop()
            self.listening_thread_finished_set_led.stop()
            self.listening_thread_received_hand_pose.stop()
            self.listening_thread_finished_set_stiffness_whole_body.stop()
            self.listening_thread_robot_data_whole_body.stop()
        else:
            # stop puppeteer threads
            self.listening_thread_finished_go_home_pose_puppeteer.stop()
            self.listening_thread_finished_set_stiffness_puppeteer.stop()
            self.listening_thread_robot_data_puppeteer.stop()
            self.listening_thread_finished_set_led_puppeteer.stop()
            self.listening_thread_received_hand_pose_puppeteer.stop()
            self.listening_thread_finished_set_stiffness_whole_body_puppeteer.stop()
            self.listening_thread_robot_data_whole_body_puppeteer.stop()

            # stop performer threads
            self.listening_thread_finished_say_performer.stop()
            self.listening_thread_finished_go_home_pose_performer.stop()
            self.listening_thread_finished_set_stiffness_performer.stop()
            self.listening_thread_robot_data_performer.stop()
            self.listening_thread_received_hand_pose_performer.stop()
            self.listening_thread_robot_data_whole_body_performer.stop()

    ''' Blocking Publishing Services '''
    # send command to say something and wait until it finishes
    def say(self, message, role=None):
        if role is None:
            self.r.publish('say', message)
        else:
            self.r.publish(role + '_' + 'say', message)

        while not self.whether_finished_say:
            pass

        self.whether_finished_say = False
        # print("[Agent Say]: Finished say service")

    # send command to go home pose and wait until it finishes
    def go_home_pose(self, blocking=True, role=None):
        if role is None:
            self.r.publish('go_home_pose', str(0))
        else:
            self.r.publish(role + '_' + 'go_home_pose', str(0))

        if blocking:
            while not self.whether_finished_go_home_pose:
                pass

        self.whether_finished_go_home_pose = False
        # print("[Agent Go-Home-Pose]: Finished go-home-pose service")

    # send command to set stiffness and wait until it finishes
    def set_stiffness(self, stiffness, role=None):
        if role is None:
            self.r.publish('set_stiffness', str(stiffness))
        else:
            self.r.publish(role + '_' + 'set_stiffness', str(stiffness))

        while not self.whether_finished_set_stiffness:
            pass

        self.whether_finished_set_stiffness = False
        # print("[Agent Set-Stiffness]: Finished set-stiffness service")

    def set_stiffness_whole_body(self, stiffness, role=None):
        if role is None:
            self.r.publish('set_stiffness_whole_body', str(stiffness))
        else:
            self.r.publish(role + '_' + 'set_stiffness_whole_body', str(stiffness))

        while not self.whether_finished_set_stiffness_whole_body:
            pass

        self.whether_finished_set_stiffness_whole_body = False

    # send command to retrieve joint values and wait until data are received
    def collect_robot_data(self, role=None):
        # send command to robot for retrieving the current joint values
        if role is None:
            self.r.publish('retrieve_joint_values', str(0))
        else:
            self.r.publish(role + '_' + 'retrieve_joint_values', str(0))
        # print("[Collect Robot Data]: Just sent command to retrieve data, going to wait ...")

        while not self.whether_received_robot_data:
            # sleep(0.01)
            pass

        # print("[Collect Robot Data]: Received robot data")
        self.whether_received_robot_data = False
        # print("[Agent Collect-Data]: Finished collect-data service")

        return self.latest_robot_data

    def collect_robot_data_whole_body(self, role=None):
        if role is None:
            self.r.publish('retrieve_joint_values_whole_body', str(0))
        else:
            self.r.publish(role + '_' + 'retrieve_joint_values_whole_body', str(0))

        while not self.whether_received_robot_data_whole_body:
            # sleep(0.01)
            pass

        # print("[Collect Robot Data]: Received robot data")
        self.whether_received_robot_data_whole_body = False
        # print("[Agent Collect-Data]: Finished collect-data service")

        return self.latest_robot_data_whole_body

    def set_led(self, intensity=1, role=None, led_name='ear'):
        if led_name == 'ear':
            led_id = 0
        elif led_name == 'ear_front':
            led_id = 1
        elif led_name == 'ear_back':
            led_id = 2
        else:
            led_id = 3

        message = str(led_id) + ' '

        if role is None:
            self.r.publish('set_led', message + str(intensity))
        else:
            self.r.publish(role + '_' + 'set_led', message + str(intensity))

        while not self.whether_finished_set_led:
            pass

        self.whether_finished_set_led = False

    def get_hand_pose(self, role=None):
        if role is None:
            self.r.publish('retrieve_hand_pose', str(0))
        else:
            self.r.publish(role + '_' + 'retrieve_hand_pose', str(0))
        # print("[Collect Robot Data]: Just sent command to retrieve data, going to wait ...")

        while not self.whether_received_hand_pose:
            # sleep(0.01)
            pass

        # print("[Collect Robot Data]: Received robot data")
        self.whether_received_hand_pose = False
        # print("[Agent Collect-Data]: Finished collect-data service")

        return self.latest_hand_pose



    ''' Non-Blocking Publishing Services '''
    # send command to move joints and won't wait for it finishes
    def take_action(self, action, role=None, vel=None):
        joint_values_message = ''

        # action input is in the form of 1d np.array of shape (action_dimension,)
        for joint_value in action:
            joint_values_message += str(joint_value) + ' '

        if vel is None:
            vel = -1.0
            joint_values_message += str(vel) + ' '
        else:
            joint_values_message += str(vel) + ' '

        if role is None:
            self.r.publish('target_joint_values', joint_values_message)
        else:
            self.r.publish(role + '_' + 'target_joint_values', joint_values_message)
        # print("[Agent joint-motion]: Finished sending joint command")

        """while not self.whether_finished_take_action:
            pass

        self.whether_finished_take_action = False"""

    def take_action_wrist(self, action, role=None):
        joint_values_message = str(action)

        if role is None:
            self.r.publish('target_joint_values_wrist', joint_values_message)
        else:
            self.r.publish(role + '_' + 'target_joint_values_wrist', joint_values_message)

    ''' Listening Services '''
    def finished_say_message_handler(self, message):
        self.whether_finished_say = True
        # print("[Agent Say Message Handler]: Received the finished say message")

    def finished_go_home_pose_message_handler(self, message):
        self.whether_finished_go_home_pose = True
        # print("[Agent Go-Home-Pos Message Handler]: Received the finished go home pose message")

    def finished_set_stiffness_message_handler(self, message):
        self.whether_finished_set_stiffness = True
        # print("[Agent Set-Stiffness Message Handler]: Received the finished set stiffness message")

    def finished_set_stiffness_whole_body_message_handler(self, message):
        self.whether_finished_set_stiffness_whole_body = True

    def robot_data_message_handler(self, message):
        # print("[Robot Data Handler]: Received robot data is: {}".format(message['data']))
        # print(type(message['data']))
        joint_values = [float(i) for i in message['data'].split()]
        self.latest_robot_data = np.array(joint_values)
        self.whether_received_robot_data = True
        # print("[Agent Robot-data Message Handler]: Received robot data")

    def robot_data_whole_body_message_handler(self, message):
        joint_values = [float(i) for i in message['data'].split()]
        self.latest_robot_data_whole_body = np.array(joint_values)
        self.whether_received_robot_data_whole_body = True

    def finished_take_action_message_handler(self, message):
        self.whether_finished_take_action = True

    def finished_set_led_message_handler(self, message):
        self.whether_finished_set_led = True

    def received_hand_pose_message_handler(self, message):
        hand_pose = [float(i) for i in message['data'].split()]
        self.latest_hand_pose = np.array(hand_pose)
        self.whether_received_hand_pose = True







