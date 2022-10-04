import numpy as np
import pybullet as p

class Fsn_policy:
    def __init__(self, bodyUniqueId, endEffectorLinkIndex):
        # load robot model
        self.bodyUniqueId = bodyUniqueId
        self.endEffectorLinkIndex = endEffectorLinkIndex

        self.P_ori_hip2robot = np.array([0.75, 0.0, 0.9])
        self.last_human_pos = None # human right wrist position in human hip frame
        self.current_human_pos = None # human right wrist position in human hip frame

        # state transition flags and thresholds
        self.whether_finished_still = False
        self.human_moving_backward = False
        self.backward_steps = 0
        self.whether_during_initiating = False
        self.whether_finished_reach = False
        self.trigger_thres_dist = -0.2 # distance threshold for right wrist-z position beyond which robot will start reaching human hands
        self.back_steps_thres = 10
        self.stop_tres_dist = 0.001 # distance threshold for robot to

    # return joint angle command for the input state
    def act(self, human_pos):
        # right wrist position in hip frame
        wrist_x_h = human_pos[0]
        wrist_y_h = human_pos[1]
        wrist_z_h = human_pos[2]

        if self.last_human_pos is None:
            self.last_human_pos = np.array([human_pos[0], human_pos[1], human_pos[2]])
            self.whether_during_initiating = True
        else:
            self.last_human_pos = self.current_human_pos
            self.whether_during_initiating = False

        self.current_human_pos = np.array([human_pos[0], human_pos[1], human_pos[2]])

        if not self.whether_finished_reach:
            # when the human hand is not reaching
            if (not self.whether_finished_still) and wrist_z_h >= self.trigger_thres_dist or self.whether_during_initiating:
                print("[Finite State Machine]: Stay still")
                # print("whether_finished_reach: {}, wrist_z_h >= trigger_thres: {}, whether_during_initiating: {}".format(self.whether_finished_reach, wrist_z_h >= self.trigger_thres_dist, self.whether_during_initiating))
                action = 'stay_still'
                return None, action
            else:
                self.whether_finished_still = True

                if self.current_human_pos[2] - self.last_human_pos[2] > 0.0 and wrist_z_h >= -0.4:
                    self.backward_steps += 1
                if self.backward_steps >= self.back_steps_thres:
                    self.human_moving_backward = True

                # when the human hand is moving forward
                if self.current_human_pos[2] - self.last_human_pos[2] <= 0.0 or (not self.human_moving_backward):
                    wrist_x_r = wrist_z_h + self.P_ori_hip2robot[0]
                    wrist_y_r = - wrist_x_h
                    wrist_z_r = - wrist_y_h + self.P_ori_hip2robot[2]
                    wrist_pos_r = np.array([wrist_x_r, wrist_y_r, wrist_z_r])

                    target_joint_angles = self.solve_ik(wrist_pos_r)
                    print("[Finite State Machine]: Reach for hands")
                    action = 'reach'
                    return target_joint_angles, action
                # when the human hand is moving backward
                else:
                    print('[Finite State Machine]: Go home pose (not triggered)')
                    self.whether_finished_reach = True
                    action = 'go_home_untriggered'
                    return np.array([np.inf, np.inf, np.inf, np.inf]), action
        else:
            # when the human hand is moving backward
            print('[Finite State Machine]: Go home pose (triggered)')
            action = 'go_home_triggered'
            return None, action
            # return np.array([np.inf, np.inf, np.inf, np.inf])

    def reset(self):
        self.last_human_pos = None
        self.current_human_pos = None

        self.whether_finished_still = False
        self.human_moving_backward = False
        self.backward_steps = 0
        self.whether_during_initiating = False
        self.whether_finished_reach = False

    def calculate_distance(self, pos_1, pos_2):
        return np.linalg.norm(pos_1 - pos_2)

    def solve_ik(self, target_pos):
        ik = p.calculateInverseKinematics(bodyIndex=self.bodyUniqueId,
                                          endEffectorLinkIndex=self.endEffectorLinkIndex,
                                          targetPosition=target_pos)

        knee_pitch = ik[0]
        hip_pitch = ik[1]
        hip_roll = ik[2]
        rshoulder_pitch = ik[25]
        rshoulder_roll = ik[26]
        relbow_yaw = ik[27]
        relbow_roll = ik[28]

        # return np.array([rshoulder_pitch, rshoulder_roll, relbow_yaw, relbow_roll])
        return np.array([rshoulder_pitch, rshoulder_roll, relbow_yaw, relbow_roll,
                         knee_pitch, hip_pitch, hip_roll])








