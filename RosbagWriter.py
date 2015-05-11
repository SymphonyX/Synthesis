import rosbag
from ubot_msgs.msg import JointPositions 

class Writer:
    JH_WHEEL1     = 0     #hinge joint for wheel 1
    JH_WHEEL2     = 1     #hinge joint for wheel 2
    JH_TORSO      = 2     #hinge joint for torso
    JH_SHOULDER1  = 3     #hinge joint for shoulder 1
    JH_SHOULDER2  = 4     #hinge joint for shoulder 2
    JH_SPACER1    = 5
    JH_SPACER2    = 6
    JH_ARM1       = 7     #hinge joint for arm 1
    JH_ARM2       = 8     #hinge joint for arm 2
    JH_FARM1      = 9     #hinge joint for forearm 1
    JH_FARM2      = 10    #hinge joint for forearm 2

    @staticmethod
    def writePositions(bagname, values, times):
        bag = rosbag.Bag(bagname, "w")
        rostopic = "/uBot/joint_positions"

        for i in range(values.shape[0]):
            joint_pos = JointPositions()

            joint_pos.numJoints = len(values[i])
            joint_pos.JointPositions = values[i]
            joint_pos.stamp = times[i]

            bag.write(rostopic, joint_pos)




