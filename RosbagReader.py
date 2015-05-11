import rosbag
import re
import ast

class Reader:
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
    def readBag(bagname, topic):
        bag = rosbag.Bag(bagname)
        rostopic = "/uBot/joint_" + topic
        trajectories, times =[], []
        for top, msg, t in bag.read_messages(topics=[rostopic]):
            message = str(msg)
            ###Parse timestamps#####
            timestamp = int(re.search("secs: .*", message).group(0).split(" ")[1])
            times.append(timestamp)
            ###Parse positions###
            pattern = topic+": .*"
            parsedPositions = re.search(pattern, message).group(0).split(":")[1]
            listEval = ast.literal_eval(parsedPositions[1:])
            trajectories.append(listEval)
        
        bag.close()
        return trajectories, times

    @staticmethod
    def jointPositions(bagname):
        return Reader.readBag(bagname, topic="positions")

    @staticmethod
    def jointTorques(bagname):
        return Reader.readBag(bagname, topic="torques")

    @staticmethod
    def jointVelocities(bagname):
        return Reader.readBag(bagname, topic="velocities")