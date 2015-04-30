import rosbag
import re
import ast

class Reader:


    @staticmethod
    def readBag(bagname, topic):
        bag = rosbag.Bag(bagname)
        rostopic = "/uBot/joint_" + topic
        trajectories, times =[], []
        for top, msg, t in bag.read_messages(topics=[rostopic]):
            message = str(msg)
            ###Parse timestamps#####
            timestamp = int(re.search("nsecs: .*", message).group(0).split(" ")[1])
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