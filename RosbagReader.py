import rosbag

class Reader:

    @staticmethod
    def readBag(filename, topic_list):
        bag = rosbag.Bag(filename)
        all_topics = list()
        for topic in bag.read_messages(topics=topic_list):
            all_topics.append(topic)
        bag.close()
        return all_topics