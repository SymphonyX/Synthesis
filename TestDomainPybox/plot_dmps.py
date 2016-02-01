
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np



if __name__ == '__main__':
	if len(sys.argv) < 2:
		print "Not enough arguments"
	else:
		param_file = open(sys.argv[1], "r")
		all_pos = pickle.load(param_file)

		print all_pos
		xs = np.linspace(0,len(all_pos[0])-1, num=len(all_pos[0]))

		plt.plot(xs, np.asarray(all_pos[0]), xs, np.asarray(all_pos[1]))
		plt.show()




