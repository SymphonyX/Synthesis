import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    all_params = [ ]	
    for param in sys.argv:
    	if param is not sys.argv[0]:
    		param_file = open(param, "r")
    		result = pickle.load(param_file)
    		all_params.append( result )
    
    axis = np.linspace(0, len(all_params[0]), num=len(all_params[0]))

    for param_list in all_params:
    	plt.plot(axis, param_list)

    plt.show()


