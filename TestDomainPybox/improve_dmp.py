import sys
import cPickle
sys.path.insert(0, "../")
from dmp import DMP

import numpy as np
import matplotlib.pyplot as plt

tau = 2.0
basis = 5
dmp_dt = 0.002


def diff_demonstration(demonstration, time):
    velocities = np.zeros( (len(demonstration), 1) )
    accelerations = np.zeros( (len(demonstration), 1) )

    times = np.linspace(0, time, num=len(demonstration))


    for i in range(1,len(demonstration)):
        dx = demonstration[i] - demonstration[i-1]
        dt = times[i] - times[i-1]

        velocities[i] = dx / dt
        accelerations[i] = (velocities[i] - velocities[i-1]) / dt

    velocities[0] = velocities[1] - (velocities[2] - velocities[1])
    accelerations[0] = accelerations[1] - (accelerations[2] - accelerations[1])

    return demonstration, velocities, accelerations, times

param_file = open(sys.argv[1], "r")
parameters = cPickle.load(param_file)

K = 50.0
D = 10.0

dmp1 = DMP(basis, K, D, 0.0, parameters[0])
dmp2 = DMP(basis, K, D, 0.0, parameters[1])
dmp3 = DMP(basis, K, D, 0.0, parameters[2])

count = 3
for i in range(basis):
    dmp1.weights[i] = parameters[count]
    dmp2.weights[i] = parameters[count+basis]
    dmp3.weights[i] = parameters[count+(2*basis)]
    count += 1

last_step = 250
xpos, xdot, xddot, times = dmp1.run_dmp(tau, dmp_dt, dmp1.start, dmp1.goal)
x1, x1dot, x1ddot, t1 = diff_demonstration(xpos[:last_step], tau)
dmp1new = DMP(basis, K, D, 0.0, xpos[last_step])
dmp1new.learn_dmp(t1, x1, x1dot, x1ddot)

xpos1, xdot1, xddot1, times1 = dmp1new.run_dmp(tau, dmp_dt, dmp1new.start, dmp1new.goal)
plt.plot(t1, x1, times1, xpos1, times, xpos)
plt.show()

xpos, xdot, xddot, times = dmp2.run_dmp(tau, dmp_dt, dmp2.start, dmp2.goal)
x2, x2dot, x2ddot, t2 = diff_demonstration(xpos[:last_step], tau)
dmp2new = DMP(basis, K, D, 0.0, xpos[last_step])
dmp2new.learn_dmp(t2, x2, x2dot, x2ddot)

xpos2, xdot2, xddot2, times2 = dmp2new.run_dmp(tau, dmp_dt, dmp2new.start, dmp2new.goal)
plt.plot(t2, x2, times2, xpos2, times, xpos)
plt.show()


xpos, xdot, xddot, times = dmp3.run_dmp(tau, dmp_dt, dmp3.start, dmp3.goal)
x3, x3dot, x3ddot, t3 = diff_demonstration(xpos[:last_step], tau)
dmp3new = DMP(basis, K, D, 0.0, xpos[last_step])
dmp3new.learn_dmp(t3, x3, x3dot, x3ddot)

xpos3, xdot3, xddot3, times3 = dmp3new.run_dmp(tau, dmp_dt, dmp3new.start, dmp3new.goal)
plt.plot(t3, x3, times3, xpos3, times, xpos)
plt.show()


new_parameters = np.zeros( (basis*3+3) )

new_parameters[0] = dmp1new.goal
new_parameters[1] = dmp2new.goal
new_parameters[2] = dmp3new.goal

count = 3
for i in range(basis):
    new_parameters[count] = dmp1new.weights[i]
    new_parameters[count+basis] = dmp2new.weights[i]
    new_parameters[count+(2*basis)] = dmp3new.weights[i]
    count += 1

filename = sys.argv[1].split("/")[1]
file_params = open(filename, "w")
cPickle.dump(new_parameters, file_params)


