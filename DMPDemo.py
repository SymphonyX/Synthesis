from dmp import DMP
import numpy as np
import matplotlib.pyplot as plt
from RosbagReader import Reader

def quadraticFunc(x):
    out = list()
    for d in x:
        out.append((-d**2))
    return  out

def generate_example(time=1.0):
    train = open("trajectory2.txt", "r")
    lines = train.readlines()
    del lines[0]
    trajectory = list()
    for line in lines:
        line = line.split("\n")[0]
        line = line.split(",")
        trajectory.append( float(line[0]) )

    demonstration = [pos for pos in trajectory]

    #Other test
    #demonstration = quadraticFunc(np.linspace(-4,4.0, num=100))
    #demonstration = np.sin(np.arange(0,time,.01)*5)

    return diff_demonstration(demonstration, time)

    
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


if __name__ == '__main__':
    K = 200.0
    D = K / 4
    basis = 10

    ubot_joint_traj, times = Reader.jointPositions("idontknow.bag")
    ubot_joint_vel, times = Reader.jointVelocities("idontknow.bag")
    ubot_joint_torques, times = Reader.jointTorques("idontknow.bag")

    t_demonstration = 1.0
    #demonstration, velocities, accelerations, times = generate_example(t_demonstration)
    demonstration, velocities, accelerations, times = diff_demonstration(demonstration, t_demonstration)


    dmp = DMP(basis, K, D, demonstration[0], demonstration[-1])
    dmp.learn_dmp(times, demonstration, velocities, accelerations)

    tau = 1.0
    x, xdot, xddot, t = dmp.run_dmp(tau, 0.01, demonstration[0]+5, demonstration[-1])

    plt.plot(t, x, c="b")
    plt.plot(times, demonstration, c="r")


    plt.show()
