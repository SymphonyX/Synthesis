import numpy as np 
import matplotlib.pyplot as plt
import scipy.interpolate

def quadraticFunc(x):
    out = list()
    for d in x:
        out.append((-d**2))
    return  out

def generate_example(time=1.0):
    train = open("train.txt", "r")
    lines = train.readlines()
    del lines[0]
    trajectory = list()
    for line in lines:
        line = line.split("\n")[0]
        line = line.split(",")
        trajectory.append( ( float(line[0]), float(line[1]) ) )

    trajectory = sorted(trajectory, key=lambda x: x[0])
    demonstration = [pos[1] for pos in trajectory]

    #Other test
    demonstration = quadraticFunc(np.linspace(-4,4.0, num=100))

    velocities = np.zeros( (len(demonstration), 1) )
    accelerations = np.zeros( (len(demonstration), 1) )

    times = np.linspace(0, time, num=len(demonstration))


    for i in range(1,len(demonstration)):
        dx = demonstration[i] - demonstration[i-1]
        dt = times[i] - times[i-1]

        velocities[i] = dx / (dt / time) - velocities[i-1]
        accelerations[i] = (velocities[i] - velocities[i-1]) / dt

    velocities[0] = velocities[1] - (velocities[2] - velocities[1])
    accelerations[0] = accelerations[1] - (accelerations[2] - accelerations[1])

    return demonstration, velocities, accelerations, times


class DMP:

    def __init__(self, bfs, start, goal, K, D, dt=0.01, tau=1.0, h=0.01):
        self.bfs = bfs
        self.start = start
        self.goal = goal
        self.K = K
        self.D = D

        self.dt = dt
        self.alphax = 1.0 #For canonical system
        self.timesteps =  int(tau / dt)


        first = np.exp(-self.alphax*tau)
        last = 1.05 - first
        des_c = np.linspace(first,last,self.bfs)

        self.c = np.ones(len(des_c))
        for n in range(len(des_c)):
        # x = exp(-c), solving for c
            self.c[n] = -np.log(des_c[n])
        self.h = np.ones(self.bfs) * self.bfs**1.5 / self.c


        self._reset()

    def _reset(self):
        self.pos = self.start
        self.vel = 0
        self.acc = 0
        self.x = 1.0


    def rollout(self, tau=1.0):

        self._reset()
        pos_track = np.zeros( (self.timesteps, 1) )
        vel_track = np.zeros( (self.timesteps, 1) )
        acc_track = np.zeros( (self.timesteps, 1) )

        for t in range(self.timesteps):
            pos, vel, acc = self.step(tau=tau)

            pos_track[t] = pos
            vel_track[t] = vel
            acc_track[t] = acc

        return pos_track, vel_track, acc_track


    def _step_canonical(self, tau=1.0, error_coupling=1.0):
        self.x += (-self.alphax * self.x * error_coupling) * tau * self.dt
        return self.x

    def step(self, tau=1.0, coupling=1.0):
        x = self._step_canonical(tau, coupling)
        gaussians = self._gaussians(x)

        f = self._dim_term(x) * np.transpose(gaussians).dot(self.weights) / np.sum(gaussians)

        self.acc = (self.K * (self.D * (self.goal - self.pos) - self.vel/tau) + f) * tau**2
        self.vel += self.acc * self.dt * coupling
        self.pos += self.vel * self.dt * coupling

        return self.pos, self.vel, self.acc


    def _offset_goal(self):
        if (self.start == self.goal):
            self.goal += 1e-4

    def _gaussians(self, x):
        if isinstance(x, np.ndarray):
            x = x[:,None]
        return np.exp( -self.h * (x - self.c)**2)

    def _dim_term(self, x):
        return x * (self.goal - self.start)

    def _gen_weights(self, f_target, tau=1.0, coupling=1.0):
        xtrack = np.zeros(self.timesteps)
        self.x = 1.0
        for t in range(self.timesteps):
            xtrack[t] = self.x
            self._step_canonical(tau, coupling)

        gaussians = self._gaussians(xtrack)
        self.weights = np.zeros((self.bfs, 1))

        #spatial scaling term
        k = (self.goal - self.start)
        for i in range(self.bfs):
            numer = np.sum(xtrack * gaussians[:,i] * f_target[:,0])
            denom = np.sum(xtrack**2 * gaussians[:,i])
            self.weights[i] = numer / (k * denom)

    def follow_demonstration(self, demonstration, tau=1.0, coupling=1.0):
        self.start = demonstration[0]
        self.goal = demonstration[-1]

        self._offset_goal()
        path = np.zeros( (self.timesteps, 1) )
        x = np.linspace(0, tau, len(demonstration))
        path_gen = scipy.interpolate.interp1d(x, demonstration)
        for t in range(self.timesteps):
            path[t] = path_gen(t * self.dt)

        y_des = path
        ydot_des = np.diff(y_des, axis=0) / self.dt
        ydot_des = np.vstack( (np.zeros((1, 1)), ydot_des) )

        yddot_des = np.diff(ydot_des, axis=0) / self.dt
        yddot_des = np.vstack( (np.zeros((1, 1)), yddot_des) )

        f_target = yddot_des - self.K * (self.D * (self.goal - y_des) - ydot_des)

        self._gen_weights(f_target, tau, coupling)
        self._reset()




if __name__ == "__main__":


    K = 25.0
    D = K / 4
    dt = 0.01
    time = 1.0
    h = 0.1

    demonstration, velocities, accelerations, times = generate_example(time)

    path1 = np.sin(np.arange(0,1,.01)*5)
    path_times = np.linspace(0, 1, len(path1))

    start = path1[0]
    goal = path1[-1]
    bfs = 30


    tau = 1.0
    dmp = DMP(bfs, start, goal, K, D, dt, tau, h)
    dmp.follow_demonstration(path1, tau)

    dmp.start += 1
    dmp.goal += 1
    y_track, ydot_track, yddot_track = dmp.rollout()

    sample_times = np.linspace(0, time, time / dt)

    plt.plot(path_times, path1)
    plt.plot(sample_times, y_track)
    plt.ylim(-2, 2)

    plt.show()
