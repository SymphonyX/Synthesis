import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

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


class DMP:

    def __init__(self, num_basis, K, D, start, goal):
        self.weights = np.random.random( (num_basis, 1) )
        self.mus = np.zeros( (num_basis, 1) )
        self.sigmas = np.zeros( (num_basis, 1) )

        for i in range(self.mus.shape[0]):
            step = (1.0 / num_basis)
            self.mus[i] =  (step / 2) + i * step
            self.sigmas[i] = 1.0 / num_basis

        self.K = K
        self.D = D
        self.start = start
        self.goal = goal
        self.vel = 0
        self.acc = 0
        self.pos = start

    def _gaussians(self, s):
        return np.exp( np.multiply(-1 / (2*(self.sigmas)**2), np.square(s - self.mus) ) )

    def _acceleration(self, s):
        #Original
        #return (self.K * (self.goal - self.pos)) - (self.D * self.vel) + ((self.goal - self.start) * self._forcing_term(s))
        #Newer
        return (self.K * (self.goal - self.pos)) - (self.D * self.vel) - (self.K * (self.goal - self.start) * s) + (self.K * self._forcing_term(s))

    def _forcing_term(self, s):
        gaussian = self._gaussians(s)
        s_gaussians = gaussian * s
        sum_basis_weight = np.transpose(self.weights).dot(s_gaussians)
        sum_basis = np.sum(gaussian)

        return (sum_basis_weight[0][0] / sum_basis)

    def _f_target(self, s, tau=1.0):
        #Original
        #return ( (-self.K * (self.goal - self.pos)) + (self.D * self.vel) + (tau * self.acc) ) / (self.goal - self.start)
        #Newer
        return (((tau * self.acc) + (self.D * self.vel)) / self.K) - (self.goal - self.pos) + ((self.goal - self.start) * s)


    def solve_canonical_system(self, t, tau=1.0, alpha = -math.log(0.05)):
        # tau * s' = -alpha * s: s depends on time, so this diffeq is actually a function of time.
        # the solution has the form given below: s(t) = c * e^(-(alpha*t)/tau)
        return math.exp(- (alpha * t) / tau )

    def find_weights(self, Amat, fvec):

        At = np.transpose(Amat)
        At_dotA = At.dot(Amat)
        weights = np.linalg.inv(At_dotA).dot(At).dot(fvec)
        return weights

    def _offset_goal(self):
        if (self.start == self.goal):
            self.goal += 1e-4


    def learn_dmp(self, times, positions, velocities, accelerations):

        self._offset_goal()

        subdivide = 10

        Amat = np.zeros( (len(times)*subdivide, self.weights.shape[0] ) )
        fvec = np.zeros( (len(times)*subdivide, 1) )

        for i in range(len(times)-1):
            time = times[i]
            dt = (times[i+1] - times[i]) / subdivide
            pos = positions[i]


            for j in range(subdivide):
                t = time + (dt * j) 
                s = self.solve_canonical_system(t, tau=times[-1])

                gauss = self._gaussians(s)
                gauss = (gauss * s) / np.sum(gauss)

                Amat[(i * subdivide) + j] = np.transpose(gauss)
                fvec[(i * subdivide) + j] = self._f_target(s, tau=times[-1])

                delta_pos = ((positions[i+1] - pos) / subdivide)
                new_vel = delta_pos / dt
                new_acc = (new_vel - self.vel) / dt

                self.pos += delta_pos
                self.vel = new_vel
                self.acc = new_acc

        self.weights = self.find_weights(Amat, fvec)

    def _reset(self, start, goal):
        self.start = start
        self.goal = goal
        self.pos = start
        self.vel = 0
        self.acc = 0

    def run_dmp(self, tau, dt, start, goal):
        self._reset(start, goal)
        timesteps = int(tau / dt)

        x = np.zeros( (timesteps, 1) )
        xdot = np.zeros( (timesteps, 1) )
        xddot = np.zeros( (timesteps, 1) )
        times = np.zeros( (timesteps, 1) )

        t = 0
        for i in range(timesteps):
            x[i] = self.pos
            xdot[i] = self.vel
            xddot[i] = self.acc
            times[i] = t

            s = self.solve_canonical_system(t, tau)

            self.acc = self._acceleration(s)
            self.vel += (self.acc * dt) / tau
            self.pos += (self.vel * dt) / tau


            t += dt

        return x, xdot, xddot, times


if __name__ == '__main__':
    K = 200.0
    D = K / 4
    basis = 30

    t_demonstration = 1.0
    demonstration, velocities, accelerations, times = generate_example(t_demonstration)

    dmp = DMP(basis, K, D, demonstration[0], demonstration[-1])
    dmp.learn_dmp(times, demonstration, velocities, accelerations)

    tau = 1.0
    x, xdot, xddot, t = dmp.run_dmp(tau, 0.01, demonstration[0], demonstration[-1]-5)

    plt.plot(t, x, c="b")
    plt.plot(times, demonstration, c="r")


    plt.show()
