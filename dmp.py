import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

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

    def acceleration(self, s):
        #Original
        #return (self.K * (self.goal - self.pos)) - (self.D * self.vel) + ((self.goal - self.start) * self._forcing_term(s))
        #Newer
        return (self.K * (self.goal - self.pos)) - (self.D * self.vel) - (self.K * (self.goal - self.start) * s) + (self.K * self._forcing_term(s))

    def _forcing_term(self, s):
        sum_basis = 0
        sum_basis_weight = 0
        gaussian = self._gaussians(s)
        for i in range(self.weights.shape[0]):
            sum_basis_weight += self.weights[i] * gaussian[i] * s
            sum_basis += gaussian[i]

        return (sum_basis_weight[0] / sum_basis[0]) * s

    def _f_target(self, s, tau=1.0):
        #Original
        #return ( (-self.K * (self.goal - self.pos)) + (self.D * self.vel) + (tau * self.acc) ) / (self.goal - self.start)
        #Newer
        return (((tau * self.acc) + (self.D * self.vel)) / self.K) - (self.goal - self.pos) + ((self.goal - self.start) * s)


    def solve_canonical_system(self, t, tau=1.0, alpha = -math.log(0.01)):
        # tau * s' = -alpha * s: s depends on time, so this diffeq is actually a function of time.
        # the solution has the form given below: s(t) = c * e^(-(alpha*t)/tau)
        return math.exp(- (alpha * t) / tau )

    def compute_error(self, positions, velocities, accelerations, times):
        error = 0
        for i in range(len(velocities)):
            self.vel = velocities[i][0]
            self.acc = accelerations[i][0]
            self.pos = positions[i]
            t = times[i]
            s = self.solve_canonical_system(t)
            error += ( self._f_target(s) - self._forcing_term(s) )**2
        return error


    def find_best_alpha(self, alpha_decay, sum_grad, weights, positions, velocities, accelerations, times):
        error = float("inf")
        alpha_error = float("inf")
        prev_alpha_error = 0
        alpha_iter = 0
        while prev_alpha_error - alpha_error > 0.01 or alpha_error == float("inf"):
            alpha = alpha_decay**alpha_iter
            self.weights = weights - (alpha * sum_grad)
            prev_alpha_error = alpha_error

            alpha_error = self.compute_error(positions, velocities, accelerations, times)

            if alpha_error < error: error = alpha_error

            alpha_iter += 1

        return alpha_decay**(alpha_iter-1), error

    def find_weights(self, Amat, fvec):

        At = np.transpose(Amat)
        At_dotA = At.dot(Amat)
        weights = np.linalg.inv(At_dotA).dot(At).dot(fvec)
        return weights




    def train_dmp(self, times, positions, velocities, accelerations, learning_rate, decay):

        error = float("inf")
        previous_error = 0
        iteration = 0
        while  iteration < 1000 or (previous_error - error) >  0.000001:

            sum_grad = 0.0
            sum_new_weights = 0.0
            fvec = np.zeros( (len(velocities), 1) )
            Amat = np.zeros( (len(velocities), self.weights.shape[0]) )

            for i in range(len(velocities)):
                self.vel = velocities[i][0]
                self.acc = accelerations[i][0]
                self.pos = positions[i]
                t = times[i]

                s = self.solve_canonical_system(t)
                gaussians = self._gaussians(s)

                Am