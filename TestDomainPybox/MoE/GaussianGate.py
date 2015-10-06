from Gate import *
import numpy as np
import math
from scipy import optimize


class GaussianGate(Gate):
    def __init__(self, num_experts, dimensions_in, dimensions_out, mode, training_x):
        self.alphas = np.random.random( (num_experts, 1) )
        self.sigma = np.random.random( (num_experts, dimensions_in, dimensions_in) )
        self.mode = mode

        #For visualization only
        self.tracking_alphas = list()
        self.tracking_sigmas = list()
        self.tracking_ms = list()
        self.experts_weights = list()

        for i in range(num_experts):
            self.sigma[i] = np.identity(dimensions_in) / 1000.0

        self.saveBestParams()

    def saveBestParams(self):
        self.best_alphas = self.alphas.copy()
        self.best_sigma = self.sigma.copy()

    def setToBestParams(self):
        self.alphas = self.best_alphas.copy()
        self.sigma = self.best_sigma.copy()

    def outputs(self, experts, x):
        n = x.shape[0]
        px_times_alpha = list()
        for expert in experts:
            sigma_i = self.sigma[expert.index]
            m_i = expert.mean()
            if math.isnan(m_i[0,0]):
                print "a"
            xm_diff = x - m_i
            mult = -0.5 * (xm_diff.dot(np.linalg.pinv(sigma_i)).dot(np.transpose(xm_diff)))
            prod = math.pow((2*math.pi), -n/2.0) * np.power(np.linalg.det(sigma_i), -1.0/2.0)
            if mult < -200.0:
                mult = -200.0
            px_given_vj = prod * np.exp(mult)
            px_times_alpha.append( (px_given_vj * self.alphas[expert.index]) )


        params = np.random.random( (len(experts), 1) )
        sum_px = sum(px_times_alpha)
        for i in range(params.shape[0]):
            params[i] = px_times_alpha[i] / sum_px
            if math.isnan(params[i]):
                print "A"

        return np.transpose(params)[0]

    #E-step
    def _compute_hs(self, experts, x, y):
        temp_hs = np.zeros( (len(experts), 1) )
        hs = np.zeros( (len(experts), 1) )

        y = np.transpose(y)
        for i, expert in enumerate(experts):
            expertOutput = expert.computeExpertyhat(x)
            if expertOutput.shape != y.shape:
                raise Exception("Output shape does not align. Output shape: ", expertOutput.shape, " Y: ", y.shape)

            exponent = -0.5 * np.transpose((y - expertOutput)).dot((y - expertOutput))
            if exponent < -200.0:
                exponent = -200.0
            g_xv = self.outputs(experts, x)

            temp_hs[i] = g_xv[i] * math.exp( exponent )

        for i in range(temp_hs.shape[0]):
            sum_temp_hs = np.sum(temp_hs, axis=0)
            hs[i] = temp_hs[i] / sum_temp_hs
            if sum_temp_hs == 0:
                print "b"

        return hs

    #M-step
    def _update_alphas(self, experts, xs, ys):
        n = xs.shape[0] 
        sum_hs = np.zeros( (len(experts), 1) )
        for i in xrange(ys.shape[0]):
            sum_hs += self._compute_hs(experts, xs[i], ys[i])

        return sum_hs / n

    def _update_ms(self, experts, xs, ys):
        sum_hs = np.zeros( (len(experts), 1) )
        sum_hs_x = np.zeros( (len(experts), xs[0].shape[0]) )
        for i in xrange(ys.shape[0]):
            hs = self._compute_hs(experts, xs[i], ys[i])
            sum_hs += hs
            sum_hs_x += hs * xs[i]

        ms = list()
        for i in xrange(len(experts)):
            ms.append( sum_hs_x[i] / sum_hs[i] )

        return ms


    def _update_sigma(self, experts, xs, ys):
        xshape = (1, xs[0].shape[0])
        yshape = (1, ys[0].shape[0])
        sum_hs = np.zeros( (len(experts), 1) )
        sum_hs_xm = np.zeros( self.sigma.shape )
        for i in range(ys.shape[0]):
            x, y = xs[i], ys[i]
            x.shape = xshape
            y.shape = yshape
            hs = self._compute_hs(experts, x, y)
            sum_hs += hs
            for j in range(hs.shape[0]):
                m = experts[j].mean()
                hj = hs[j]
                xm_xmt = np.transpose((x - m)).dot((x - m))
                sum_hs_xm[j] += hj * xm_xmt

        sigmas = list()
        for i in range(self.sigma.shape[0]):
            sigmas.append( sum_hs_xm[i] / sum_hs[i]  )

        return sigmas


    def find_best_means(self, expert, training_x, training_y):
        vals = list()
        for i in range(training_x.shape[0]):
            x = training_x[i]
            y = training_y[i]
            hs = self._compute_hs([ expert ], x, y)
            vals.append( (hs[0], x) )

        vals = sorted(vals, key=lambda x:x[0][0], reverse=True)
        return vals[0][1], vals[1][1]


    def weighted_2_means(self, training_x, training_y, newExpert, oldExpert):
        htsum = np.zeros( (2, 1) )
        htxsum = np.zeros( (2, training_x.shape[1]) )

        for i in range(training_x.shape[0]):
            x = training_x[i]
            y = training_y[i]

            hs = self._compute_hs( [ newExpert, oldExpert ], x, y)

            if hs[0] > hs[1]:
                htsum[0] += hs[0]
                htxsum[0] += hs[0] * x
            else:
                htsum[1] += hs[1]
                htxsum[1] += hs[1] * x

        return htxsum[0] / htsum[0], htxsum[1] / htsum[1]


    def update_weights_wls(self, experts, training_x, training_y):
        Cmat = np.zeros( (len(experts), training_x.shape[0], training_x.shape[0]) )
        Amat = np.zeros( (training_x.shape[0], training_x.shape[1]) )
        yvec = np.zeros( (training_y.shape[0], 1) )
        all_weights = []

        for k in range(training_y.shape[1]):
            for i in range(training_x.shape[0]):
                x = training_x[i]
                y = training_y[i]
                hs = self._compute_hs(experts, x, y)

                Amat[i] = x
                yvec[i] = y[k]
                for j in range(len(experts)):
                    Cmat[j][i][i] = hs[j]
                    # if hs[j] == 0:
                    #     print "Weights cannot be 0!!!!"

            return_weights = []
            for i, expert in enumerate(experts):
                try:
                    weights = np.linalg.pinv( np.transpose(Amat).dot(Cmat[i]).dot(Amat) ).dot(np.transpose(Amat)).dot(Cmat[i]).dot(yvec)
                    return_weights.append( np.transpose(weights) )
                except np.linalg.LinAlgError:
                    print "Linalg Error"

            all_weights.append(return_weights)

        return all_weights


    def train(self, training_x, training_y, experts, learningRate):

        new_alphas = self._update_alphas(experts, training_x, training_y)
        new_ms = self._update_ms(experts, training_x, training_y)
        new_sigmas = self._update_sigma(experts, training_x, training_y)

        new_weights = self.update_weights_wls(experts, training_x, training_y)

        for j in range(training_y.shape[1]):
            for i, expert in enumerate(experts):
                expert.weights[j] = new_weights[j][i]

        self.alphas = new_alphas

        new_sigmas = np.array(new_sigmas) * np.eye( new_sigmas[0].shape[0] )
        new_sigmas[ new_sigmas < 0.00001 ] = 0.00001

        for i, m in enumerate(new_ms):
            experts[i].setMean(m)

        thetas = []
        for i, expert in enumerate(experts):
            thetas.append( expert.weights )


        #for visualization only
        self.tracking_alphas.append( self.alphas.copy() )
        self.tracking_sigmas.append( self.sigma.copy() )
        self.tracking_ms.append( [e.mean().copy() for e in experts] )
        self.experts_weights.append( thetas )


    def setIterationValues(self, iteration, experts):
        self.alphas = self.tracking_alphas[iteration]
        self.sigma = self.tracking_sigmas[iteration]

        for i, expert in enumerate(experts):
            expert.weights = self.experts_weights[iteration][i]
            expert.setMean( self.tracking_ms[iteration][i] )
