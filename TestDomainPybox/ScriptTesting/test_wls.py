import numpy as np
import matplotlib.pyplot as plt


def update_weights_wls(Cmat, Amat, yvec):
    print "C: ", Cmat.shape, " : ", Amat.shape, " : ", yvec.shape
    
    weights = np.linalg.inv( np.transpose(Amat).dot(Cmat).dot(Amat) ).dot(np.transpose(Amat)).dot(Cmat).dot(yvec)
    
    return weights



if __name__ == '__main__':

    data = np.linspace(0, 10, num=50)
    data2 = np.linspace(10, 0, num=50)

    data = np.concatenate( (data, data2) )

    yvec = np.zeros( (data.shape[0], 1) )
    Cmat = np.zeros( (2, 100, 100) )

    x1 = np.linspace(0, 10, num=100)
    x2 = np.linspace(1, 1, num=100)
    Amat = np.zeros( (100, 2) )

    for i in range(data.shape[0]):
        yvec[i][0] = data[i] + np.random.normal(0, 1)

        Cmat[0][i][i] = 0.99 if i < 50 else 0.01
        Cmat[1][i][i] = 0.01 if i < 50 else 0.99
        Amat[i][0] = x1[i]
        Amat[i][1] = x2[i]


    for i in range(yvec.shape[1]):
        weights = update_weights_wls(Cmat[0], Amat, yvec )
        weights1 = update_weights_wls(Cmat[1], Amat, yvec )
        
        est = Amat.dot(weights)
        est1 = Amat.dot(weights1)

        plt.plot(x1, est, x1, est1, x1, yvec[:,i])

        plt.show()


    
