import numpy as np

from numpy.linalg import det

def generate_curve_fitting_polynomial(degree, x, y, debug=False):
        assert type(x) is np.ndarray and type(y) is np.ndarray, "Both x and y should be numpy arrays"
        assert x.shape == y.shape, "Shape mismatch of arrays"
                
        pow_sum = [sum([x_n**i for x_n in x]) for i in range(0, 2*degree + 1)]
                        
        M = np.matrix([[pow_sum[i + j] for i in range(0, degree + 1)] for j in range(0, degree + 1)])
                                
        det_M = det(M)
                                        
        coeffs = np.empty((degree + 1))
        b = np.array([sum([(x_n**i)*y_n for (x_n,y_n) in zip(x, y)]) for i in range(0, degree + 1)])
                                                        
        # Calculating the coefficients according to Cramer's rule
        for i in range(0, degree + 1):
            M_i = np.copy(M)
            M_i[:,i] = b
            coeffs[i] = det(M_i) / det_M
                                                                                                    
        return coeffs

def eval_polynomial(x, coeffs):
        assert type(coeffs) is np.ndarray, "Type of coeffs must be an ndarray"
        x_pows = np.array([x**i for i in range(coeffs.shape[0])])
        return np.dot(x_pows, coeffs)
