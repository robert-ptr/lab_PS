import numpy as np

def get_polynomial_roots(coeffs): # doesn't include the dominant coefficient(that is 1)
    coeffs=  np.array(coeffs, dtype=float)

    n = len(coeffs)

    C = np.zeros((n, n))

    idx = np.arange(n - 1)
    C[idx + 1, idx] = 1
    C[:, -1] = -coeffs[::-1]
    print(C)
    roots = np.linalg.eigvals(C)

    return roots

# example:
# print(get_polynomial_roots([1,0,5,-7,2,3]))
