import numpy as np

def croutLU(A, b):
    n = len(b)
    L = np.zeros((n, n))
    U = np.eye(n, n)

    for j in range(n):
        for i in range(j, n):
            sum_val = np.dot(L[i, :j], U[:j, j])
            L[i, j] = A[i, j] - sum_val

        for i in range(j + 1, n):
            sum_val = np.dot(L[j, :j], U[:j, i])
            U[j, i] = (A[j, i] - sum_val) / L[j, j]

    print(L)
    print(U)

    # 대입법
    z = np.zeros((n, 1))
    z[0] = b[0] / L[0, 0]
    for i in range(1, n):
        z[i] = (b[i] - np.dot(L[i, 0:i], z[0:i])) / L[i, i]

    # 역대입법
    x = np.zeros((n, 1))
    x[n - 1] = z[n - 1] / U[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (z[i] - np.dot(U[i, i + 1:n], x[i + 1:n])) / U[i, i]

    return x

def doolittleLU(A, b):
    n = len(b)
    U = np.zeros((n, n))
    L = np.eye(n, n)

    for j in range(n):
        for i in range(j, n):
            sum_val = np.dot(L[j, :j], U[:j, i])
            U[j, i] = A[j, i] - sum_val

        for i in range(j + 1, n):
            sum_val = np.dot(L[i, :j], U[:j, j])
            L[i, j] = (A[i, j] - sum_val) / U[j, j]

    print(L)
    print(U)

    # 대입법
    z = np.zeros((n, 1))
    z[0] = b[0] / L[0, 0]
    for i in range(1, n):
        z[i] = (b[i] - np.dot(L[i, 0:i], z[0:i])) / L[i, i]

    # 역대입법
    x = np.zeros((n, 1))
    x[n - 1] = z[n - 1] / U[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (z[i] - np.dot(U[i, i + 1:n], x[i + 1:n])) / U[i, i]

    return x

A = np.array([[2, 4, 0, 2, 6], [1, 5, 3, -2, 9], [2, 3, 3, \
11, 0], [1, 4, 3, 6, 11], [3, 7, -1, 0, 21] ])

b = np.array([14, 16, 19, 25, 30])

print(croutLU(A, b))
print(doolittleLU(A, b))