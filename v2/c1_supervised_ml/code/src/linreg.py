import numpy as np


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2

    total_cost = 1 / (2 * m) * cost
    return total_cost


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b

        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]

        dj_db += dj_db_i
        dj_dw += dj_dw_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    w = w_in
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        b = b - alpha * dj_db
        w = w - alpha * dj_dw

    return w, b


if __name__ == '__main__':
    # Sample data
    x_train = np.array([1.0, 2.0])
    y_train = np.array([300.0, 500.0])

    # Initialize parameters
    w_init = 0
    b_init = 0

    # Gradient descent settings
    iterations = 10000
    alpha = 1.0e-2

    # Run gradient descent
    w_final, b_final = gradient_descent(x_train,
                                        y_train,
                                        w_init,
                                        b_init,
                                        alpha,
                                        iterations,
                                        compute_cost,
                                        compute_gradient)

    # Print results
    print(f'Final w: {w_final}, Final b: {b_final}')
