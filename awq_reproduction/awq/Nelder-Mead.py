
def Nelder_Mead(objective_func, initial_points, alpha=1, gamma=2, rho=0.5, sigma=0.5, max_iter=1000, tol=1e-6):
    '''
    This function implements the Nelder-Mead algorithm for optimization.
    '''
    # Initialize the simplex
    n = len(initial_points)
    simplex = initial_points
    values = [objective_func(x) for x in simplex]
    iter = 0

    while iter < max_iter:
        # Step 1: Order the simplex
        order = sorted(range(n), key=lambda k: values[k])
        simplex = [simplex[k] for k in order]
        values = [values[k] for k in order]

        # Step 2: Calculate the centroid of the simplex excluding the worst point
        centroid = sum([simplex[i] for i in range(n)]) / n

        # Step 3: Reflect the worst point through the centroid
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = objective_func(x_r)

        if values[0] <= f_r < values[-2]:
            simplex[-1], values[-1] = x_r, f_r
            iter += 1
            continue

        # Step 4: Expand the reflected point
        if f_r < values[0]:
            x_e = centroid + gamma * (x_r - centroid)
            f_e = objective_func(x_e)

            if f_e < f_r:
                simplex[-1], values[-1] = x_e, f_e
            else:
                simplex[-1], values[-1] = x_r, f_r

            iter += 1
            continue

        # Step 5: Contract the worst point
        x_c = centroid + rho * (simplex[-1] - centroid)
        f_c = objective_func(x_c)

        if f_c < values[-1]:
            simplex[-1], values[-1] = x_c, f_c
            iter += 1
            continue

        # Step 6: Shrink the simplex
        for i in range(1, n):
            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
            values[i] = objective_func(simplex[i])

        iter += 1

    return simplex[0], values[0]


# Example
if __name__ == '__main__':
    def objective_func(x):
        return sum([xi ** 2 for xi in x])

    initial_points = [[0, 0], [1, 0], [0, 1]]
    result = Nelder_Mead(objective_func, initial_points)
    print(result)
