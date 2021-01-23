# a module which contains mathematical functions for machine learning
from matplotlib import cm


def d_point_line(a, b, p):
    # d_point_line(a,b,p):
    # Distance of a point from a line. a and b are intercept and slope of the line p is a pandas data frame. Each row represent a point. The first column contains xs and the second column contains ys. Equation of the lin is given as y = ax + b we can rewrite as ax - y + b = 0 And the distance is calculated as follows:
    # ∣𝑎𝑥𝑝−𝑦𝑝+𝑏∣/(𝑎^2+1)
    import pandas as pd
    import numpy as np
    import math
    return p.join(pd.DataFrame({'Distance': abs(a * p['x'] - p['y'] + b) / math.sqrt(a ** 2 + 1)}))


def ms(a, b, p):
    import numpy as np
    results = d_point_line(a, b, p)
    return np.mean(results['Distance'] ** 2)


def visualize_ab_min(a, b, p, N):
    # This function takes a dataframe of points as the input. This is called p. Then use scikit to calculate
    # regression line. The intercept of this line is $a$ and its slop is $b$. After that the following variables are
    # produced: a_vector = numpy.linspace(.5*a,1.5*a,N) b_vector = numpy.linspace(.5*b,1.5*b,N) In each pair of
    # intercept and slop, it calculate the distances of points from the line. The results are ploted in 3D surface
    # and it can be seen the it gives a minimum at the intercept and slop of intercept.
    
    a_vector = np.linspace(.5 * a, 2 * a, N)
    b_vector = np.linspace(.5 * b, 2 * b, N)
    A, B = np.meshgrid(a_vector, b_vector)
    Z = np.zeros(np.shape(A))
    for i in range(N):
        for j in range(N):
            Z[i, j] = ms(A[i, j], B[i, j], p)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(A, B, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # from sklearn import linear_model
    # regr = linear_model.LinearRegression()
    # regr.fit(diabetes_X_train, diabetes_y_train)
    # diabetes_y_pred = regr.predict(diabetes_X_test)

    p = pd.DataFrame({'x': [3, 4, 2, 5, -1], 'y': [7, 3, 1, 1, 4]})
    a, b = np.polyfit(p['x'], p['y'], 1)
    print(a)

    plt.scatter(p['x'], p['y'])
    minX = min(p['x'])
    maxX = max(p['x'])
    plt.plot([minX, maxX], [a * minX + b, a * maxX + b], 'r')

    d_point_line(1, 2, p)
    N = 10
    visualize_ab_min(a, b, p, N)
