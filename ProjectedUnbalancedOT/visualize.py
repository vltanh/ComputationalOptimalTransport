import numpy as np
from scipy.linalg import null_space
import plotly.graph_objs as go
import matplotlib.pyplot as plt


def get_plane(M, v, a=5):
    I = np.nonzero(v)[0]
    if len(I) == 0:
        raise ValueError('v is the null vector and cannot be the normal vec')

    D = np.dot(M, v)
    if I[0] == 0:
        y = np.linspace(-a, a, 100)
        y, z = np.meshgrid(y, y)
        x = - (v[1]*y + v[2]*z) / v[0] + D
        return x, y, z
    elif I[0] == 1:
        x = np.linspace(-a, a, 100)
        x, z = np.meshgrid(x, x)
        y = -(v[0]*x + v[2]*z) / v[1] + D
        return x, y, z
    else:
        x = np.linspace(-a, a, 100)
        x, y = np.meshgrid(x, x)
        z = -(v[0]*x + v[1]*y) / v[2] + D
        return x, y, z


def visualize_3D_plan(X, Y, U, pi):
    n, m = pi.shape

    data = []
    for i in range(n):
        for j in range(m):
            data.append(go.Scatter3d(
                x=[X[i, 0], Y[j, 0]],
                y=[X[i, 1], Y[j, 1]],
                z=[X[i, 2], Y[j, 2]],
                line=dict(
                    width=pi[i, j] * 50,
                    color='black',
                ),
                marker=dict(
                    size=5,
                    color=['blue', 'red'],
                    opacity=0.5,
                ),
                showlegend=False
            ))

    xp, yp, zp = get_plane(np.zeros(3), null_space(U.T).reshape(-1))
    data += [
        go.Surface(x=xp, y=yp, z=zp,
                   opacity=0.1,
                   showscale=False)
    ]

    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    )

    plot_figure = go.Figure(data=data, layout=layout)

    plot_figure.show()


def visualize_2D_plan(X, Y, U, pi):
    n, m = pi.shape

    projX = X @ U
    projY = Y @ U

    plt.figure(figsize=(10, 10), dpi=100)
    plt.scatter(projX[:, 0], projX[:, 1])
    plt.scatter(projY[:, 0], projY[:, 1])
    for i in range(n):
        for j in range(m):
            plt.plot([projX[i, 0], projY[j, 0]],
                     [projX[i, 1], projY[j, 1]],
                     lw=pi[i, j] * 10,
                     color='black')
    plt.show()
