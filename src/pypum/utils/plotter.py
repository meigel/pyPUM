from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('WxAgg')
#matplotlib.interactive(True)
import matplotlib.pyplot as plt
import mayavi.mlab as mlab 
from math import ceil, pi

class Plotter(object):
    @classmethod
    @mlab.show
    def plot(cls, F, geomdim, domain=[[0, 1]], resolution=1 / 50, vectorized=True, style="-", opacity=1.0):
        try:
            F[0]
        except:
            F = (F,)
            domain = (domain,)
        try:
            N = len(F)
        except:
            N = 1
        if len(domain) != N:
            domain = domain * N
        for f, dom in zip(F, domain):
            p = cls.get_points(geomdim, dom, resolution)
            if vectorized:
                if geomdim == 1:
                    x = p[0]
                if geomdim == 2:
                    x = np.vstack((p[0].flatten(), p[1].flatten())).T
                y = f(x)
            else:
                y = np.array([f(p[0][i], p[1][i]) for i in range(len(p[0]))])
            if geomdim == 2:
                y.shape = p[0].shape
            cls.plot_data(geomdim, p, y, style=style, opacity=opacity)

    @classmethod
    def plot_data(cls, geomdim, p, data, style="-", opacity=1.0):
        if geomdim == 1:
            # 2d plot
            plt.plot(p[0], data, style)
#            plt.ylim([-1, 1]) 
            plt.show()
        else:
            # 3d plot
            assert geomdim == 2
            if style == "-":
                mlab.surf(p[0], p[1], data, opacity=opacity)
            else:
                mlab.points3d(p[0], p[1], data)
    
    @classmethod
    def get_points(cls, geomdim, domain, resolution=1 / 50):
        if geomdim == 1:
            dx = domain[1] - domain[0]
            N = ceil(dx / resolution)
            p = (np.linspace(domain[0], domain[1] + dx, num=N),)
        elif geomdim == 2:
            x, y = np.mgrid[domain[0][0]:domain[0][1]:resolution, domain[1][0]:domain[1][1]:resolution]
            p = (x, y)
        else:
            assert("unsupported dimension %i" % geomdim)
        return p

##f1 = (lambda x:np.sin(2 * pi * x), lambda x: x * x)
##Plotter.plot(f1, 1, [[0, 2]], resolution=1 / 100)
##Plotter.plot(f1, 1, ([0, 2], [1, 3]), resolution=1 / 100)
#f = lambda x:np.sin(2 * pi * x[:, 0]) * np.sin(5 * pi * x[:, 1])
##Plotter.plot(f, 2, [[0, 1], [1, 2]], resolution=1 / 50, opacity=0.5)
##f2 = (lambda x:np.sin(2 * pi * x[:, 0]) * np.sin(5 * pi * x[:, 1]), lambda x:x[:, 0] ** 2 + x[:, 1])
#f2 = (lambda x:np.sin(2 * pi * x[:, 0]) * np.sin(5 * pi * x[:, 1]), lambda x:x[:, 0] * 0 + 1)
##Plotter.plot(f2, 2, [[[0, 1], [1, 2]]], resolution=1 / 50, opacity=0.5)
#Plotter.plot(f2, 2, [[[0, 1], [0, 1]], [[1, 2], [1, 2]]], resolution=1 / 50, opacity=0.5)
