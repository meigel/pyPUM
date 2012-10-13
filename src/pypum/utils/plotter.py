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
    def plot(cls, f, domain=[[0, 1]], resolution=1 / 50, vectorized=True, style="-"):
        geomdim = len(domain)
        p = cls.get_points(domain, resolution)
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
        cls.plot_data(p, y)

    @classmethod
    @mlab.show
    def plot_data(cls, p, data, style="-"):
        geomdim = len(p)
        if geomdim == 1:
            # 2d plot
            plt.plot(p[0], data, style)
#            plt.ylim([-1, 1]) 
            plt.show()
        else:
            # 3d plot
            assert geomdim == 2
            if style == "-":
                mlab.surf(p[0], p[1], data, warp_scale="auto")
            else:
                mlab.points3d(p[0], p[1], data)
    
    @classmethod
    def get_points(cls, domain=[[0, 1]], resolution=1 / 50):
        geomdim = len(domain) 
        if geomdim == 1:
            dx = domain[0][1] - domain[0][0]
            N = ceil(dx / resolution)
            p = (np.linspace(domain[0][0], domain[0][1] + dx, num=N),)
        elif geomdim == 2:
            x, y = np.mgrid[domain[0][0]:domain[0][1]:resolution, domain[1][0]:domain[1][1]:resolution]
            p = (x, y)
        else:
            assert("unsupported dimension %i" % geomdim)
        return p

#Plotter.plot(lambda x:np.sin(2 * pi * x), [[0, 2]], resolution=1 / 100)
#f = lambda x:np.sin(2 * pi * x[:, 0]) * np.sin(5 * pi * x[:, 1])
#Plotter.plot(f, [[0, 1], [1, 2]], resolution=1 / 50)
