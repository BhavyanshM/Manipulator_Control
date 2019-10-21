import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')


fig = plt.figure(figsize=(18,18))
ax = plt.axes(xlim=(0, 100), ylim=(-1, 5))
lines = []
n_lines = 1
for i in range(n_lines):
	line_i, = ax.plot([], [], lw=3)
	lines.append(line_i)

def firstorder(y,t):
    tau = 5.0
    k = 2.0
    u = 1.0
    dydt = (-y + k*u)/tau
    return dydt


def init():
    return tuple(lines)

def animate(i):

    t = np.linspace(0, i*0.1, 1000)
    y = odeint(firstorder, 0, t)

    for k in range(n_lines):
    	lines[k].set_data(t, y)

    return tuple(lines)

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=1000, interval=10, blit=True)

plt.show()