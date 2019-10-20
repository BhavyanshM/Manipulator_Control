import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')


fig = plt.figure(figsize=(18,18))
ax = plt.axes(xlim=(-500, 500), ylim=(-500, 500))
lines = []
n_lines = 4

traj_x = []
traj_y = []

line, = ax.plot([], [], lw=10)
traj, = ax.plot([], [], lw=2)


l = np.array([200, 200])
j = np.array([	[0,0],
				[10, -10],
				[10, -20]])

def init():
    return line,traj

def animate(i):
	print(i)
	r = np.array([360*np.sin(i*0.01),360*np.sin(i*0.015)])*np.pi/180

	j[1,0] = l[0]*np.sin(r[0])
	j[1,1] = l[0]*np.cos(r[0])
	j[2,0] = l[1]*np.sin(r[0]+r[1]) + l[0]*np.sin(r[0])
	j[2,1] = l[1]*np.cos(r[0]+r[1]) + l[0]*np.cos(r[0])

	traj_x.append(j[2,0])
	traj_y.append(-1*j[2,1])

	traj.set_data(traj_x, traj_y)

	a = j[:,0]
	b = -1*j[:,1]

	line.set_data(a, b)

	return line,traj

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=10000, interval=30, blit=True)

# mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
plt.show()