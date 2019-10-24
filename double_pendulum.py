import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
from matplotlib.patches import Circle

plt.style.use('seaborn-pastel')


fig = plt.figure(figsize=(18,18))
ax = plt.axes(xlim=(-500, 500), ylim=(-500, 500))
lines = []
n_lines = 4

traj_x = []
traj_y = []

line, = ax.plot([], [], lw=10)
traj, = ax.plot([], [], lw=1)


l = np.array([200, 200])
j = np.array([	[0,0],
				[10, -10],
				[10, -20]])

# Pendulum rod lengths (m), bob masses (kg).
L1, L2 = 200, 200
M1, M2 = 100, 10
# The gravitational acceleration (m.s-2).
g = 9.81

y0 = np.array([np.pi+0.01, 0, np.pi, 0])
theta1 = y0[0]
theta2 = y0[2]
theta1dot= y0[1]
theta2dot= y0[3]

def init():
    return line,traj

def calc_E(y):
	"""Return the total energy of the system."""
	# print(y)
	th1, th1d, th2, th2d = y[0], y[1], y[2], y[3]
	V = -(M1+M2)*L1*g*np.cos(th1) - M2*L2*g*np.cos(th2)
	T = 0.5*M1*(L1*th1d)**2 + 0.5*M2*((L1*th1d)**2 + (L2*th2d)**2 +\
			 2*L1*L2*th1d*th2d*np.cos(th1-th2))
	return T + V

def deriv(y, t, L1, L2, m1, m2):
	"""Return the first derivatives of y = theta1, z1, theta2, z2."""
	theta1, z1, theta2, z2 = y

	c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

	theta1dot = z1
	z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) - \
			 (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
	theta2dot = z2
	z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + \
			m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
	return theta1dot, z1dot, theta2dot, z2dot

def animate(i):
	global j, theta1, theta2, theta1ddot, theta2ddot, theta2dot, theta1dot
	# print(i)

	# Maximum time, time point spacings and the time grid (all in s).
	dt = 0.00043
	# t = np.arange(0, tmax+dt, dt)
	# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
	for i in range(1000):
		theta1ddot=(-g*(2*M1+M2)*np.sin(theta1)-M2*g*np.sin(theta1-2*theta2)-2*np.sin(theta1-theta2)*M2*(L2*theta2dot**2+L1*np.cos(theta1-theta2)*theta1dot**2))/(L1*(2*M1+M2-M2*np.cos(2*theta1-2*theta2)))
		theta2ddot=(2*np.sin(theta1-theta2)*((M1+M2)*L1*theta1dot**2+g*(M1+M2)*np.cos(theta1)+L2*M2*np.cos(theta1-theta2)*theta2dot**2))/(L2*(2*M1+M2-M2*np.cos(2*theta1-2*theta2)))
		theta2dot=theta2dot+theta2ddot*dt
		theta1dot=theta1dot+theta1ddot*dt
		theta1=theta1+theta1dot*dt
		theta2=theta2+theta2dot*dt

	x1=L1*np.sin(theta1)
	y1=-L1*np.cos(theta1)
	x1dot=L1*theta1dot*np.cos(theta1)
	y1dot=L1*theta1dot*np.sin(theta1)
	x2=x1+L2*np.sin(theta2)
	y2=y1-L2*np.cos(theta2)
	x2dot=L1*theta1dot*np.cos(theta1)+L2*theta2dot*np.cos(theta2)
	y2dot=L1*theta1dot*np.sin(theta1)+L2*theta2dot*np.sin(theta2)
	T=.5*M1*(x1dot**2+y1dot**2)+.5*M2*(x2dot**2+y2dot**2)
	U=M1*g*y1+M2*g*y2
	print(T + U)

	rel_theta_21 = theta2-theta1
	r = np.array([theta1,rel_theta_21])
	# print(r)

	j[1,0] = L1*np.sin(r[0])
	j[1,1] = L1*np.cos(r[0])
	j[2,0] = L2*np.sin(r[0]+r[1]) + L1*np.sin(r[0])
	j[2,1] = L2*np.cos(r[0]+r[1]) + L1*np.cos(r[0])

	traj_x.append(j[2,0])
	traj_y.append(-1*j[2,1])

	traj.set_data(traj_x, traj_y)

	a = j[:,0]
	b = -1*j[:,1]

	# radius = 10

	# c0 = Circle((0, 0), radius, fc='k', zorder=10)
	# c1 = Circle((j[1,0], j[1,1]), radius, fc='b', ec='b', zorder=10)
	# c2 = Circle((j[2,0], j[2,1]), radius, fc='r', ec='r', zorder=10)
	# ax.add_patch(c0)
	# ax.add_patch(c1)
	# ax.add_patch(c2)

	line.set_data(a, b)

	return line,traj

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=10000, interval=1, blit=True)

# mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
plt.show()