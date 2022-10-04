import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

x = np.linspace(0, 10, num=60)
y = np.linspace(0, 10, num=60)
z = np.linspace(0, 10, num=60)


x2 = np.linspace(0, 10, num=60)
y2 = np.linspace(0, -10, num=60)
z2 = np.linspace(0, 10, num=60)

experiment_data_dir = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/experiment_data/robot_factor_exp/'
subject_id = 2
factor_id = 4
robot_hand_trajs = np.genfromtxt(experiment_data_dir + 'subject_' + str(subject_id) + '/' + 'factor_' + str(factor_id) + '/robot_hand_position_trajectories.csv')
robot_hand_traj = robot_hand_trajs[1:61]

human_hand_trajs = np.genfromtxt(experiment_data_dir + 'subject_' + str(subject_id) + '/' + 'factor_' + str(factor_id) + '/human_hand_position_trajectories.csv')
human_hand_traj = human_hand_trajs[1:61]

x = robot_hand_traj[:, 0]
y = robot_hand_traj[:, 1]
z = robot_hand_traj[:, 2]

x2 = human_hand_traj[:, 0]
y2 = human_hand_traj[:, 1]
z2 = human_hand_traj[:, 2]




#1 colored by value of `z`
# ax.scatter(x, y, z, c = plt.cm.jet(z/max(z)))

#2 colored by index (same in this example since z is a linspace too)
N = len(z)
cmap = mpl.colors.ListedColormap(mpl.cm.Blues(np.linspace(0, 1, 20))[5:, :-1])
cmap2 = mpl.colors.ListedColormap(mpl.cm.Reds(np.linspace(0, 1, 20))[5:, :-1])

ax.scatter(x, y, z, s=20, c = range(N), cmap=cmap, alpha=1.0)
ax.scatter(x2, y2, z2,s=20, c = range(N), cmap=cmap2, alpha=1.0)

for i in range(N-1):
    ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], c=mpl.cm.Blues((i + 15)/N))
    ax.plot(x2[i:i + 2], y2[i:i + 2], z2[i:i + 2], c=mpl.cm.Reds((i + 15) / N))

ax.set_xlabel('$X$', fontsize=10)
ax.set_ylabel('$Y$', fontsize=10)
ax.set_zlabel('$Z$', fontsize=10)

plt.show()