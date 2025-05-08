# Simulates a 2D fish with a non-linear dynamics model solving the differential equations of motion numerically
# author: Umar Masood
# date: 2024-04-01
# version: 1.0
# Bio-inspired Robotics & Control Lab (BRCL) @ UH


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import csv
import datetime



class Fish:
    def __init__(self, x, y, psi, delta, alpha1, alpha2, u = 0.0, v = 0, r = -0.01):
        self.x = x
        self.y = y
        self.psi = psi
        self.delta = delta
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.u = u
        self.v = v
        self.r = r

    def get_state(self):
        return self.x, self.y, self.psi, self.delta, self.alpha1, self.alpha2, self.u, self.v, self.r

    def set_state(self, x, y, psi, delta, alpha1, alpha2, u, v, r):
        self.x = x
        self.y = y
        self.psi = psi
        self.delta = delta
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.u = u
        self.v = v
        self.r = r


    def set_shape(self, l0=0.042, l1=0.058, l2=0.022, d0=0.04, d=0.08, L=0.04, m = 0.9, I = 0.0047):
        self.l0 = l0
        self.l1 = l1
        self.l2 = l2
        self.d0 = d0
        self.d = d
        self.L = L
        self.m = m
        self.I = I

    def plot(self, ax, scale = 1):
        # Plotting the pool
        pool = plt.Rectangle((0, 0), _pool_length, _pool_width, color='blue', alpha=0.2)
        ax.add_patch(pool)

        ax.clear()
        ax.set_xlim(0, _pool_length)
        ax.set_ylim(0, _pool_width)
        ax.set_title('2D Robotic Fish Simulator - BRCL @ UH')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.add_patch(pool)

        # define scaled fish parameters
        d0_scaled = self.d0*scale
        d_scaled = self.d*scale
        l0_scaled = self.l0*scale
        l1_scaled = self.l1*scale
        l2_scaled = self.l2*scale

        fish_body = Ellipse((self.x, self.y), d0_scaled, d_scaled, angle=self.psi*180/np.pi, color='red')

        x0_scaled = self.x - 0.5*d0_scaled*np.cos(self.psi)
        y0_scaled = self.y - 0.5*d0_scaled*np.sin(self.psi)
        x1_scaled = x0_scaled - l0_scaled*np.cos(self.psi + self.delta)
        y1_scaled = y0_scaled - l0_scaled*np.sin(self.psi + self.delta)
        x2_scaled = x1_scaled - l1_scaled*np.cos(self.psi + self.delta + self.alpha1)
        y2_scaled = y1_scaled - l1_scaled*np.sin(self.psi + self.delta + self.alpha1)
        x3_scaled = x2_scaled - l2_scaled*np.cos(self.psi + self.delta + self.alpha2)
        y3_scaled = y2_scaled - l2_scaled*np.sin(self.psi + self.delta + self.alpha2)

        fish_tail0 = plt.Line2D([x0_scaled, x1_scaled], [y0_scaled, y1_scaled], color='red')
        fish_tail1 = plt.Line2D([x1_scaled, x2_scaled], [y1_scaled, y2_scaled], color='red')
        fish_tail2 = plt.Line2D([x2_scaled, x3_scaled], [y2_scaled, y3_scaled], color='red')

        ax.add_patch(fish_body)
        ax.add_line(fish_tail0)
        ax.add_line(fish_tail1)
        ax.add_line(fish_tail2)

    def move(self, omega, _del, t):

        def ode_system(t, states):
            u, v, r = states
            states_dot = np.array([
                (F_x + (self.m + k_22*self.m) * v * r - D_u * u)/(self.m + k_11 * self.m),
                (F_y - (self.m + k_11*self.m) * u * r - D_v * v)/(self.m + k_22 * self.m),
                (F_theta - (k_22 * self.m - k_11 * self.m) * v * u - D_r * r)/(self.I + k_55*self.I)
            ])
            return states_dot
        
        self.delta = _del
        A_1 = 0.1
        A_2 = 0.1
    
        self.alpha1 = A_1*np.sin(omega*t)                       # omega is rad/sec
        self.alpha2 = A_2*np.sin(omega*t + np.pi/2)             # omega is rad/sec

        alpha2_dot = A_2 * omega * np.cos(omega * t + np.pi / 2)  # The derivative of alpha2 with respect to time
        alpha1_dot = A_1 * omega * np.cos(omega * t)  # The derivative of alpha1 with respect to time

        # quarter-chord point of the caudel fin in DSC frame
        xd = self.l1*np.cos(self.alpha1) + self.l2*np.cos(self.alpha2)              # alpha1 and alpha2 are in radians
        yd = self.l1*np.sin(self.alpha1) + self.l2*np.sin(self.alpha2)              # alpha1 and alpha2 are in radians

        xd_dot = -self.l1*np.sin(self.alpha1)*A_1*np.cos(omega*t)*omega - self.l2*np.sin(self.alpha2)*A_2*np.cos(omega*t + np.pi/2)*omega   # xd_dot = d(xd)/dt in m/s
        yd_dot = self.l1*np.cos(self.alpha1)*A_1*np.cos(omega*t)*omega + self.l2*np.cos(self.alpha2)*A_2*np.cos(omega*t + np.pi/2)*omega    # yd_dot = d(yd)/dt in m/s

        xd_ddot =   - self.l1 * A_1**2 * omega**2 * np.cos(A_1 * np.sin(omega * t)) * np.cos(omega * t)**2 \
                    + self.l1 * A_1 * omega**2 * np.sin(A_1 * np.sin(omega * t)) * np.sin(omega * t) \
                    - self.l2 * A_2**2 * omega**2 * np.sin(omega * t)**2 * np.cos(A_2 * np.cos(omega * t)) \
                    + self.l2 * A_2 * omega**2 * np.sin(A_2 * np.cos(omega * t)) * np.cos(omega * t)
        
        yd_ddot =   - self.l1 * A_1**2 * omega**2 * np.sin(A_1 * np.sin(omega * t)) * np.cos(omega * t)**2 \
                    - self.l1 * A_1 * omega**2 * np.cos(A_1 * np.sin(omega * t)) * np.sin(omega * t) \
                    - self.l2 * A_2**2 * omega**2 * np.cos(omega * t)**2 * np.cos(A_2 * np.cos(omega * t)) \
                    - self.l2 * A_2 * omega**2 * np.cos(A_2 * np.cos(omega * t)) * np.sin(omega * t)

        V_n = - xd_dot*np.sin(self.alpha2) + yd_dot*np.cos(self.alpha2) + V_c*np.sin(self.alpha2)
        V_m = xd_dot*np.cos(self.alpha2) + yd_dot*np.sin(self.alpha2) - V_c*np.cos(self.alpha2)

        V_n_dot = - xd_ddot * np.sin(self.alpha2) \
          - xd_dot * alpha2_dot * np.cos(self.alpha2) \
          + yd_ddot * np.cos(self.alpha2) \
          - yd_dot * alpha2_dot * np.sin(self.alpha2) \
          + V_c * alpha2_dot * np.cos(self.alpha2)  # V_c is constant

        V_vect = np.array([V_n, V_m])
        m_hat = np.array([np.cos(self.alpha2), np.sin(self.alpha2)])
        n_hat = np.array([-np.sin(self.alpha2), np.cos(self.alpha2)])

        m_i = 0.5 * np.pi * rho_w * self.d**2
        F_rf_d = - 0.5 * m_i * V_n**2 * m_hat + m_i * V_n * V_m * n_hat - m_i * self.L * V_n_dot * n_hat

        del_transform = 0.2* np.array([[np.cos(self.delta), -np.sin(self.delta)], 
                          [np.sin(self.delta), np.cos(self.delta)]])
        
        F = np.dot(del_transform, F_rf_d) # in Wenyu's paper defined as T_x
        F_x = F[0]
        F_y = F[1]

        F_theta = (self.d0 + (self.l0 + self.l1 + self.l2) * np.cos(self.delta))*F_y - (self.l0 + self.l1 + self.l2) * np.sin(self.delta)*F_x
        
        # append F_theta in the F array
        F = np.append(F, F_theta)

        states = [self.u, self.v, self.r]
        
        ODE_sol = solve_ivp(ode_system, [t - dt, t], states, t_eval=[t])
        
        states = ODE_sol.y[:,-1]

        self.u = states[0]
        self.v = states[1]
        self.r = states[2]

        x_dot = self.u*np.cos(self.psi) - self.v*np.sin(self.psi)
        y_dot = self.u*np.sin(self.psi) + self.v*np.cos(self.psi)
        psi_dot = self.r

        self.x = self.x + x_dot*dt
        self.y = self.y + y_dot*dt
        self.psi = self.psi + psi_dot*dt




        data_logger(t, F, self.u)
        


def data_logger(_t,_F, u = 0, v = 0, r = 0): # function to update the data in the csv file and global variables
    _Fx = _F[0]
    _Fy = _F[1]
    _Ft = _F[2]

    Fx_logged.append(_Fx)
    Fy_logged.append(_Fy)
    Ft_logged.append(_Ft)
    timestamps.append(_t)
    u_logged.append(u)
    v_logged.append(v)
    r_logged.append(r)

    # Update data for each line
    line_fx.set_data(timestamps, Fx_logged)
    line_fy.set_data(timestamps, Fy_logged)
    line_ft.set_data(timestamps, Ft_logged)
    line_u.set_data(timestamps, u_logged)
    line_v.set_data(timestamps, v_logged)
    line_r.set_data(timestamps, r_logged)

    # Adjust the plot range dynamically
    ax_plots.relim()
    ax_plots.autoscale_view(True,True,True)

    ax2_plots.relim()
    ax2_plots.autoscale_view(True,True,True)

    # # Draw the new data
    fig_plots.canvas.draw()
    # fig_plots.canvas.flush_events()

    # Filename with date time
    filename = r'Sim_Data\data_logger_' + program_start_time.strftime("%Y%m%d%H%M%S") + '.csv'

    # Append new data to the csv file
    with open(filename, mode='a', newline='') as data_logger:
        data_logger_writer = csv.writer(data_logger, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_logger_writer.writerow([_t, _Fx, _Fy, _Ft, u, v, r])




# initiaze constants
_pool_width = 5 # meters
_pool_length = 10 # meters
_pool_depth = 2 # meters
_water_filled_frac = 0.9 # fraction of pool filled with water
_water_filled_depth = _pool_depth * _water_filled_frac # meters


# initialize fish parameters
_fish_l0 = 0.042 # meters
_fish_l1 = 0.058 # meters
_fish_l2 = 0.044 # meters
_fish_L = 0.04 # meters
_fish_d0 = 0.04 # meters
_fish_d = 0.08 # meters
_fish_m = 0.9 # kg
_fish_I = 0.0047 # kg*m^2

# initialize fish state variables
_fish_x = 2 # meters
_fish_y = 2 # meters
_fish_psi = 0 # np.pi/2 # radians anlge of fish wrt x-axis
_fish_delta = 0 # radians angle of fish tail (l0) wrt fish body
_fish_alpha1 = 0 # radians angle between l0 and l1
_fish_alpha2 = 0 # radians angle between l0 and l2


Fx_logged = []
Fy_logged = []
Ft_logged = []
u_logged = []
v_logged = []
r_logged = []
timestamps = []

program_start_time = datetime.datetime.now()

# second order system parameters
rho_w = 1025
c_x = 0.39
c_y = 2.2
c_r = 0.0055
V_c = 0.08
s_y = 0.02
s_x = 0.0025
k_11 = 0.095
k_22 = 0.83
k_55 = 0.55

D_v = 0.5
D_u = 0.0
D_r = 0.1


# initial condition
v0 = 0
u0 = 0
r0 = 0
x0 = 0
y0 = 0
psi0 = 0

x0 = [u0, v0, r0]



# Setting up the figure and axis for the plot
fig, ax = plt.subplots()
ax.set_xlim(0, _pool_length)
ax.set_ylim(0, _pool_width)
ax.set_title('2D Robotic Fish Simulator - BRCL @ UH')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_aspect('equal')
ax.grid(True)



# Initialize the plot only once, outside of the function
fig_plots, ax_plots = plt.subplots()
ax2_plots = ax_plots.twinx()

# Set up the plot style
ax_plots.set_xlabel('Time (s)')
ax_plots.set_ylabel('Force (N)')
ax_plots.set_title('Force vs Time')
line_fx, = ax_plots.plot(timestamps, Fx_logged, label='Fx', color='red')
line_fy, = ax_plots.plot(timestamps, Fy_logged, label='Fy', color='green')
line_ft, = ax_plots.plot(timestamps, Ft_logged, label='Ft', color='blue')
line_u, = ax_plots.plot(timestamps, u_logged, label='u', color='black')
line_v, = ax2_plots.plot(timestamps, v_logged, label='v', color='orange')
line_r, = ax2_plots.plot(timestamps, r_logged, label='r', color='purple')
ax_plots.grid(True)
ax_plots.legend()
ax2_plots.legend()
ax_plots.autoscale_view(True,True,True)
# fit axis 2
ax2_plots.relim()

# plt.ion()  # Turn on interactive plotting

# Creating the fish object
roboticfish = Fish(_fish_x, _fish_y, _fish_psi, _fish_delta, _fish_alpha1, _fish_alpha2)
roboticfish.set_shape(_fish_l0, _fish_l1, _fish_l2, _fish_d0*1.5, 0.015)

# Initialization function for the animation
def init():
    roboticfish.plot(ax)

    return []

# Update function for the animation
def update(frame):
    # move to the right
    roboticfish.move(1, 0*np.pi/180, (frame + 1)*dt)
    
    # update the plot
    
    roboticfish.plot(ax, 10)
    return []


# Time step for the simulation
dt = 0.1
# Creating the animation

anim = FuncAnimation(fig, update, init_func=init, frames=3000, interval=dt, blit=True, repeat=False)

# To display the animation in a Jupyter notebook, use the following line:
# from IPython.display import HTML
# HTML(anim.to_jshtml())

# To save the animation as an mp4 video file, uncomment the line below:
# anim.save('fish_simulation.mp4', writer='ffmpeg')

plt.show()


