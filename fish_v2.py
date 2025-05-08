from vpython import canvas, color, vector, cylinder, ellipsoid, rate, box
import numpy as np
from scipy.integrate import solve_ivp

class Fish:
    def __init__(self, x, y, psi, delta, alpha1, alpha2, u=0.0, v=0, r=-0.0):
        # Fish state variables
        # Units: x, y, u, v in meters; psi, delta, alpha1, alpha2, r in radians
        self.x = x
        self.y = y
        self.psi = psi
        self.delta = delta
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.u = u
        self.v = v
        self.r = r
        self.shape_set = False

        # Fish state history
        self.x_hist = []
        self.y_hist = []
        self.psi_hist = []
        self.delta_hist = []
        self.u_hist = []
        self.v_hist = []
        self.r_hist = []


    def set_shape(self, l0=0.06, l1=0.1176, l2=0.097, d0=0.145, w = 0.110, h = 0.155, m=4.14, I=0.049, trail=False):
        # Scale will also change the simulation (don't use anything other than 1)
        self.l0, self.l1, self.l2 = l0, l1, l2
        self.d0, self.w, self.h = d0, w, h
        self.m, self.I = m, I
        self.shape_set = True

        self.body = ellipsoid(pos=vector(self.x, self.y, 0), length= self.d0*2, height=self.h , width = self.w,
                              color=color.gray(0.6), axis=vector(np.cos(self.psi), np.sin(self.psi), 0), 
                              make_trail=trail, trail_type = "points", trail_color = color.red)

        self.tail_segment1 = cylinder(pos=self.body.pos + vector(-self.body.length/2, 0, 0),
                                    axis=vector(-self.l0 * np.cos(self.delta), -self.l0 * np.sin(self.delta), 0),
                                    radius=0.015, color=color.red)

        self.tail_segment2 = cylinder(pos=self.tail_segment1.pos + self.tail_segment1.axis,
                                    axis=vector(-self.l1 * np.cos(self.delta + self.alpha1), 
                                                -self.l1 * np.sin(self.delta + self.alpha1), 0),
                                    radius=0.012, color=color.red)

        self.tail_segment3 = cylinder(pos=self.tail_segment2.pos + self.tail_segment2.axis,
                                    axis=vector(-self.l2 * np.cos(self.delta + self.alpha2), 
                                                -self.l2 * np.sin(self.delta + self.alpha2), 0),
                                    radius=0.01, color=color.red)


    def update_position(self):
        # Update the fish position and angles
        self.body.pos = vector(self.x, self.y, 0)
        self.body.axis = vector(np.cos(self.psi), np.sin(self.psi), 0)

        # Calculating tail segment positions based on delta, alpha1, alpha2
        x0 = self.x - self.d0 * np.cos(self.psi)
        y0 = self.y - self.d0 * np.sin(self.psi)
        x1 = x0 - self.l0 * np.cos(self.psi + self.delta)
        y1 = y0 - self.l0 * np.sin(self.psi + self.delta)
        x2 = x1 - self.l1 * np.cos(self.psi + self.delta + self.alpha1)
        y2 = y1 - self.l1 * np.sin(self.psi + self.delta + self.alpha1)
        x3 = x2 - self.l2 * np.cos(self.psi + self.delta + self.alpha2)
        y3 = y2 - self.l2 * np.sin(self.psi + self.delta + self.alpha2)

        # Update each tail segment's position and orientation
        self.tail_segment1.pos = vector(x0, y0, 0)
        self.tail_segment1.axis = vector(x1 - x0, y1 - y0, 0)
        self.tail_segment2.pos = vector(x1, y1, 0)
        self.tail_segment2.axis = vector(x2 - x1, y2 - y1, 0)
        self.tail_segment3.pos = vector(x2, y2, 0)
        self.tail_segment3.axis = vector(x3 - x2, y3 - y2, 0)

    def move(self, omega, _del, t, dt):
        
        # second order system parameters
        a_e = 0.234
        b_e = 0.065
        rho_w = 1025
        c_x = 0.04
        c_y = 18.5
        c_r = 0.49
        V_c = 0.0
        # AREA OF CIRCLE IN a_e
        A_u = np.pi * a_e**2
        # AREA OF ECCENTRIC CIRCLE IN a_e and b_e
        A_v = np.pi * a_e * b_e
        k_11 = 0.0947
        k_22 = 0.8408
        k_55 = 0.5587
        

        def ode_system(t, states):
            u, v, r = states

            D_u = rho_w * c_x * A_u * np.abs(u)
            D_v = rho_w * c_y * A_v * np.abs(v)
            D_r = c_r * np.abs(r)

            states_dot = np.array([
                (F_x + (self.m + k_22*self.m) * v * r - D_u * u)/(self.m + k_11 * self.m),
                (F_y  - (self.m + k_11*self.m) * u * r - D_v * v)/(self.m + k_22 * self.m),
                (F_theta - (k_11 * self.m - k_22 * self.m) * v * u - D_r * r)/(self.I + k_55*self.I)
            ])
            return states_dot
        # 
        self.delta = _del

        self.delta_hist.append(self.delta)

        # A_1 and A_2 are the amplitudes (in rad) of the oscillations of the caudal fin
        A_1 = 10*np.pi/180
        A_2 = 13*np.pi/180
    
        self.alpha1 = A_1*np.sin(omega*t)                       # omega is rad/sec
        self.alpha2 = A_2*np.sin(omega*t + np.pi/2)             # omega is rad/sec

        alpha1_dot = A_1 * omega * np.cos(omega * t)  # The derivative of alpha1 with respect to time
        alpha2_dot = - A_2 * omega * np.sin(omega * t)  # The derivative of alpha2 with respect to time

        alpha1_ddot = - A_1**2 * omega**2 * np.sin(omega * t)  # The second derivative of alpha1 with respect to time
        alpha2_ddot = - A_2**2 * omega**2 * np.cos(omega * t)  # The second derivative of alpha2 with respect to time

        # quarter-chord point of the caudel fin in DSC frame
        xd = self.l1*np.cos(self.alpha1) + self.l2*np.cos(self.alpha2)              # alpha1 and alpha2 are in radians
        yd = self.l1*np.sin(self.alpha1) + self.l2*np.sin(self.alpha2)              # alpha1 and alpha2 are in radians

        xd_dot = -self.l1*np.sin(self.alpha1)*alpha1_dot - self.l2*np.sin(self.alpha2)*alpha2_dot   # xd_dot = d(xd)/dt in m/s
        yd_dot = self.l1*np.cos(self.alpha1)*alpha1_dot + self.l2*np.cos(self.alpha2)*alpha2_dot    # yd_dot = d(yd)/dt in m/s

        xd_ddot =   - self.l1 * (np.sin(self.alpha1) * alpha1_ddot + alpha1_dot**2 * np.cos(self.alpha1)) \
                    - self.l2 * (np.sin(self.alpha2) * alpha2_ddot + alpha2_dot**2 * np.cos(self.alpha2))
        
        yd_ddot =   self.l1 * (np.cos(self.alpha1) * alpha1_ddot - alpha1_dot**2 * np.sin(self.alpha1)) \
                    + self.l2 * (np.cos(self.alpha2) * alpha2_ddot - alpha2_dot**2 * np.sin(self.alpha2))

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

        m_i = 0.25 * np.pi * rho_w * self.h**2

        self.L = self.d0*2 + self.l0 + self.l1 + self.l2

        F_rf_d =  0.5 * m_i * V_n**2 * m_hat - m_i * V_n * V_m * n_hat + m_i * self.L * V_n_dot * n_hat

        del_transform = 0.45 * np.array([[np.cos(self.delta), -np.sin(self.delta)], 
                          [np.sin(self.delta), np.cos(self.delta)]])
        
        F = np.dot(del_transform, F_rf_d) # in Wenyu's paper defined as T_x
        F_x = F[0]
        F_y = F[1]

        F_theta = - (self.d0 + (self.l0 + self.l1 + self.l2) * np.cos(self.delta))*F_y + (self.l0 + self.l1 + self.l2) * np.sin(self.delta)*F_x
        
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

        # Append the fish state to the history
        self.x_hist.append(self.x)
        self.y_hist.append(self.y)
        self.psi_hist.append(self.psi)
        self.delta_hist.append(self.delta)
        self.u_hist.append(self.u)
        self.v_hist.append(self.v)
        self.r_hist.append(self.r)


        self.update_position()


class Pool:
    def __init__(self,wall_thickness=0.1, length=1.0, width=1.0, depth=0.5, water_level=0.0):
        self.length = length
        self.width = width
        self.depth = depth
        self.water_level = water_level
        self.wall_thickness = wall_thickness

    def plot(self, scene, walls_color=color.green, water_color=color.blue, opacity=0.5):
        # plot walls
        self.wall_front = box(pos=vector(self.length/2 , 0, self.depth/2 - self.water_level), length=self.length, height=self.wall_thickness, width=self.depth, color=walls_color, opacity=opacity)
        self.wall_back = box(pos=vector(self.length/2, self.width, self.depth/2 - self.water_level), length=self.length, height=self.wall_thickness, width=self.depth, color=walls_color, opacity=opacity)
        self.wall_left = box(pos=vector(0, self.width/2, self.depth/2 - self.water_level), length=self.wall_thickness, height=self.width, width=self.depth, color=walls_color, opacity=opacity)
        self.wall_right = box(pos=vector(self.length, self.width/2, self.depth/2 - self.water_level), length=self.wall_thickness, height=self.width, width=self.depth, color=walls_color, opacity=opacity)
        self.wall_bottom = box(pos=vector(self.length/2, self.width/2, -self.water_level -self.wall_thickness/2), length=self.length, height=self.width, width=self.wall_thickness, color=walls_color, opacity=opacity)

        # plot water
        self.water = box(pos=vector(self.length/2, self.width/2, -self.water_level/2), length=self.length, height=self.width, width=self.water_level, color=water_color)
        # change the opacity of the water
        self.water.opacity = 0.5
            

class Pipeline:
    def __init__(self, scene, _pool, _start, _end, _radius):
        # _start and _end 2D vectors in list format with x and y coordinates
        # pipeline always in the bottom of the pool
        self.start = vector(_start[0], _start[1], -_pool.water_level + _radius)
        self.end = vector(_end[0], _end[1], -_pool.water_level + _radius)
        self.radius = _radius
        self.axis = self.end - self.start
        self.length = np.linalg.norm(np.array([self.axis.x, self.axis.y, self.axis.z]))

        self.pipe = cylinder(pos=self.start, axis=self.axis, radius=self.radius, color=color.gray(0.5))
        


        