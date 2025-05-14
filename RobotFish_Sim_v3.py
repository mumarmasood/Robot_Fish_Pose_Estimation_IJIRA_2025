# Simulates a 2D fish with a non-linear dynamics model solving the differential equations of motion numerically
# author: Umar Masood
# update: 2025-05-08
# version: 3.0
# This version is updated after first reversion for IJIRA
# Bio-inspired Robotics & Control Lab (BRCL) @ UH

import vpython as vp
import numpy as np
from scipy.integrate import solve_ivp
from fish_v2 import Pool, Fish, Pipeline
from IPython.display import display, HTML
import matplotlib.pyplot as plt

# Create a VPython scene

_canvas_width = 1260
_canvas_height = 720
# make another canvas on the right
scene = vp.canvas(title='Robotic Fish Simulator', width=_canvas_width, height=_canvas_height, background=vp.vector(0.529, 0.807, 0.921))
scene.align = 'left'
scene.select()



camera_offset_plane = None
pipeline_projection_arrow = None

image_width = _canvas_width
image_height = _canvas_height

plot_tracking_vectors = True


def world_to_image_view_point(camera, point):
    # Camera position and axis transformation
    camera_pos = camera.pos
    camera_axis = vp.norm(camera.axis)                          # Forward direction of the camera
    camera_right = vp.norm(vp.cross(camera_axis, camera.up))    # Right vector perpendicular to axis and global z-axis
    camera_up = vp.cross(camera_right, camera_axis)             # Up vector perpendicular to axis and right vector
    camera_right = vp.cross(camera_axis, camera_up)             # Right vector perpendicular to axis and up vector

    # check if camera up and axis are perpendicular to each other
    if vp.dot(camera_up, camera_axis) < -0.1 or vp.dot(camera_right, camera_axis) < -0.1 or vp.dot(camera_right, camera_up) < -0.1:
        print("Camera Axis: ", camera_axis)
        print("Camera Up: ", camera_up)
        print("Camera Right: ", camera_right)
        print("CameraAxis.CameraUp: ", vp.dot(camera_up, camera_axis))
        print("CameraAxis.CameraRight: ", vp.dot(camera_right, camera_axis))
        print("CameraUp.CameraRight: ", vp.dot(camera_right, camera_up))
        raise ValueError("Camera axis and up vector must be perpendicular to each other")
    
    # Transformation matrix (rotation component)
    wRc = np.array([[camera_right.x, camera_up.x, camera_axis.x],
                    [camera_right.y, camera_up.y, camera_axis.y],
                    [camera_right.z, camera_up.z, camera_axis.z]])
    
    wPc = np.array([camera_pos.x, camera_pos.y, camera_pos.z])

    cRw = wRc.T
    cPw = -np.dot(cRw, wPc)

    w2c_matrix = np.array([[cRw[0][0], cRw[0][1], cRw[0][2], cPw[0]],
                            [cRw[1][0], cRw[1][1], cRw[1][2], cPw[1]],
                            [cRw[2][0], cRw[2][1], cRw[2][2], cPw[2]],
                            [0, 0, 0, 1]])
    
    # print("Transformation matrix: \n", w2c_matrix)
    
    # Transform the point to the camera frame
    point_transformed = w2c_matrix @ np.array([point.x, point.y, point.z, 1])
    print("Point before transformation: ", point)

    point = vp.vector(point_transformed[0], point_transformed[1], point_transformed[2])

    print("Point after transformation: ", point)
    # Calculate the field of view angles in the x and y directions
    fov_x = camera.fov
    fov_y = 2*np.arctan(np.tan(fov_x / 2) * (image_height / image_width))  # Adjusted for aspect ratio

    print("FOV: ", np.degrees(fov_x), np.degrees(fov_y))

    # print ("Relative point: ", point)
    
    # Calculate the view vector components in the camera frame
    view_x = vp.dot(point, vp.vector(1, 0, 0))
    view_y = vp.dot(point, vp.vector(0, 1, 0))
    view_z = vp.dot(point, vp.vector(0, 0, 1))

    print("View vector: ", view_x, view_y, view_z)

    # Ignore points behind the camera
    if view_z <= 0:
        print("Point is behind the camera")
        print("Camera Axis: ", camera_axis)
        print("point: ", point)
        return None  # Point is behind the camera; no projection possible

    # Map view_x and view_y to pixel coordinates in the 2D image frame
    fx = image_width / (2 * np.tan(fov_x / 2))  # Focal length in x direction
    fy = image_height / (2 * np.tan(fov_y / 2))  # Focal length in y direction

    print("Focal length: ", fx, fy)

    pixel_x = (fx * view_x / view_z)
    pixel_y = (fy * view_y / view_z) + (image_height / 2)  # Adjusted for image center 

    print("Pixel coordinates: ", pixel_x, pixel_y)

    # image frame to my frame transformation 
    # bring down the origin to the bottom center of the image and horizontal to be x axis and vertical to be y axis
    # pixel_y = image_height - pixel_y
    # pixel_x = pixel_x - image_width / 2

    return vp.vector(pixel_x, pixel_y, 0)

def swtich_view():
    print("Switching view")
    if scene_checkbox.checked:
        hc_arrow.visible = False
        projection_arrow.visible = False
        hd_vect_arrow.visible = False
        
    else:
        scene.camera.pos = vp.vector(pool.length / 2, pool.width / 2, 10)
        # scene.camera.pos = vp.vector(pool.length / 2, -pool.width, 5)
        # scene.center = vp.vector(pool.length / 2, pool.width / 2, 0)
        # scene.camera.axis = vp.vector(0, 2*pool.width, -10)
        scene.camera.axis = vp.vector(0, 0, -5)
        scene.camera.up = vp.vector(0, 1, 0)

def pbvs_swtich():
    global pbvs
    if pbvs_checkbox.checked:
        pbvs = True
        scene_checkbox.checked = False
        swtich_view()
    else:
        pbvs = False
        scene_checkbox.checked = True
        swtich_view()

def track_pipeline_pbvs(pipeline, fish, rc, plot_vectors= True ):
    """
    Track the pipeline using the PBVS algorithm"
    """
    global hc_arrow, projection_arrow, hd_vect_arrow
    hc_vect = rc* vp.vector(np.cos(fish.psi), np.sin(fish.psi), 0)

    surface_pipeline_pos = vp.vector(pipeline.start.x, pipeline.start.y, 0)
    surface_pipeline_axis = vp.norm(vp.vector(pipeline.axis.x, pipeline.axis.y, 0))

    print("Pipeline axis: ", surface_pipeline_axis)
    print("Pipeline position: ", surface_pipeline_pos)

    # Calculate the end point of the `hc` vectoread
    hc_end = vp.vector(fish.x + hc_vect.x, fish.y + hc_vect.y, 0)

    # Project `hc_end` onto the surface pipeline
    to_hc_vector = hc_end - surface_pipeline_pos

    # Calculate the projection of `to_hc_vector` onto the pipeline axis
    projection_length = vp.dot(to_hc_vector, surface_pipeline_axis)  # Scalar projection length
    projection_vector = projection_length * surface_pipeline_axis  # Projection vector along pipeline

    # Calculate the perpendicular (normal) vector from `hc_end` to the pipeline
    n_vect = -to_hc_vector + projection_vector

    hd_vect = hc_vect + n_vect    # Desired vector

    psi_e = np.arctan2(hd_vect.y, hd_vect.x) - fish.psi
    psi_e_deg = np.degrees(psi_e)

    if plot_vectors:

        if hc_arrow is not None:
            hc_arrow.visible = False  # Make the previous arrow invisible if it exists

        # Draw the perpendicular vector as an arrow from `hc_end` to the pipeline
        if projection_arrow is not None:
            projection_arrow.visible = False  # Make the previous arrow invisible if it exists

            # draw hd_vect
        if hd_vect_arrow is not None:
            hd_vect_arrow.visible = False

        if scene_checkbox.checked is False:
            hd_vect_arrow = vp.arrow(pos=fish.body.pos, axis=hd_vect, color=vp.color.green, round=True, shaftwidth=0.02)
            projection_arrow = vp.arrow(pos=hc_end, axis=n_vect, color=vp.color.blue, round=True, shaftwidth=0.02)
            hc_arrow = vp.arrow(pos=vp.vector(fish.x, fish.y, 0), axis=hc_vect, color=vp.color.red, round = True, shaftwidth = 0.02) 

    return psi_e_deg


def limit(value, min_value, max_value):
    """
    Limit the value between min_value and max_value
    """
    if value < min_value:
        return min_value
    elif value > max_value:
        return max_value
    else:
        return value
    
    

pool = Pool(wall_thickness=0.1, length=20, width=5, depth=1, water_level=0.8)
pool.plot(scene, walls_color = vp.color.white, water_color=vp.color.cyan, opacity=1)


# Pipeline parameters
pipeline_diameter_outer = 0.025  # Outer diameter of 8 cm
pipeline_thickness = 0.01       # Pipeline wall thickness of 1 cm

pipe1 = Pipeline(scene=scene, _pool=pool, _start = [5,2.5], _end = [14,2.5], _radius = pipeline_diameter_outer)
pipeline_axis = pipe1.axis
pipeline_start = pipe1.start
pipeline_end = pipe1.end

# # plot x, y and z axis on the absolute frame
# vp.arrow(pos=vp.vector(0, 0, 0), axis=vp.vector(1, 0, 0), color=vp.color.red, shaftwidth=0.05)
# vp.arrow(pos=vp.vector(0, 0, 0), axis=vp.vector(0, 1, 0), color=vp.color.green, shaftwidth=0.05)
# vp.arrow(pos=vp.vector(0, 0, 0), axis=vp.vector(0, 0, 1), color=vp.color.blue, shaftwidth=0.05)



# Initialize the fish
fish = Fish(x=2, y=3.5, psi= -0*np.pi/180, delta=0, alpha1=0, alpha2=0, u=0, v=0, r=0)
fish.set_shape(trail=True)

# print fish position and psi
print("Fish position: ", fish.x, fish.y)
print("Fish psi: ", fish.psi)


# change the view angle of the camera
scene.camera.fov = np.radians(45)  # Set the camera's field of view to 60 degrees
scene.camera.pos = vp.vector(pool.length / 2, pool.width / 2, 5)
scene.camera.axis = vp.vector(0,0, -5)

    

# add user input in the canvas to swtich between camera view and isometric view
g1 = vp.graph(width=600, height=400, title='Fish position and oreintation', xtitle='Time', 
              ytitle='Angle (degrees)/position (m)', fast=False, align='right')
psi_e_curve = vp.gcurve(graph=g1, color=vp.color.red, label='$\psi_e$')
delta_curve = vp.gcurve(graph=g1, color=vp.color.blue, label='$\delta$')
scene_checkbox = vp.checkbox(bind = swtich_view, text = 'Switch to Fish Camera', checked = True)
pbvs_checkbox = vp.checkbox(text='PBVS', bind = pbvs_swtich, pos=scene.caption_anchor)
                           
hc_arrow = None
psi_e_hist = np.zeros(10)

pbvs = False
psi_e = 0
psi_e_deg = 0
P_axis_2d = vp.vector(0, 0, 0)
hd_vect_arrow = None
projection_arrow = None
P_start_2d = vp.vector(0, 0, 0)
hd_start_2d = vp.vector(0, 0, 0)
n_start_2d = vp.vector(0, 0, 0)
n_axis_2d = vp.vector(0, 0, 0)
hc_2d = vp.vector(0, 0, 0)
P_camera = vp.vector(0, 0, 0)
hc_arrow = None
P_vect = vp.vector(0, 0, 0)
hc_vect = vp.vector(0, 0, 0)
hd_vect = vp.vector(0, 0, 0)
n_vect = vp.vector(0, 0, 0)

kp = 3
ki = 0.1
kd = 1

rc = 1
rc_pixel = 200

# Animation loop
t_end = 35
t = 0
dt = 0.05

timestamps = np.arange(0, t_end, dt)
psi_e_store = np.zeros(len(timestamps))


while t < t_end:
    vp.rate(1/dt)

    if scene_checkbox.checked:

        # setup the scene camera to the position of the fish head and camera is mounted on the fish 30 degrees seeing downward
        scene.camera.pos = vp.vector(fish.x + 0.2*np.cos(fish.psi), fish.y + 0.2*np.sin(fish.psi), -0.05)
        fish_camera_x, fish_camera_y, fish_camera_z = scene.camera.pos.x, scene.camera.pos.y, scene.camera.pos.z
        # Set the camera's viewing direction with a 30-degree downward angle
        scene.camera.up = vp.vector(0, 0, 1)
        downward_angle = np.radians(30)  # Convert 30 degrees to radians
        scene.camera.axis = vp.vector(np.cos(fish.psi), np.sin(fish.psi), -np.tan(downward_angle))

        # # plot x, y, and z axis on the camera frame
        # # z axis being camera axis x camera right vector y being camera up vector
        # vp.arrow(pos=vp.vector(fish_camera_x, fish_camera_y, fish_camera_z), 
        #          axis=scene.camera.axis, color=vp.color.blue, shaftwidth=0.05) # z axis
        # vp.arrow(pos=vp.vector(fish_camera_x, fish_camera_y, fish_camera_z), 
        #          axis=vp.cross(scene.camera.axis, scene.camera.up), color=vp.color.red, shaftwidth=0.05) # x axis
        # vp.arrow(pos=vp.vector(fish_camera_x, fish_camera_y, fish_camera_z), 
        #          axis=vp.cross(vp.cross(scene.camera.axis, scene.camera.up), scene.camera.axis), color=vp.color.green, 
        #          shaftwidth=0.05) # y axis

    # plot fish x, y and psi in degrees
    # pipeline_angle_curve.plot(t, psi_e_deg)
    # plot delta
    
    # another plot in pipeline_angle_curve
    # pipeline_angle_curve.plot(t, np.arctan2(P_axis_2d.y, P_axis_2d.x))
    
    # vector in the heading direction of the fish from the fish head
    hc_vect = rc* vp.vector(np.cos(fish.psi), np.sin(fish.psi), 0)

    # Define the pipeline on the surface
    surface_pipeline_pos = vp.vector(pipeline_start.x, pipeline_start.y, 0)
    surface_pipeline_axis = vp.norm(vp.vector(pipeline_axis.x, pipeline_axis.y, 0))  # Normalized direction of the pipeline


    # pipeline vector from start to end of the pipeline denoted as P
    P_vect = vp.vector(pipeline_end.x - pipeline_start.x, pipeline_end.y - pipeline_start.y, 0)

    # Calculate the end point of the `hc` vectoread
    hc_end = vp.vector(fish.x + hc_vect.x, fish.y + hc_vect.y, 0)

    # Project `hc_end` onto the surface pipeline
    to_hc_vector = hc_end - surface_pipeline_pos

    # Calculate the projection of `to_hc_vector` onto the pipeline axis
    projection_length = vp.dot(to_hc_vector, surface_pipeline_axis)  # Scalar projection length
    projection_vector = projection_length * surface_pipeline_axis  # Projection vector along pipeline

    # Calculate the perpendicular (normal) vector from `hc_end` to the pipeline
    n_vect = -to_hc_vector + projection_vector

    hd_vect = hc_vect + n_vect    

    
    if pbvs is False:
        # angle measure from hc to hd_vect signed angle
        # psi_e = vp.diff_angle(hd_axis_2d, hc_2d)
        pipeline_start_pixel = world_to_image_view_point(scene.camera, pipeline_end - pipeline_axis * 0.2)
        # print("Pipeline start: ", pipeline_start)
        # print("Pipeline start pixel: ", pipeline_start_pixel)
        pipeline_end_pixel = world_to_image_view_point(scene.camera, pipeline_end)
        # print("Pipeline end: ", pipeline_end)
        # print("Pipeline end pixel: ", pipeline_end_pixel)


        pipeline_angle_image = np.arctan2(pipeline_end_pixel.x - pipeline_start_pixel.x, pipeline_end_pixel.y - pipeline_start_pixel.y)

        print("P_2d angle: ", np.degrees(pipeline_angle_image))

        hc_2d = vp.vector(0, rc_pixel, 0)
        P_start_2d = pipeline_end_pixel
        P_axis_2d = pipeline_end_pixel - pipeline_start_pixel

        print("P_axis_2d: ", P_axis_2d)
        print("P_start_2d: ", P_start_2d)
        print("hc_2d: ", hc_2d)


        # Drop normal vector from hc_2d to P_axis_2d
        hd_2d = vp.proj(hc_2d, P_axis_2d) + P_start_2d
        n_axis_2d = hd_2d - hc_2d

        # Calculate the signed angle from the `hc` vector to the `hd` vector using numpy functions
        psi_e = np.arctan2(hd_2d.x * hc_2d.y - hd_2d.y * hc_2d.x,
                   hd_2d.x * hc_2d.x + hd_2d.y * hc_2d.y)


        if psi_e > np.pi/2:
            psi_e = psi_e - np.pi
        if psi_e < -np.pi/2:
            psi_e = psi_e + np.pi
        psi_e_deg = -np.degrees(psi_e)

        psi_e_deg = (psi_e_deg + np.sum(psi_e_hist)) / (len(psi_e_hist) + 1)

        delta = kp * psi_e_deg + ki * np.sum(psi_e_hist) + kd * (psi_e_hist[0] - psi_e_hist[1])
        delta = limit(delta, -20, 20)  # Limit delta to be between -30 and 30 degrees

        fish.move(omega = 11, _del= -np.radians(delta), t=t, dt=dt)

        # if camera_offset_plane is not None:
        #     camera_offset_plane.visible = False
        # camera_offset_plane = vp.box(pos = scene.camera.pos + 0.5 * scene.camera.axis, size = vp.vector(0.001, 1, 1), color = vp.color.red,
        #                              axis = scene.camera.axis, opacity = 0.5)

    if pbvs is True:
        # angle measure from hc to hd_vect signed angle
        psi_e = np.arctan2(hd_vect.y, hd_vect.x) - fish.psi
        psi_e_deg = np.degrees(psi_e)
        # print("Angle between hc and p_vect: ", psi_e_deg)
        # psi_e_deg = (psi_e_deg + np.sum(psi_e_hist)) / (len(psi_e_hist) + 1)
        delta = kp * psi_e_hist[0] + ki * np.sum(psi_e_hist) + kd * (psi_e_hist[0] - psi_e_hist[1])
        delta = limit(delta, -20, 20)  # Limit delta to be between -30 and 30 degrees

        fish.move(omega=11, _del= -np.radians(delta), t=t, dt=dt)       

    if plot_tracking_vectors is True:

        if hc_arrow is not None:
            hc_arrow.visible = False  # Make the previous arrow invisible if it exists

        # Draw the perpendicular vector as an arrow from `hc_end` to the pipeline
        if projection_arrow is not None:
            projection_arrow.visible = False  # Make the previous arrow invisible if it exists

            # draw hd_vect
        if hd_vect_arrow is not None:
            hd_vect_arrow.visible = False

        if scene_checkbox.checked is False:
            hd_vect_arrow = vp.arrow(pos=fish.body.pos, axis=hd_vect, color=vp.color.green, round=True, shaftwidth=0.02)
            projection_arrow = vp.arrow(pos=hc_end, axis=n_vect, color=vp.color.blue, round=True, shaftwidth=0.02)
            hc_arrow = vp.arrow(pos=vp.vector(fish.x, fish.y, 0), axis=hc_vect, color=vp.color.red, round = True, shaftwidth = 0.02) 

    # store the last 10 psi_e values to apply pid control

    psi_e_hist[1:] = psi_e_hist[:-1]
    psi_e_hist[0] = psi_e_deg
    psi_e_store[int(t/dt)] = psi_e_deg
    print("Psi_e_deg: ", psi_e_deg)
    print("Delta: ", fish.delta * 180 / np.pi)

    delta_curve.plot(t, fish.delta * 180 / np.pi)
    psi_e_curve.plot(t, psi_e_deg)
    
    # i = i + 1
    t += dt

# uhist cut to length of time
fish.u_hist = fish.u_hist[:len(timestamps)]
delta_hist = np.array(fish.delta_hist[:len(timestamps)]) * 180 / np.pi


# Save the simulation data into csv file
import csv
path = 'Sim_Data/v3/'

export_file_name = 'Sim_' + str(pbvs) + '_kp' + str(kp) + '_ki' + str(ki) + '_kd' + str(kd) + '_rc' + str(rc) + '_' + str(rc_pixel) + '.csv'

# check if file doesn't exist create one
import os
if not os.path.exists(path):
    os.makedirs(path)

with open(path + export_file_name, mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['Time', 'X', 'Y', 'Psi', 'Delta', 'U', 'V', 'R'])
    for i in range(len(timestamps)):
        writer.writerow([timestamps[i], fish.x_hist[i], fish.y_hist[i], psi_e_store[i], delta_hist[i], fish.u_hist[i], fish.v_hist[i], fish.r_hist[i]])



# Apply avg filter to delta_hist
delta_hist_filt = np.convolve(delta_hist, np.ones(15)/15, mode='valid')
timestamps_filt = timestamps[:-14]
psi_e_store_filt = np.convolve(psi_e_store, np.ones(15)/15, mode='valid')

filt_export_file_name = 'Sim_' + str(pbvs) + '_kp' + str(kp) + '_ki' + str(ki) + '_kd' + str(kd) + '_rc' + str(rc) + '_' + str(rc_pixel) + 'filt.csv'

with open(path + filt_export_file_name, mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['Time', 'X', 'Y', 'Psi', 'Delta', 'U', 'V', 'R'])
    for i in range(len(timestamps_filt)):
        writer.writerow([timestamps_filt[i], fish.x_hist[i], fish.y_hist[i], psi_e_store_filt[i], delta_hist_filt[i], fish.u_hist[i], fish.v_hist[i], fish.r_hist[i]])


# Set font and rendering options
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',   # Computer Modern
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

# Plot Fish speeds over time
plt.figure()
plt.plot(timestamps, delta_hist, label='delta')
plt.plot(timestamps, psi_e_store, label='psi_e')
plt.plot(timestamps_filt, delta_hist_filt, label=r'$\delta$')
plt.plot(timestamps_filt, psi_e_store_filt, label=r'$\psi_{e}$')
plt.grid()
plt.title('Fish Control Signals')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.show()


# if pbvs is True:
#     plt.savefig(path + 'Sim_PBVS' + str(pbvs) + '_kp' + str(kp) + '_ki' + str(ki) + '_kd' + str(kd) + '_rc' + str(rc) + '.png')
# else:
#     plt.savefig(path + 'Sim_IBVS' + str(pbvs) + '_kp' + str(kp) + '_ki' + str(ki) + '_kd' + str(kd) + '_rc' + str(rc_pixel) + '.png')

# plt.show()