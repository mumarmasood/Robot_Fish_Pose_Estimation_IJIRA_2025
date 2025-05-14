# testing the transformation from world coordinates to image coordinates
import numpy as np
import vpython as vp
import math
import matplotlib.pyplot as plt

# Set the image dimensions
image_width = 640
image_height = 480


scene = vp.canvas(title='Camera Projection', width=image_width, height=image_height, background=vp.vector(0.5, 0.5, 0.5))





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
    fov_y = camera.fov * (image_height / image_width)

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
    pixel_x = (view_x / (np.tan(fov_x / 2) * view_z))
    pixel_y = (view_y / (np.tan(fov_y / 2) * view_z))

    print("Pixel coordinates: ", pixel_x, pixel_y)

    # image frame to my frame transformation 
    # bring down the origin to the bottom center of the image and horizontal to be x axis and vertical to be y axis
    # pixel_y = image_height - pixel_y
    # pixel_x = pixel_x - image_width / 2

    return vp.vector(pixel_x, pixel_y, 0)




# make a box
box = vp.box(pos=vp.vector(0, 0, 0), size=vp.vector(1, 1, 1), color=vp.color.red)
# Create a camera object
scene.camera.pos = vp.vector(0, 0, 5)  # Camera position
scene.camera.up = vp.vector(0, 1, 0)  # Camera up vector (global y-axis)
scene.camera.axis = vp.vector(np.tan(np.radians(30)), np.tan(np.radians(30)), -1)  # Camera direction (looking down the negative z-axis)
scene.camera.fov = math.radians(60)  # Field of view in radians
# Define a point in world coordinates
point = vp.vector(1, 1, 1)
# make a sphere at the point
sphere = vp.sphere(pos=point, radius=0.05, color=vp.color.green)


t_end = 3  # seconds
t_start = 0  # seconds
t = 0  # seconds
dt = 0.01  # seconds

while t < t_end:
    
    vp.rate(100)  # Limit the loop to 100 iterations per second

    

    

    # Project the point onto the image plane
    pixel_coordinates = world_to_image_view_point(scene.camera, point)
    print("Pixel coordinates: ", pixel_coordinates)

    t += dt  # Increment time