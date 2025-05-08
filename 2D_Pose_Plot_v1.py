# Program to plot speed and orientation data from a 2D pose file in csv format

import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

# define the file name and path
file_name = '2D_Pose_Data_20240319_155306_angle_-20.0_speed_90.0.csv'
file_path = 'Data/' + file_name
# TRY THESE FILES
#'Data/2D_Pose_Data_20240319_155306_angle_-20.0_speed_90.0.csv'
#'Data/2D_Pose_Data_20240319_155416_angle_-20.0_speed_100.0.csv'


# Open the file which has four columns: time, x, y, theta skip the 
# first row which is the header

time = []
x = []
y = []
theta = []

with open(file_path, 'r') as file:
    reader = csv.reader(file)
    next(reader) # skip the header
    for row in reader:
        time.append(float(row[0]))
        x.append(float(row[1]))
        y.append(float(row[2]))
        theta.append(float(row[3]))

# smoothen the x an y data
# median filter
x_med_filt = scipy.signal.medfilt(x, kernel_size=31)
y_med_filt = scipy.signal.medfilt(y, kernel_size=31)
# average filter
x_med_avg_filt = np.convolve(x_med_filt, np.ones(30)/30, mode='valid')
y_med_avg_filt = np.convolve(y_med_filt, np.ones(30)/30, mode='valid')


# # Plot the x and y position
# plt.figure(1)
# plt.plot(x, y, color='blue', linewidth=2)
# # change the font style to jsmath-cmsy10
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.xlabel('X position (m)')
# plt.ylabel('Y position (m)')
# plt.title('2D Pose')
# plt.grid(True)
# plt.show()

# Plot the orientation
plt.figure(2)
plt.plot(time, theta, 'b')
plt.xlabel('Time (s)')
plt.ylabel('Orientation (rad)')
plt.title('Orientation')
plt.grid(True)
plt.show()

# Plot the speed (magnitude of the velocity) as a function of time

# Calculate the speed

speed = []
for i in range(1, len(time)):
    dx = x[i] - x[i-1]
    dy = y[i] - y[i-1]
    dt = time[i] - time[i-1]
    speed.append(np.sqrt(dx**2 + dy**2)/dt)

# median and average filter the speed data to smoothen it,


# median filter
speed_med_filt = scipy.signal.medfilt(speed, kernel_size=21)


# Plot the speed
plt.figure(3)
plt.plot(time[0:len(speed)], speed, 'g')
plt.plot(time[0:len(speed_med_filt)], speed_med_filt, 'm')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.title('Speed')
plt.grid(True)
plt.show()

# Plot the heading angle as a function of time

# Calculate the heading angle (degrees) based on x y position
heading = []
for i in range(1, len(time)):
    dx = x[i] - x[i-1]
    dy = y[i] - y[i-1]
    angle_deg = np.degrees(np.arctan2(dy, dx))
    heading.append(angle_deg)

# Smooth out the heading signal
heading_smooth = scipy.signal.medfilt(heading, kernel_size=21)

# Plot the heading angle
plt.figure(4)
plt.plot(time[1:], heading_smooth, 'm')
plt.xlabel('Time (s)')
plt.ylabel('Heading (degrees)')
plt.title('Heading')
plt.grid(True)
plt.show()



def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

