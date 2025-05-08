import tkinter as tk
from tkinter import ttk
import csv
import requests
import time
from datetime import datetime
import cv2
import threading
from threading import Thread, Event
from PIL import Image, ImageTk
import queue
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from tkinter import messagebox
import re


# Define increment step for each parameter
INCREMENT_STEP = 0.005
SPEED_STEP = 10
ANGLE_STEP = 3

# Define initial values
tu = -0.02
tail_speed = 0
tail_angle = 0

fish_x_meters = None
fish_y_meters = None
fish_theta = None

# Define constraints for each parameter
TU_MIN = -0.02
TU_MAX = 0.02
SPEED_MIN = -100
SPEED_MAX = 100
ANGLE_MIN = -45
ANGLE_MAX = 45

TIMEOUT_DURATION = 2  # seconds
RETRY_DELAY = 1  # seconds

# Constants
RETRY_INTERVAL = 1000  # 1 second in milliseconds
FETCH_INTERVAL = 2000  # 1 second in milliseconds

CAMERASOURCE = 1  # 0 for built-in webcam, 1 for external webcam
RESIZE_SCALE = 0.5  # SCALE FACTOR FOR DISPLAYING VIDEO FEED



fish_attributes = None
initBB = None
dragging = False
roi_start = (0, 0)
roi_end = (0, 0)
mission_start_flag = False
start_time = time.time()
video_writer_handle = None



#Posiiton and orientation data
tracking_data = []  # Create an empty list outside of the loop

FRAME_WIDTH = 1280  # in pixels
FRAME_HEIGHT = 720  # in pixels
POOL_WIDTH = 10  # in meters
POOL_HEIGHT = 5  # in meters

CO2_DATA = 600  # in ppm

# Function to send data to ESP32
def send_data():
    data = f"{tu},{tail_speed},{tail_angle}"
    ip_address = ip_var.get()
    try:
        requests.post(f"http://{ip_address}/command", data={"data": data})
        tu_label_var.set(f"tu: {tu}")
        speed_label_var.set(f"Tail Speed: {tail_speed}")
        angle_label_var.set(f"Tail Angle: {tail_angle}")
    except:
        print("Failed to send data. Check if IP is correct.")

##
        
# Define a flag to track the state of the test
is_test_running = False

# Function to toggle the test
def toggle_test():
    global is_test_running  # Access the global variable is_test_running
    if is_test_running:  # If the test is currently running
        stop_test()  # Call the function to stop the test
        test_button.config(text="Start Test")  # Change the text of the test button to "Start Test"
    else:  # If the test is not currently running
        start_test()  # Call the function to start the test
        test_button.config(text="Stop Test")  # Change the text of the test button to "Stop Test"
    is_test_running = not is_test_running  # Toggle the state of is_test_running

# Function to start the test
def start_test():
    global start_time, start_date_time, end_time, mission_start_flag, test_tail_angle, test_tail_speed, video_writer_handle, tail_angle, tail_speed, tu
    
    tail_speed = tail_speed_var.get()
    tu = depth_rate_var.get()
    tail_angle = tail_angle_var.get()

    test_tail_angle = tail_angle
    test_tail_speed = tail_speed

    
    mission_start_flag = True
    start_time = time.time()
    start_date_time = datetime.now()
    clear_plot()
    send_data()
    video_writer_handle = None
    # initilize saving video if the checkbox is checked
    if save_video_var.get():
    
        date_time_str = start_date_time.strftime("%Y%m%d_%H%M%S")
        # Combine with base name
        filename = f"{save_video_path_var.get()}/Fish_Video_{date_time_str}.avi"
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer_handle = cv2.VideoWriter(filename, fourcc, 20.0, (FRAME_WIDTH, FRAME_HEIGHT))
        print("Video will be saved to:", filename)


# Function to stop the test
def stop_test():
    # Global variables that will be modified in this function
    global start_time, end_time, tracking_data, mission_start_flag, video_writer_handle, tail_angle, tail_speed, tu
    end_time = time.time()  # Record the end time of the test
    tail_speed = 0  # Reset tail speed
    tu = 0  # Reset tu
    tail_angle = 0  # Reset tail angle

    send_data()  # Send the data
    mission_start_flag = False  # Indicate that the mission has ended
    save_data_to_csv("2D_Pose_Data", tracking_data)  # Save the tracking data to a CSV file

    if video_writer_handle is not None:
        video_writer_handle.release()  # Release the video writer handle
        video_writer_handle = None  # Reset the video writer handle



fish_color = None
selected_fish_position = None

def open_cv_window():
    global selected_fish_position, cap
    selected_fish_position = False
    cv2.namedWindow('Tracking')
    cv2.setMouseCallback('Tracking', click_event)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        if dragging:
            cv2.rectangle(frame, roi_start, roi_end, (255, 0, 0), 2)
        cv2.imshow('Tracking', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or selected_fish_position:
            break

    cv2.destroyAllWindows()

def click_event(event, x, y, flags, params):
    global fish_attributes, roi_start, roi_end, dragging, selected_fish_position,tracker
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        roi_start = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        roi_end = (x, y)
        selected_fish_position = True
        bbox = (roi_start[0], roi_start[1], roi_end[0] - roi_start[0], roi_end[1] - roi_start[1])
        tracker = cv2.TrackerMIL.create()
        tracker.init(frame, bbox)
        fish_attributes = {'position': bbox}
        print("Fish selected:", bbox)
        cv2.destroyAllWindows()

    elif dragging and event == cv2.EVENT_MOUSEMOVE:
        roi_end = (x, y)



def video_stream():
    global frame, cap, fish_attributes, dragging, tracker, tracking_data, ax, canvas, start_time, mission_start_flag, fish_x_meters, fish_y_meters, fish_theta, video_writer_handle

    if cap is None:
        cap = _init_camera(CAMERASOURCE)
        # Set the camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)


    ret, frame = cap.read()



    X_RATIO = POOL_WIDTH / FRAME_WIDTH
    Y_RATIO = POOL_HEIGHT / FRAME_HEIGHT
    
    if dragging:
        cv2.rectangle(frame, roi_start, roi_end, (255, 0, 0), 2)

    if fish_attributes:
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cropped_frame = frame[y+2:y+h, x+2:x+w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # Calculate orientation using Canny and Hough
            
            # Convert to grayscale
            gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Perform edge detection
            edges = cv2.Canny(blurred, 50, 100, apertureSize=3)

            # Perform line detection
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=5, maxLineGap=20)
            if lines is not None:
                for line in lines[0]:
                    x1, y1, x2, y2 = line
                    cv2.line(cropped_frame, (x1, y1), (x2, y2), (0, 0, 150), 2)
                    fish_theta = - np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

                    # Calculate the middle point of the rectangle
                    fish_x = x + w/2
                    fish_y = y + h/2

                    fish_x_meters = fish_x * X_RATIO
                    fish_y_meters = (FRAME_HEIGHT - fish_y) * Y_RATIO  # Subtracting from FRAME_HEIGHT to set origin at bottom left                    
                    # print("Orientation: {:.2f} degrees".format(theta))
                    if mission_start_flag:
                        tracking_data.append((time.time()-start_time, fish_x_meters, fish_y_meters, fish_theta))
                        update_plot(tracking_data, ax, canvas)
            else:
                print("No line found for orientation")
            
            # show cropped frame in gray scale in the image panel small
            img = Image.fromarray(blurred)
            imgtk = ImageTk.PhotoImage(image=img)
            image_panel_small.imgtk = imgtk
            image_panel_small.config(image=imgtk)

        else:
            messagebox.showerror("Error","Lost track of the fish, can you identify again?")
            open_cv_window()

    # Update the display
    

    # saved video if the checkbox is checked and mission is running
    if video_writer_handle is not None:
        video_writer_handle.write(frame)

    # Update the display
    scaled_frame = cv2.resize(frame, None, fx=RESIZE_SCALE, fy=RESIZE_SCALE)
    cv2image = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    image_panel.imgtk = imgtk  # Reference to avoid garbage collection
    image_panel.config(image=imgtk)
    root.after(10, video_stream) 

def _init_camera(source):
    
    while True:
        try:
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                raise ValueError("Unable to open video source or could not read frame.")
            break
        except Exception as e:
            result = messagebox.askretrycancel("Error", str(e))
            if result:
                continue
            else:
                exit(0) # Schedule the function to run again after 10ms
    return cap

cap = None


#Plot co2 data
def update_plot(data_points, ax, canvas):
    ax.cla()  # Clear the previous plot
    # # ax.autoscale(False)  # Disable automatic scaling
    ax.set_xlim([0, POOL_WIDTH])
    ax.set_ylim([0, POOL_HEIGHT])
    if data_points:
        time, xs, ys, zs = zip(*data_points)
        ax.scatter(xs, ys, c='r', marker='o')
    canvas.draw()



# Functions to update values
def increase_tu():
    global tu
    if tu < TU_MAX:
        tu += INCREMENT_STEP
        tu = round(tu,3)
        send_data()

def decrease_tu():
    global tu
    if tu > TU_MIN:
        tu -= INCREMENT_STEP
        tu = round(tu,3)
        send_data()

def increase_speed():
    global tail_speed
    if tail_speed < SPEED_MAX:
        tail_speed += SPEED_STEP
        send_data()

def decrease_speed():
    global tail_speed
    if tail_speed > SPEED_MIN:
        tail_speed -= SPEED_STEP
        send_data()

def increase_angle():
    global tail_angle
    if tail_angle < ANGLE_MAX:
        tail_angle += ANGLE_STEP
        send_data()

def decrease_angle():
    global tail_angle
    if tail_angle > ANGLE_MIN:
        tail_angle -= ANGLE_STEP
        send_data()


def clear_plot():
    # Show a confirmation messagebox
    response = messagebox.askyesno("Confirmation", "Are you sure you want to clear the plot and delete the data?")
    if response:
        # Clear the data and the plot
        tracking_data.clear()
        ax.cla()
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_xlim([0, POOL_WIDTH])
        ax.set_ylim([0, POOL_HEIGHT])
        ax.set_aspect('equal', adjustable='box')  # Set the aspect ratio based on the pool dimensions
        ax.autoscale(False)  # Disable automatic scaling
        canvas.draw()

def save_data_to_csv(base_name, _tracking_data):
    global test_tail_angle, test_tail_speed, start_date_time

    date_time_str = start_date_time.strftime("%Y%m%d_%H%M%S")
    # Combine with base name
    filename =  f"Data/{base_name}_{date_time_str}_angle_{test_tail_angle}_speed_{test_tail_speed}.csv"
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "x", "y","theta"])
        writer.writerows(_tracking_data)

def update_position_label():
    if fish_x_meters is not None and fish_y_meters is not None and fish_theta is not None:
        position_label_var.set(f"Fish Position:\n X={fish_x_meters:.1f},\n Y={fish_y_meters:.1f},\n Theta={fish_theta:.1f}")
    else:
        position_label_var.set("Fish Position: tracking...")

    # Schedule the function to run again after 100ms
    root.after(100, update_position_label)



def on_closing():
    global tu, tail_speed, tail_angle, should_run 
    tu = 0.02
    tail_speed = 0
    tail_angle = 0
    should_run = False
    # send_data()
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

# Create tkinter window
root = tk.Tk()
root.title("Robot Fish Response Data Collection")

# Assuming a fixed window size for simplicity; you may adjust as needed
window_width = 1920
window_height = 1080
root.geometry(f"{window_width}x{window_height}")

# Configure the root window's grid
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=2)
root.grid_columnconfigure(2, weight=1)
root.grid_rowconfigure(0, weight=1)

# Main Frame for Left Side Controls
control_root_frame = ttk.Frame(root, padding="10")
control_root_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

# Center Frame for video feed
video_root_frame = ttk.Frame(root, padding="10")
video_root_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

# Right Side Frame for Plots
plot_frame = ttk.Frame(root, padding="10")
plot_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)

# IP Address Frame (Nested within the control_root_frame)
ip_frame = ttk.Frame(control_root_frame, padding="10")
ip_frame.pack(pady=5)
ttk.Label(ip_frame, text="Robot Fish IP Address:").pack(pady=5)
ip_var = tk.StringVar(value="192.168.30.2")
ip_entry = ttk.Entry(ip_frame, textvariable=ip_var)
ip_entry.pack(pady=5)

# Controls Frame (Nested within the control_root_frame)
controls_frame = ttk.Frame(control_root_frame, padding="10")
controls_frame.pack(pady = 5)
tu_label_var = tk.StringVar(value=f"tu: {tu}")
ttk.Label(controls_frame, textvariable=tu_label_var).grid(row=0, column=0)
ttk.Button(controls_frame, text="+", command=increase_tu).grid(row=0, column=1)
ttk.Button(controls_frame, text="-", command=decrease_tu).grid(row=0, column=2)

speed_label_var = tk.StringVar(value=f"Tail Speed: {tail_speed}")
ttk.Label(controls_frame, textvariable=speed_label_var).grid(row=1, column=0)
ttk.Button(controls_frame, text="+", command=increase_speed).grid(row=1, column=1)
ttk.Button(controls_frame, text="-", command=decrease_speed).grid(row=1, column=2)

angle_label_var = tk.StringVar(value=f"Tail Angle: {tail_angle}")
ttk.Label(controls_frame, textvariable=angle_label_var).grid(row=2, column=0)
ttk.Button(controls_frame, text="+", command=increase_angle).grid(row=2, column=1)
ttk.Button(controls_frame, text="-", command=decrease_angle).grid(row=2, column=2)


error_label_var = tk.StringVar()
error_label = ttk.Label(control_root_frame, textvariable=error_label_var)
error_label.pack(pady=5)

# Mission Frame
mission_frame = ttk.Frame(control_root_frame, padding="10")
mission_frame.pack(pady=5)

# Create the button
test_button = ttk.Button(mission_frame, text="Start Test", command=toggle_test)
test_button.pack(fill=tk.X, pady=5)

# Label to display current mode
status_var = tk.StringVar(value="Manual Mode")
status_label = ttk.Label(mission_frame, textvariable=status_var)
status_label.pack(padx=10, pady=10)

frame = ttk.Frame(mission_frame, padding="10")
frame.pack(padx=10, pady=10)

# Tail Speed Entry
ttk.Label(frame, text="Tail Speed:").grid(row=1, column=0, sticky=tk.W, pady=2)
tail_speed_var = tk.DoubleVar(value=50)
tail_speed_entry = ttk.Entry(frame, textvariable=tail_speed_var, width=15)
tail_speed_entry.grid(row=1, column=1, sticky=tk.W, pady=2)

# Depth Rate Entry
ttk.Label(frame, text="Depth Rate:").grid(row=2, column=0, sticky=tk.W, pady=2)
depth_rate_var = tk.DoubleVar(value=0.01)
depth_rate_entry = ttk.Entry(frame, textvariable=depth_rate_var, width=15)
depth_rate_entry.grid(row=2, column=1, sticky=tk.W, pady=2)

# Tail Angle Entry
ttk.Label(frame, text="Tail Angle:").grid(row=3, column=0, sticky=tk.W, pady=2)
tail_angle_var = tk.DoubleVar(value=30)
tail_angle_entry = ttk.Entry(frame, textvariable=tail_angle_var, width=15)
tail_angle_entry.grid(row=3, column=1, sticky=tk.W, pady=2)

# label to display current position and orientation of the fish
position_label_var = tk.StringVar(value="Fish Position: ")
if fish_x_meters is not None and fish_y_meters is not None and fish_theta is not None:
    position_label_var.set(f"Fish Position:\n X={fish_x_meters:.1f},\n Y={fish_y_meters:.1f},\n Theta={fish_theta:.1f}")
else:
    position_label_var.set("Fish Position: tracking...")
position_label = ttk.Label(control_root_frame, textvariable=position_label_var)
position_label.pack(pady=5)


# Video Panel
video_frame = ttk.Frame(video_root_frame)
video_frame.pack(pady=10)

image_panel = ttk.Label(video_root_frame)
image_panel.pack(pady=10)



fig = Figure()
ax = fig.add_subplot(111)
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
# ax.set_zlabel('CO2 Concentration (ppm)')

ax.set_xlim([0, POOL_WIDTH])
ax.set_ylim([0, POOL_HEIGHT])
ax.autoscale(False)  # Disable automatic scaling
canvas = FigureCanvasTkAgg(fig, master=plot_frame)  # Assuming 'root' is your Tkinter root window
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(pady=10)

# make a small image panel down to the plot and above to the clear button
image_panel_small = ttk.Label(plot_frame)
image_panel_small.pack(pady=10)


open_cv_button = ttk.Button(video_root_frame, text="Select Fish", command=open_cv_window)
open_cv_button.pack(side=tk.LEFT, padx=10, pady=10)

# Save video checkbox
save_video_var = tk.BooleanVar()
save_video_checkbox = ttk.Checkbutton(video_root_frame, text="Save Video", variable=save_video_var, onvalue=True, offvalue=False)
save_video_checkbox.pack(side=tk.LEFT, padx=10, pady=10)

# path to save the video
save_video_path_var = tk.StringVar(value=r"C:\Users\mumar\OneDrive - University Of Houston\UoH\Dr Chen\Robot Fish\Program\Python\Robot_Fish_Pose_Estimation")
save_video_path_entry = ttk.Entry(video_root_frame, textvariable=save_video_path_var, width=30)
save_video_path_entry.pack(side=tk.LEFT, padx=10, pady=10)

# Browse button to select the path
def browse_path():
    path = tk.filedialog.askdirectory()
    save_video_path_var.set(path)

browse_button = ttk.Button(video_root_frame, text="Browse", command=browse_path)
browse_button.pack(side=tk.LEFT, padx=10, pady=10)


clear_plot_button = ttk.Button(plot_frame, text="Clear Plot & Data", command=clear_plot)
clear_plot_button.pack(pady=10)





if __name__ == "__main__":
    root.after(10, update_position_label)
    # stop_thread = fetch_co2_data(ip_var, root, set_error_label_empty, set_error_label_error)
    # cv2.namedWindow('Tracking')
    # cv2.setMouseCallback('Tracking', click_event)
    # root.after(100, check_queue)
    root.after(10, video_stream)
    # send_data()  # Assuming send_data doesn't need to be scheduled periodically
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
    cv2.destroyAllWindows()
    # stop_thread.set()