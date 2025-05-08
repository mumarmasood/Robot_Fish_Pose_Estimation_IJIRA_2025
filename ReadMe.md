# Robotic Fish Tracking and Testing System

This system is designed for real-time tracking and monitoring of a robotic fish's movement and orientation. It features a user interface for initiating and controlling the tracking process, visualizing data, and handling mission control.

## Features

- Real-time video streaming for monitoring robotic fish.
- Manual control adjustment for `tu`, `tail_speed`, and `tail_angle`.
- Tracking of the robotic fish's X, Y coordinates and orientation angle.
- Visualization of the fish's trajectory in a 2D plot.
- Mission control with start and stop functionality.
- Data collection and export to CSV format for analysis.



## Usage

This application is designed to run on Windows environments. Follow the steps below to set up and start the system:

1. **Install Python**: Ensure Python 3.x is installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

2. **Install Dependencies**: Open the Command Prompt (cmd) and navigate to the project's directory. Install the required Python packages by running:

    ```bash
    pip install opencv-python numpy matplotlib pillow requests
    ```

3. **Start the Application**: In the Command Prompt, navigate to the directory containing the script (`robotic_fish_tracking.py`). Run the script with the following command:

    ```bash
    python robotic_fish_tracking.py
    ```

    This will launch the graphical user interface for the robotic fish tracking system.

4. **Configure and Control**: Use the GUI to enter the robotic fish's IP address, adjust control parameters (thrust, tail speed, and tail angle), and manage the tracking and data collection.

For any issues during installation or execution, please refer to the Troubleshooting section or submit an issue on GitHub.


## Code Structure

The robotic fish tracking and testing system is structured as follows to ensure modularity, ease of maintenance, and scalability:

- `robotic_fish_tracking.py`: The main script that initializes the application, sets up the GUI, and starts the video stream and tracking processes.
  
- **GUI Components**:
  - `control_root_frame`: Contains control elements for adjusting the robotic fish's motion parameters and initiating tests.
  - `video_root_frame`: Displays the live video feed from the robotic fish's environment.
  - `plot_frame`: Visualizes the robotic fish's trajectory and other relevant data in real-time.

- **Networking**:
  - `send_data()`: Handles the communication with the robotic fish by sending control commands through HTTP requests.

- **Computer Vision**:
  - `open_cv_window()`: Opens a new window for selecting the robotic fish before tracking.
  - `click_event()`: Processes mouse events for selecting the region of interest in the video feed.
  - `video_stream()`: Continuously captures frames from the video feed, applies the tracking algorithm, and updates the GUI.

- **Data Handling**:
  - `update_plot()`: Updates the data visualization based on the tracking information.
  - `clear_plot()`: Clears the current data visualization and resets data collection.
  - `save_data_to_csv()`: Saves the collected tracking data to a CSV file for further analysis.

- **Utility Functions**:
  - Parameter adjustment functions such as `increase_tu()`, `decrease_tu()`, `increase_speed()`, `decrease_speed()`, `increase_angle()`, and `decrease_angle()` allow for dynamic control of the robotic fish's movement parameters.
  - `on_closing()`: Ensures proper shutdown of the application, including releasing hardware resources and saving data.


### Directory Structure

The project is organized into the following directory structure for clarity and ease of access:
Robot_Fish_Tracking/
│//
├── robotic_fish_tracking.py - Main application script.
│//
├── README.md                 - Project documentation and setup instructions.
│//
└── requirements.txt          - List of Python package dependencies.


This structure is designed to keep the application straightforward and focused, with potential for further expansion as needed (e.g., adding modules for different tracking algorithms or supporting additional robotic models).

>>>>>>> 420a2aeca3c201790db2a87426bfbbf558db2615
## Expanding the Codebase

To add new features or modify existing functionality:
- Introduce new modules in separate files to maintain modularity.
- Update the GUI layout in `robotic_fish_tracking.py` as needed to incorporate additional controls or display components.
- Ensure new dependencies are added to `requirements.txt` for consistent setup across environments.

## Controls

- **IP Address Entry**: Enter the IP address of the robotic fish.
- **TU Controls**: Adjust the thrust value with '+' and '-' buttons.
- **Tail Speed Controls**: Adjust the tail speed for movement.
- **Tail Angle Controls**: Adjust the tail angle for steering.
- **Start/Stop Test**: Begin or end the testing sequence.
- **Clear Plot & Data**: Clear the current plot and data records.

## Data Collection

The tracking data is saved in CSV format with the following columns:

- Timestamp (relative to the start of the test)
- X Position (meters)
- Y Position (meters)
- Orientation (degrees)

## Contributing

We highly value contributions from the members of the Bio-Inspired Robotics and Control Lab. If you are outside of our lab and interested in contributing, collaboration, or using our project, please first contact us for permission.

Before contributing, please ensure you follow the guidelines outlined in `CONTRIBUTING.md`. These guidelines will provide you with the necessary steps to make your contributions as seamless as possible and ensure that our project maintains its integrity and standards.

For lab members and those who have been granted permission:

- Fork the repository.
- Create your feature branch (`git checkout -b feature/AmazingFeature`).
- Commit your changes (`git commit -am 'Add some AmazingFeature'`).
- Push to the branch (`git push origin feature/AmazingFeature`).
- Open a Pull Request.

For detailed instructions and our code of conduct, refer to `CONTRIBUTING.md`.

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.

