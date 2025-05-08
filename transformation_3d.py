# Transformation matrix (rotation component)
w2c_matrix = np.array([
            [camera_right.x, camera_up.x, camera_axis.x, camera_pos.x],
            [camera_right.y, camera_up.y, camera_axis.y, camera_pos.y],
            [camera_right.z, camera_up.z, camera_axis.z, camera_pos.z],
            [0, 0, 0, 1]])

