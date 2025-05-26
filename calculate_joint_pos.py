import os

def load_joint_data(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    data = [list(map(float, line.strip().split())) for line in lines]
    return data

def get_joint_position(data, frame_index, joint_index, joints_per_frame=22):
    index = frame_index * joints_per_frame + joint_index
    if index >= len(data):
        raise IndexError("Frame or joint index out of range.")
    return data[index]  # [x, y, z]

def get_min_y_and_frame_for_joint(data, joint_index, joints_per_frame=22):
    min_y = float('inf')
    min_frame = -1
    total_frames = len(data) // joints_per_frame

    for frame_index in range(total_frames):
        joint = data[frame_index * joints_per_frame + joint_index]
        y = joint[1]  # y is at index 1
        if y < min_y:
            min_y = y
            min_frame = frame_index

    return min_y, min_frame

def main():
    txt_path = 'save/seed52_6834_a_person_is_doing_a_squat/initial_motion.txt'
    if not os.path.exists(txt_path):
        print(f"File not found: {txt_path}")
        return
    data = load_joint_data(txt_path)

    #####################################################################
    frame_index = 60
    joint_index = 5
    try:
        position = get_joint_position(data, frame_index, joint_index)
        print(f"Frame {frame_index}, Joint {joint_index}: x={position[0]:.4f}, y={position[1]:.4f}, z={position[2]:.4f}")
    except IndexError as e:
        print(f"Error: {e}")

    #####################################################################
    joint_index = 4 # left_knee
    try:
        min_y, min_frame = get_min_y_and_frame_for_joint(data, joint_index)
        print(f"Minimum y value for joint {joint_index}: {min_y:.4f} (at frame {min_frame})")
    except IndexError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()