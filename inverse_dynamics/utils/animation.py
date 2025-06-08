import matplotlib.pyplot as plt
import sys
sys.path.append(".")  # Add current directory to path


def animate_with_dynamics_analysis(joints, joints_ori, trans, contact_output, com_results, fps=20, force_scale=0.001, save_path=None):
    """
    Create an animation showing skeleton with dynamics analysis including:
    - Total Ground Reaction Force (GRF) vector applied at Center of Pressure (CoP)
    - Required force vector m*(CoM_acc + g) applied at Center of Mass (CoM)
    - Linear dynamics loss for all frames
    
    Args:
        joints: Joint positions (num_frames, num_joints, 3)
        joints_ori: Joint orientations (num_frames, num_joints, 3, 3)
        trans: Global translation (num_frames, 3)
        contact_output: ContactOutput object containing forces and sphere positions
        com_results: Dictionary containing COM analysis results
        fps: Frames per second for the animation
        force_scale: Scale factor for force vectors (to make them visible)
        save_path: Path to save the animation (if None, will display instead)
    """
    import torch
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation
    
    # Extract COM data
    com_position = com_results["com_position"]
    com_acceleration = com_results["com_acceleration"]
    total_mass = com_results["total_mass"]
    
    # Extract CoP data
    cop_position = contact_output.cop
    
    # Convert to numpy for visualization
    if isinstance(com_position, torch.Tensor):
        com_position = com_position.cpu().numpy()
    if isinstance(com_acceleration, torch.Tensor):
        com_acceleration = com_acceleration.cpu().numpy()
    if isinstance(total_mass, torch.Tensor):
        total_mass = total_mass.cpu().numpy()
    if isinstance(cop_position, torch.Tensor):
        cop_position = cop_position.cpu().numpy()
    
    # Calculate required force: m*(a - g) for all frames
    # Note: gravity_vector points downward, so we subtract it from acceleration
    gravity_vector = np.array([0.0, -9.81, 0.0])  # Gravity pointing down in Y
    required_force = total_mass * (com_acceleration - gravity_vector)  # (num_frames, 3)
    
    # Calculate linear dynamics loss for all frames
    total_grf = contact_output.force.cpu().numpy()  # (num_frames, 3)
    force_residual = required_force - total_grf
    linear_loss_per_frame = np.linalg.norm(force_residual, axis=1)  # (num_frames,)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Main 3D animation subplot
    ax_main = fig.add_subplot(2, 2, (1, 3), projection="3d")
    
    # Loss plot subplot
    ax_loss = fig.add_subplot(2, 2, 2)
    
    # Force magnitude comparison subplot
    ax_forces = fig.add_subplot(2, 2, 4)
    
    # Define the kinematic tree
    kinematic_tree = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),        # Right leg
        (0, 6), (6, 7), (7, 8), (8, 9), (9, 10),       # Left leg
        (0, 11), (11, 12), (12, 13),                   # Spine to head
        (12, 14), (14, 15), (15, 16), (16, 17), (17, 18), # Right arm
        (12, 19), (19, 20), (20, 21), (21, 22), (22, 23)  # Left arm
    ]
    
    # Get global min/max for consistent axis limits
    all_data = joints.cpu().numpy()
    margin = 0.5
    global_xmin, global_ymin, global_zmin = all_data.min(axis=(0, 1)) - margin
    global_xmax, global_ymax, global_zmax = all_data.max(axis=(0, 1)) + margin
    
    # Create line objects for skeleton
    lines = []
    for _ in range(len(kinematic_tree)):
        line, = ax_main.plot([], [], [], linewidth=2, color="k")
        lines.append(line)
    
    # Create scatter object for joints
    scatter = ax_main.scatter([], [], [], color="black", s=30)
    
    # Create COM visualization
    com_scatter = ax_main.scatter([], [], [], color="red", s=100, alpha=0.8, label="Center of Mass")
    com_trail_line, = ax_main.plot([], [], [], color="red", alpha=0.6, linewidth=2, label="COM Trail")
    
    # Create CoP visualization
    cop_scatter = ax_main.scatter([], [], [], color="orange", s=80, alpha=0.8, marker="s", label="Center of Pressure")
    
    # Create force vector objects (will be updated each frame)
    grf_vector = None
    required_vector = None
    
    # Plot ground plane
    x = np.linspace(-1, 1, 2)
    y = np.linspace(-1, 1, 2)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    ground = ax_main.plot_surface(X, Y, Z, color="gray", alpha=0.3)
    
    # Setup loss plot
    frame_numbers = np.arange(len(linear_loss_per_frame))
    ax_loss.plot(frame_numbers, linear_loss_per_frame, "b-", linewidth=2, label="Linear Dynamics Loss")
    ax_loss.set_xlabel("Frame")
    ax_loss.set_ylabel("Loss (N)")
    ax_loss.set_title("Linear Dynamics Loss per Frame")
    ax_loss.grid(True)
    ax_loss.legend()
    
    # Current frame indicator line (will be updated)
    loss_vline = ax_loss.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="Current Frame")
    
    # Setup force comparison plot
    grf_magnitudes = np.linalg.norm(total_grf, axis=1)
    required_magnitudes = np.linalg.norm(required_force, axis=1)
    
    ax_forces.plot(frame_numbers, grf_magnitudes, "g-", linewidth=2, label="Total GRF Magnitude")
    ax_forces.plot(frame_numbers, required_magnitudes, "r-", linewidth=2, label="Required Force Magnitude")
    ax_forces.set_xlabel("Frame")
    ax_forces.set_ylabel("Force Magnitude (N)")
    ax_forces.set_title("Force Comparison")
    ax_forces.grid(True)
    ax_forces.legend()
    
    # Current frame indicator line for force plot
    force_vline = ax_forces.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="Current Frame")
    
    def init():
        # Setup main 3D plot
        ax_main.set_xlim([global_xmin, global_xmax])
        ax_main.set_ylim([global_zmin, global_zmax])
        ax_main.set_zlim([global_ymin, global_ymax])
        
        # Remove axis labels and grid
        ax_main.set_xticks([])
        ax_main.set_yticks([])
        ax_main.set_zticks([])
        ax_main.set_xlabel("")
        ax_main.set_ylabel("")
        ax_main.set_zlabel("")
        ax_main.grid(False)
        
        # Remove axis lines and background cube
        ax_main.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax_main.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax_main.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
        # Make panes transparent
        ax_main.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_main.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_main.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # Remove grid lines
        ax_main.xaxis._axinfo["grid"]['color'] = (1.0, 1.0, 1.0, 0.0)
        ax_main.yaxis._axinfo["grid"]['color'] = (1.0, 1.0, 1.0, 0.0)
        ax_main.zaxis._axinfo["grid"]['color'] = (1.0, 1.0, 1.0, 0.0)
        
        ax_main.view_init(elev=15, azim=-65)
        
        # Add legend for main plot
        blue_proxy = plt.Rectangle((0, 0), 1, 1, fc="blue", alpha=0.7)
        green_proxy = plt.Rectangle((0, 0), 1, 1, fc="green", alpha=0.7)
        red_proxy = plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.8)
        orange_proxy = plt.Rectangle((0, 0), 1, 1, fc="orange", alpha=0.8)
        ax_main.legend([blue_proxy, green_proxy, red_proxy, orange_proxy], 
                      ["GRF at CoP", "Required Force at CoM", "Center of Mass", "Center of Pressure"])
        
        return lines + [scatter, com_scatter, com_trail_line, cop_scatter, ground, loss_vline, force_vline]
    
    def update(frame):
        nonlocal grf_vector, required_vector
        
        # Get frame data
        frame_joints = joints[frame].cpu().numpy()
        current_com = com_position[frame]
        current_cop = cop_position[frame]
        current_grf = total_grf[frame]
        current_required = required_force[frame]
        current_loss = linear_loss_per_frame[frame]
        
        # Update skeleton
        for i, (start, end) in enumerate(kinematic_tree):
            xs = [frame_joints[start, 0], frame_joints[end, 0]]
            ys = [frame_joints[start, 2], frame_joints[end, 2]]
            zs = [frame_joints[start, 1], frame_joints[end, 1]]
            lines[i].set_data(xs, ys)
            lines[i].set_3d_properties(zs)
        
        # Update scatter points
        scatter._offsets3d = (frame_joints[:, 0], frame_joints[:, 2], frame_joints[:, 1])
        
        # Update COM position
        com_scatter._offsets3d = ([current_com[0]], [current_com[2]], [current_com[1]])
        
        # Update CoP position (only if valid, not NaN)
        if not np.isnan(current_cop).any():
            cop_scatter._offsets3d = ([current_cop[0]], [current_cop[2]], [current_cop[1]])
        else:
            # Hide CoP during flight phase by setting empty coordinates
            cop_scatter._offsets3d = ([], [], [])
        
        # Update COM trail (show last 30 frames)
        trail_start = max(0, frame - 30)
        trail_com = com_position[trail_start:frame+1]
        if len(trail_com) > 1:
            com_trail_line.set_data(trail_com[:, 0], trail_com[:, 2])
            com_trail_line.set_3d_properties(trail_com[:, 1])
        else:
            com_trail_line.set_data([], [])
            com_trail_line.set_3d_properties([])
        
        # Remove old force vectors
        if grf_vector is not None:
            grf_vector.remove()
        if required_vector is not None:
            required_vector.remove()
        
        # Plot total GRF vector at CoP (blue)
        if np.linalg.norm(current_grf) > 1e-6:
            grf_scaled = current_grf * force_scale
            grf_vector = ax_main.quiver(current_cop[0], current_cop[2], current_cop[1],
                                       grf_scaled[0], grf_scaled[2], grf_scaled[1],
                                       color="blue", alpha=0.8, arrow_length_ratio=0.1,
                                       linewidth=3)
        else:
            grf_vector = None
        
        # Plot required force vector m*(a-g) at CoM (green)
        if np.linalg.norm(current_required) > 1e-6:
            required_scaled = current_required * force_scale
            required_vector = ax_main.quiver(current_com[0], current_com[2], current_com[1],
                                           required_scaled[0], required_scaled[2], required_scaled[1],
                                           color="green", alpha=0.8, arrow_length_ratio=0.1,
                                           linewidth=3)
        else:
            required_vector = None
        
        # Update title with current values
        grf_moment_mag = np.linalg.norm(current_grf)
        req_mag = np.linalg.norm(current_required)
        title = (f"Frame {frame}/{len(joints)-1} | "
                f"GRF@CoP: {grf_moment_mag:.1f}N | Required@CoM: {req_mag:.1f}N | "
                f"Linear Loss: {current_loss:.3f}N")
        ax_main.set_title(title, fontsize=12)
        
        # Update vertical lines in subplots
        loss_vline.set_xdata([frame, frame])
        force_vline.set_xdata([frame, frame])
        
        # Update loss plot title with current value
        ax_loss.set_title(f"Linear Dynamics Loss per Frame (Current: {current_loss:.3f}N)")
        
        # Update force plot title with current values
        ax_forces.set_title(f"Force Comparison (GRF: {grf_moment_mag:.1f}N, Required: {req_mag:.1f}N)")
        
        artists = lines + [scatter, com_scatter, com_trail_line, cop_scatter, ground, loss_vline, force_vline]
        if grf_vector is not None:
            artists.append(grf_vector)
        if required_vector is not None:
            artists.append(required_vector)
        return artists
    
    # Create animation
    ani = FuncAnimation(
        fig, update, frames=range(len(joints)),
        init_func=init, blit=False, interval=1000/fps)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        # Choose writer based on file extension
        if save_path.lower().endswith('.mp4'):
            ani.save(save_path, writer="ffmpeg", fps=fps, bitrate=1800)
            print(f"Animation saved as MP4 to: {save_path}")
        elif save_path.lower().endswith('.gif'):
            ani.save(save_path, writer="pillow", fps=fps)
            print(f"Animation saved as GIF to: {save_path}")
        else:
            # Default to GIF if no recognized extension
            ani.save(save_path, writer="pillow", fps=fps)
            print(f"Animation saved as GIF to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return ani

def animate_with_angular_dynamics_analysis(joints, joints_ori, trans, contact_output, com_results, fps=20, moment_scale=0.01, save_path=None):
    """
    Create an animation showing skeleton with angular dynamics analysis including:
    - Moment from Ground Reaction Forces (M_GRF) about COM
    - Moment from Gravity about COM (M_GRM) 
    - Rate of change of angular momentum (dL/dt)
    - Angular dynamics loss for all frames
    
    The angular dynamics constraint is: M_GRF + M_GRM = dL/dt
    
    Args:
        joints: Joint positions (num_frames, num_joints, 3)
        joints_ori: Joint orientations (num_frames, num_joints, 3, 3)
        trans: Global translation (num_frames, 3)
        contact_output: ContactOutput object containing forces and sphere positions
        com_results: Dictionary containing COM analysis results
        fps: Frames per second for the animation
        moment_scale: Scale factor for moment vectors (to make them visible)
        save_path: Path to save the animation (if None, will display instead)
    """
    import torch
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation
    
    # Extract COM data
    com_position = com_results["com_position"]
    com_velocity = com_results["com_velocity"]
    moment_about_com = com_results["moment_about_com"]  # dL/dt
    total_mass = com_results["total_mass"]
    
    # Convert to numpy for visualization
    if isinstance(com_position, torch.Tensor):
        com_position = com_position.cpu().numpy()
    if isinstance(com_velocity, torch.Tensor):
        com_velocity = com_velocity.cpu().numpy()
    if isinstance(moment_about_com, torch.Tensor):
        moment_about_com = moment_about_com.cpu().numpy()
    if isinstance(total_mass, torch.Tensor):
        total_mass = total_mass.cpu().numpy()
    
    # Calculate moment from GRF about COM
    cop_position = contact_output.cop.cpu().numpy()
    total_grf = contact_output.force.cpu().numpy()
    total_torque = contact_output.torque.cpu().numpy()
    
    # M_GRF = r_com_to_cop × F_GRF + τ_GRF
    # Only calculate when there's significant contact force (> 100 N) and valid CoP
    grf_magnitudes = np.linalg.norm(total_grf, axis=1)
    contact_threshold = 100.0  # N - increased from 10.0 to better handle jumping motion
    
    # Check for valid CoP (not NaN) and sufficient force
    valid_cop_mask = ~np.isnan(cop_position).any(axis=1)  # True where CoP is not NaN
    sufficient_force_mask = grf_magnitudes > contact_threshold
    contact_frames = valid_cop_mask & sufficient_force_mask
    
    r_com_to_cop = np.zeros_like(cop_position)  # Initialize to zero
    moment_from_grf_force = np.zeros_like(cop_position)  # Initialize to zero
    
    # Only calculate for frames with valid contact
    if np.any(contact_frames):
        r_com_to_cop[contact_frames] = cop_position[contact_frames] - com_position[contact_frames]
        moment_from_grf_force[contact_frames] = np.cross(
            r_com_to_cop[contact_frames], 
            total_grf[contact_frames]
        )
    
    # Total moment from GRF (only add torque when there's contact)
    moment_from_grf = moment_from_grf_force.copy()
    moment_from_grf[contact_frames] += total_torque[contact_frames]
    
    # Calculate moment from gravity about COM (M_GRM)
    # For a rigid body, M_GRM = 0 because gravity acts at COM
    # But for multi-segment body, we need to consider each segment
    gravity_vector = np.array([0.0, -9.81, 0.0])
    
    # Calculate moment from gravity for each segment about global COM
    segment_masses = com_results["segment_masses"].cpu().numpy()
    segment_com_positions = com_results["segment_com_positions"].cpu().numpy()
    
    moment_from_gravity = np.zeros_like(com_position)  # (num_frames, 3)
    for frame in range(len(com_position)):
        for seg_idx in range(24):  # 24 segments
            if segment_masses[seg_idx] > 0:
                # Vector from global COM to segment COM
                r_com_to_seg = segment_com_positions[frame, seg_idx] - com_position[frame]
                # Gravity force on segment
                gravity_force = segment_masses[seg_idx] * gravity_vector
                # Moment contribution from this segment
                moment_from_gravity[frame] += np.cross(r_com_to_seg, gravity_force)
    
    # Calculate angular dynamics loss for all frames
    # Loss = ||M_GRF + M_GRM - dL/dt||
    total_external_moment = moment_from_grf + moment_from_gravity
    moment_residual = total_external_moment - moment_about_com
    angular_loss_per_frame = np.linalg.norm(moment_residual, axis=1)  # (num_frames,)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Main 3D animation subplot
    ax_main = fig.add_subplot(2, 2, (1, 3), projection="3d")
    
    # Loss plot subplot
    ax_loss = fig.add_subplot(2, 2, 2)
    
    # Moment magnitude comparison subplot
    ax_moments = fig.add_subplot(2, 2, 4)
    
    # Define the kinematic tree
    kinematic_tree = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),        # Right leg
        (0, 6), (6, 7), (7, 8), (8, 9), (9, 10),       # Left leg
        (0, 11), (11, 12), (12, 13),                   # Spine to head
        (12, 14), (14, 15), (15, 16), (16, 17), (17, 18), # Right arm
        (12, 19), (19, 20), (20, 21), (21, 22), (22, 23)  # Left arm
    ]
    
    # Get global min/max for consistent axis limits
    all_data = joints.cpu().numpy()
    margin = 0.5
    global_xmin, global_ymin, global_zmin = all_data.min(axis=(0, 1)) - margin
    global_xmax, global_ymax, global_zmax = all_data.max(axis=(0, 1)) + margin
    
    # Create line objects for skeleton
    lines = []
    for _ in range(len(kinematic_tree)):
        line, = ax_main.plot([], [], [], linewidth=2, color="k")
        lines.append(line)
    
    # Create scatter object for joints
    scatter = ax_main.scatter([], [], [], color="black", s=30)
    
    # Create COM visualization
    com_scatter = ax_main.scatter([], [], [], color="red", s=100, alpha=0.8, label="Center of Mass")
    com_trail_line, = ax_main.plot([], [], [], color="red", alpha=0.6, linewidth=2, label="COM Trail")
    
    # Create CoP visualization
    cop_scatter = ax_main.scatter([], [], [], color="orange", s=80, alpha=0.8, marker="s", label="Center of Pressure")
    
    # Create moment vector objects (will be updated each frame)
    grf_moment_vector = None
    gravity_moment_vector = None
    angular_momentum_moment_vector = None
    
    # Plot ground plane
    x = np.linspace(-1, 1, 2)
    y = np.linspace(-1, 1, 2)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    ground = ax_main.plot_surface(X, Y, Z, color="gray", alpha=0.3)
    
    # Setup loss plot
    frame_numbers = np.arange(len(angular_loss_per_frame))
    ax_loss.plot(frame_numbers, angular_loss_per_frame, "purple", linewidth=2, label="Angular Dynamics Loss")
    ax_loss.set_xlabel("Frame")
    ax_loss.set_ylabel("Loss (N⋅m)")
    ax_loss.set_title("Angular Dynamics Loss per Frame")
    ax_loss.grid(True)
    ax_loss.legend()
    
    # Current frame indicator line (will be updated)
    loss_vline = ax_loss.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="Current Frame")
    
    # Setup moment comparison plot
    grf_moment_magnitudes = np.linalg.norm(moment_from_grf, axis=1)
    gravity_moment_magnitudes = np.linalg.norm(moment_from_gravity, axis=1)
    angular_momentum_magnitudes = np.linalg.norm(moment_about_com, axis=1)
    total_external_magnitudes = np.linalg.norm(total_external_moment, axis=1)
    
    ax_moments.plot(frame_numbers, grf_moment_magnitudes, "b-", linewidth=2, label="M_GRF Magnitude")
    ax_moments.plot(frame_numbers, gravity_moment_magnitudes, "g-", linewidth=2, label="M_Gravity Magnitude")
    ax_moments.plot(frame_numbers, angular_momentum_magnitudes, "r-", linewidth=2, label="dL/dt Magnitude")
    ax_moments.plot(frame_numbers, total_external_magnitudes, "m--", linewidth=2, label="M_GRF + M_Gravity")
    ax_moments.set_xlabel("Frame")
    ax_moments.set_ylabel("Moment Magnitude (N⋅m)")
    ax_moments.set_title("Angular Moment Comparison")
    ax_moments.grid(True)
    ax_moments.legend()
    
    # Current frame indicator line for moment plot
    moment_vline = ax_moments.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="Current Frame")
    
    def init():
        # Setup main 3D plot
        ax_main.set_xlim([global_xmin, global_xmax])
        ax_main.set_ylim([global_zmin, global_zmax])
        ax_main.set_zlim([global_ymin, global_ymax])
        
        # Remove axis labels and grid
        ax_main.set_xticks([])
        ax_main.set_yticks([])
        ax_main.set_zticks([])
        ax_main.set_xlabel("")
        ax_main.set_ylabel("")
        ax_main.set_zlabel("")
        ax_main.grid(False)
        
        # Remove axis lines and background cube
        ax_main.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax_main.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax_main.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
        # Make panes transparent
        ax_main.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_main.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_main.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # Remove grid lines
        ax_main.xaxis._axinfo["grid"]['color'] = (1.0, 1.0, 1.0, 0.0)
        ax_main.yaxis._axinfo["grid"]['color'] = (1.0, 1.0, 1.0, 0.0)
        ax_main.zaxis._axinfo["grid"]['color'] = (1.0, 1.0, 1.0, 0.0)
        
        ax_main.view_init(elev=15, azim=-65)
        
        # Add legend for main plot
        blue_proxy = plt.Rectangle((0, 0), 1, 1, fc="blue", alpha=0.7)
        green_proxy = plt.Rectangle((0, 0), 1, 1, fc="green", alpha=0.7)
        magenta_proxy = plt.Rectangle((0, 0), 1, 1, fc="magenta", alpha=0.7)
        red_proxy = plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.8)
        orange_proxy = plt.Rectangle((0, 0), 1, 1, fc="orange", alpha=0.8)
        ax_main.legend([blue_proxy, magenta_proxy, green_proxy, red_proxy, orange_proxy], 
                      ["M_GRF", "M_Gravity", "dL/dt", "Center of Mass", "Center of Pressure"])
        
        return lines + [scatter, com_scatter, com_trail_line, cop_scatter, ground, loss_vline, moment_vline]
    
    def update(frame):
        nonlocal grf_moment_vector, gravity_moment_vector, angular_momentum_moment_vector
        
        # Get frame data
        frame_joints = joints[frame].cpu().numpy()
        current_com = com_position[frame]
        current_cop = cop_position[frame]
        current_grf_moment = moment_from_grf[frame]
        current_gravity_moment = moment_from_gravity[frame]
        current_angular_momentum_moment = moment_about_com[frame]
        current_loss = angular_loss_per_frame[frame]
        
        # Update skeleton
        for i, (start, end) in enumerate(kinematic_tree):
            xs = [frame_joints[start, 0], frame_joints[end, 0]]
            ys = [frame_joints[start, 2], frame_joints[end, 2]]
            zs = [frame_joints[start, 1], frame_joints[end, 1]]
            lines[i].set_data(xs, ys)
            lines[i].set_3d_properties(zs)
        
        # Update scatter points
        scatter._offsets3d = (frame_joints[:, 0], frame_joints[:, 2], frame_joints[:, 1])
        
        # Update COM position
        com_scatter._offsets3d = ([current_com[0]], [current_com[2]], [current_com[1]])
        
        # Update CoP position (only if valid, not NaN)
        if not np.isnan(current_cop).any():
            cop_scatter._offsets3d = ([current_cop[0]], [current_cop[2]], [current_cop[1]])
        else:
            # Hide CoP during flight phase by setting empty coordinates
            cop_scatter._offsets3d = ([], [], [])
        
        # Update COM trail (show last 30 frames)
        trail_start = max(0, frame - 30)
        trail_com = com_position[trail_start:frame+1]
        if len(trail_com) > 1:
            com_trail_line.set_data(trail_com[:, 0], trail_com[:, 2])
            com_trail_line.set_3d_properties(trail_com[:, 1])
        else:
            com_trail_line.set_data([], [])
            com_trail_line.set_3d_properties([])
        
        # Remove old moment vectors
        if grf_moment_vector is not None:
            grf_moment_vector.remove()
        if gravity_moment_vector is not None:
            gravity_moment_vector.remove()
        if angular_momentum_moment_vector is not None:
            angular_momentum_moment_vector.remove()
        
        # Plot moment from GRF at COM (blue)
        if np.linalg.norm(current_grf_moment) > 1e-6:
            grf_moment_scaled = current_grf_moment * moment_scale * 0.1
            grf_moment_vector = ax_main.quiver(current_com[0], current_com[2], current_com[1],
                                             grf_moment_scaled[0], grf_moment_scaled[2], grf_moment_scaled[1],
                                             color="blue", alpha=0.8, arrow_length_ratio=0.1,
                                             linewidth=3)
        else:
            grf_moment_vector = None
        
        # Plot moment from gravity at COM (magenta)
        if np.linalg.norm(current_gravity_moment) > 1e-6:
            gravity_moment_scaled = current_gravity_moment * moment_scale
            gravity_moment_vector = ax_main.quiver(current_com[0], current_com[2], current_com[1],
                                                 gravity_moment_scaled[0], gravity_moment_scaled[2], gravity_moment_scaled[1],
                                                 color="magenta", alpha=0.8, arrow_length_ratio=0.1,
                                                 linewidth=3)
        else:
            gravity_moment_vector = None
        
        # Plot rate of change of angular momentum at COM (green)
        if np.linalg.norm(current_angular_momentum_moment) > 1e-6:
            angular_momentum_scaled = current_angular_momentum_moment * moment_scale
            angular_momentum_moment_vector = ax_main.quiver(current_com[0], current_com[2], current_com[1],
                                                          angular_momentum_scaled[0], angular_momentum_scaled[2], angular_momentum_scaled[1],
                                                          color="green", alpha=0.8, arrow_length_ratio=0.1,
                                                          linewidth=3)
        else:
            angular_momentum_moment_vector = None
        
        # Update title with current values
        grf_moment_mag = np.linalg.norm(current_grf_moment)
        gravity_moment_mag = np.linalg.norm(current_gravity_moment)
        angular_momentum_mag = np.linalg.norm(current_angular_momentum_moment)
        contact_status = "Contact" if not np.isnan(current_cop).any() else "Flight"
        title = (f"Frame {frame}/{len(joints)-1} | {contact_status} | "
                f"M_GRF: {grf_moment_mag:.2f} | M_Grav: {gravity_moment_mag:.2f} | "
                f"dL/dt: {angular_momentum_mag:.2f} | Loss: {current_loss:.3f} N⋅m")
        ax_main.set_title(title, fontsize=11)
        
        # Update vertical lines in subplots
        loss_vline.set_xdata([frame, frame])
        moment_vline.set_xdata([frame, frame])
        
        # Update loss plot title with current value
        ax_loss.set_title(f"Angular Dynamics Loss per Frame (Current: {current_loss:.3f} N⋅m)")
        
        # Update moment plot title with current values
        ax_moments.set_title(f"Angular Moment Comparison (M_GRF: {grf_moment_mag:.2f}, dL/dt: {angular_momentum_mag:.2f} N⋅m)")
        
        artists = lines + [scatter, com_scatter, com_trail_line, cop_scatter, ground, loss_vline, moment_vline]
        if grf_moment_vector is not None:
            artists.append(grf_moment_vector)
        if gravity_moment_vector is not None:
            artists.append(gravity_moment_vector)
        if angular_momentum_moment_vector is not None:
            artists.append(angular_momentum_moment_vector)
        return artists
    
    # Create animation
    ani = FuncAnimation(
        fig, update, frames=range(len(joints)),
        init_func=init, blit=False, interval=1000/fps)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        # Choose writer based on file extension
        if save_path.lower().endswith('.mp4'):
            ani.save(save_path, writer="ffmpeg", fps=fps, bitrate=1800)
            print(f"Animation saved as MP4 to: {save_path}")
        elif save_path.lower().endswith('.gif'):
            ani.save(save_path, writer="pillow", fps=fps)
            print(f"Animation saved as GIF to: {save_path}")
        else:
            # Default to GIF if no recognized extension
            ani.save(save_path, writer="pillow", fps=fps)
            print(f"Animation saved as GIF to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return ani