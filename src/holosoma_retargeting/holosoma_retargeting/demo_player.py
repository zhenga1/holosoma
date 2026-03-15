#!/usr/bin/env python3
"""
Visualize original human demo data from .pt files.
Shows human joint positions as keypoints with optional object visualization.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

src_root = Path(__file__).resolve().parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS  # noqa: E402
from holosoma_retargeting.src.utils import load_intermimic_data  # noqa: E402


@dataclass
class DemoPlayerConfig:
    """Configuration for demo player visualization."""

    demo_file: str = "demo_data/OMOMO_new/sub3_largebox_003.pt"
    """Path to .pt demo file."""

    object_urdf: str | None = None
    """Path to object URDF file (optional)."""

    fps: int = 30
    """Frames per second for playback."""

    loop: bool = False
    """Whether to loop playback."""

    grid_width: float = 8.0
    """Grid width for visualization."""

    grid_height: float = 8.0
    """Grid height for visualization."""

    joint_radius: float = 0.03
    """Radius of joint spheres."""

    show_skeleton: bool = True
    """Whether to show skeleton connections between joints."""

    min_fps: int = 1
    """Minimum FPS setting."""

    max_fps: int = 240
    """Maximum FPS setting."""


def get_smplh_skeleton_connections():
    """Get parent-child relationships for SMPL-H skeleton."""
    # Define skeleton structure based on SMPL-H joint hierarchy
    # Format: (child_idx, parent_idx)
    connections = [
        # Pelvis is root (index 0)
        (1, 0),   # L_Hip -> Pelvis
        (5, 0),   # R_Hip -> Pelvis
        (9, 0),   # Torso -> Pelvis
        
        # Left leg
        (2, 1),   # L_Knee -> L_Hip
        (3, 2),   # L_Ankle -> L_Knee
        (4, 3),   # L_Toe -> L_Ankle
        
        # Right leg
        (6, 5),   # R_Knee -> R_Hip
        (7, 6),   # R_Ankle -> R_Knee
        (8, 7),   # R_Toe -> R_Ankle
        
        # Spine
        (10, 9),  # Spine -> Torso
        (11, 10), # Chest -> Spine
        (12, 11), # Neck -> Chest
        (13, 12), # Head -> Neck
        
        # Left arm
        (14, 11), # L_Thorax -> Chest
        (15, 14), # L_Shoulder -> L_Thorax
        (16, 15), # L_Elbow -> L_Shoulder
        (17, 16), # L_Wrist -> L_Elbow
        
        # Left hand fingers
        (18, 17), # L_Index1 -> L_Wrist
        (19, 18), # L_Index2 -> L_Index1
        (20, 19), # L_Index3 -> L_Index2
        (21, 17), # L_Middle1 -> L_Wrist
        (22, 21), # L_Middle2 -> L_Middle1
        (23, 22), # L_Middle3 -> L_Middle2
        (24, 17), # L_Pinky1 -> L_Wrist
        (25, 24), # L_Pinky2 -> L_Pinky1
        (26, 25), # L_Pinky3 -> L_Pinky2
        (27, 17), # L_Ring1 -> L_Wrist
        (28, 27), # L_Ring2 -> L_Ring1
        (29, 28), # L_Ring3 -> L_Ring2
        (30, 17), # L_Thumb1 -> L_Wrist
        (31, 30), # L_Thumb2 -> L_Thumb1
        (32, 31), # L_Thumb3 -> L_Thumb2
        
        # Right arm
        (33, 11), # R_Thorax -> Chest
        (34, 33), # R_Shoulder -> R_Thorax
        (35, 34), # R_Elbow -> R_Shoulder
        (36, 35), # R_Wrist -> R_Elbow
        
        # Right hand fingers
        (37, 36), # R_Index1 -> R_Wrist
        (38, 37), # R_Index2 -> R_Index1
        (39, 38), # R_Index3 -> R_Index2
        (40, 36), # R_Middle1 -> R_Wrist
        (41, 40), # R_Middle2 -> R_Middle1
        (42, 41), # R_Middle3 -> R_Middle2
        (43, 36), # R_Pinky1 -> R_Wrist
        (44, 43), # R_Pinky2 -> R_Pinky1
        (45, 44), # R_Pinky3 -> R_Pinky2
        (46, 36), # R_Ring1 -> R_Wrist
        (47, 46), # R_Ring2 -> R_Ring1
        (48, 47), # R_Ring3 -> R_Ring2
        (49, 36), # R_Thumb1 -> R_Wrist
        (50, 49), # R_Thumb2 -> R_Thumb1
        (51, 50), # R_Thumb3 -> R_Thumb2
    ]
    return connections


def load_demo_data(demo_file: str):
    """Load demo data from .pt file."""
    human_joints, object_poses = load_intermimic_data(demo_file)
    return human_joints, object_poses


def create_joint_sphere(radius: float):
    """Create a sphere mesh for joints."""
    import trimesh
    
    sphere = trimesh.primitives.Sphere(radius=radius)
    return sphere.vertices, sphere.faces


def main(cfg: DemoPlayerConfig) -> None:
    """Main function for demo player."""
    # Load demo data
    print(f"Loading demo data from: {cfg.demo_file}")
    human_joints, object_poses = load_demo_data(cfg.demo_file)
    
    T, num_joints, _ = human_joints.shape
    print(f"Loaded {T} frames with {num_joints} joints")
    
    if object_poses is not None:
        print(f"Object poses available: {object_poses.shape}")
    
    # Setup viser server
    server = viser.ViserServer()
    server.scene.add_grid(
        "/grid",
        width=cfg.grid_width,
        height=cfg.grid_height,
        position=(0.0, 0.0, 0.0),
    )
    
    # Load object URDF if provided
    vo = None
    object_root = None
    if cfg.object_urdf:
        object_root = server.scene.add_frame("/object", show_axes=False)
        object_urdf_y = yourdfpy.URDF.load(
            cfg.object_urdf, load_meshes=True, build_scene_graph=True
        )
        vo = ViserUrdf(server, urdf_or_path=object_urdf_y, root_node_name="/object")
        vo.show_visual = True
    
    # Create joint spheres
    sphere_vertices, sphere_faces = create_joint_sphere(cfg.joint_radius)
    
    # Create skeleton connections if enabled
    skeleton_lines = None
    if cfg.show_skeleton:
        connections = get_smplh_skeleton_connections()
        num_connections = len(connections)
        # Initialize with empty points (will be updated in update_frame)
        # Shape should be (N, 2, 3) for N line segments
        skeleton_lines = server.scene.add_line_segments(
            "/skeleton",
            points=np.zeros((num_connections, 2, 3)),
            colors=np.tile(np.array([200, 200, 200]), (num_connections, 2, 1)),
            line_width=2.0,
        )
    
    # Create joint meshes
    joint_meshes = server.scene.add_batched_meshes_simple(
        "/joints",
        vertices=sphere_vertices,
        faces=sphere_faces,
        batched_positions=np.zeros((num_joints, 3)),
        batched_wxyzs=np.tile(np.array([1, 0, 0, 0]), (num_joints, 1)),
        batched_colors=(100, 150, 255),
        opacity=0.8,
    )
    
    # Playback controls
    with server.gui.add_folder("Playback"):
        play_pause = server.gui.add_button("Play/Pause")
        t_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=T - 1,
            step=1,
            initial_value=0,
        )
        fps_slider = server.gui.add_slider(
            "FPS",
            min=cfg.min_fps,
            max=cfg.max_fps,
            step=1,
            initial_value=cfg.fps,
        )
        loop_cb = server.gui.add_checkbox("Loop", initial_value=cfg.loop)
    
    # Playback state
    is_playing = False
    current_frame = 0
    
    def update_frame(frame_idx: int):
        """Update visualization for a given frame."""
        idx = int(np.clip(frame_idx, 0, T - 1))
        
        # Update joint positions
        joint_positions = human_joints[idx]  # (num_joints, 3)
        joint_meshes.batched_positions = joint_positions
        
        # Update skeleton connections
        if cfg.show_skeleton and skeleton_lines is not None:
            connections = get_smplh_skeleton_connections()
            segments = []
            colors = []
            for child_idx, parent_idx in connections:
                if child_idx < num_joints and parent_idx < num_joints:
                    # Each segment is [parent_pos, child_pos] with shape (2, 3)
                    segments.append([joint_positions[parent_idx], joint_positions[child_idx]])
                    colors.append([200, 200, 200])
            if segments:
                # Stack to get shape (N, 2, 3) for N segments
                skeleton_lines.points = np.array(segments)
                # Colors should also be (N, 2, 3) or (N, 2, 1) - repeat for start and end points
                color_array = np.array(colors)  # (N, 3)
                skeleton_lines.colors = np.repeat(color_array[:, np.newaxis, :], 2, axis=1)  # (N, 2, 3)
        
        # Update object pose if available
        if object_poses is not None and vo is not None and object_root is not None:
            obj_pose = object_poses[idx]  # [qw, qx, qy, qz, x, y, z]
            obj_pos = obj_pose[4:7]
            obj_quat = obj_pose[0:4]  # wxyz
            object_root.position = obj_pos
            object_root.wxyz = obj_quat
    
    @play_pause.on_click
    def _(_):
        nonlocal is_playing
        is_playing = not is_playing
    
    @t_slider.on_update
    def _(_):
        nonlocal current_frame
        current_frame = int(t_slider.value)
        update_frame(current_frame)
    
    @fps_slider.on_update
    def _(_):
        pass  # FPS is used in the update loop
    
    # Initialize frame 0
    update_frame(0)
    
    # Playback loop
    print(f"Demo player ready. Open the URL above to view.")
    print(f"Frames: {T}, Joints: {num_joints}")
    
    last_time = time.time()
    
    while True:
        now = time.time()
        dt = 1.0 / fps_slider.value if fps_slider.value > 0 else 1.0 / 30.0
        
        if is_playing:
            if now - last_time >= dt:
                current_frame += 1
                if current_frame >= T:
                    if loop_cb.value:
                        current_frame = 0
                    else:
                        current_frame = T - 1
                        is_playing = False
                
                t_slider.value = current_frame
                update_frame(current_frame)
                last_time = now
        
        time.sleep(0.01)


if __name__ == "__main__":
    cfg = tyro.cli(DemoPlayerConfig)
    main(cfg)
