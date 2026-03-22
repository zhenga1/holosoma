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
from holosoma_retargeting.config_types.data_type import DEMO_JOINTS_REGISTRY  # noqa: E402
from holosoma_retargeting.src.utils import load_intermimic_data  # noqa: E402


@dataclass
class DemoPlayerConfig:
    """Configuration for demo player visualization."""

    demo_file: str = "demo_data/OMOMO_new/sub3_largebox_003.pt"
    """Path to demo file (.pt, .npy, or .npz)."""

    data_format: str = "smplh"
    """Data format ('smplh', 'lafan', 'mocap', etc.)."""

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


def _maybe_idx(joint_to_idx: dict[str, int], name: str) -> int | None:
    return joint_to_idx.get(name)


def get_mocap_skeleton_connections(demo_joints: list[str]) -> list[tuple[int, int]]:
    """Get parent-child relationships for the 'mocap' skeleton by joint names."""
    joint_to_idx = {name: i for i, name in enumerate(demo_joints)}

    def add(edges: list[tuple[int, int]], child: str, parent: str) -> None:
        c = _maybe_idx(joint_to_idx, child)
        p = _maybe_idx(joint_to_idx, parent)
        if c is not None and p is not None:
            edges.append((c, p))

    edges: list[tuple[int, int]] = []

    # Core body (Mixamo-ish)
    add(edges, "Spine", "Hips")
    add(edges, "Spine1", "Spine")
    add(edges, "Neck", "Spine1")
    add(edges, "Head", "Neck")

    # Legs
    add(edges, "LeftUpLeg", "Hips")
    add(edges, "LeftLeg", "LeftUpLeg")
    add(edges, "LeftFoot", "LeftLeg")
    add(edges, "LeftToeBase", "LeftFoot")
    add(edges, "LeftFootMod", "LeftFoot")

    add(edges, "RightUpLeg", "Hips")
    add(edges, "RightLeg", "RightUpLeg")
    add(edges, "RightFoot", "RightLeg")
    add(edges, "RightToeBase", "RightFoot")
    add(edges, "RightFootMod", "RightFoot")

    # Arms
    add(edges, "LeftShoulder", "Spine1")
    add(edges, "LeftArm", "LeftShoulder")
    add(edges, "LeftForeArm", "LeftArm")
    add(edges, "LeftHand", "LeftForeArm")

    add(edges, "RightShoulder", "Spine1")
    add(edges, "RightArm", "RightShoulder")
    add(edges, "RightForeArm", "RightArm")
    add(edges, "RightHand", "RightForeArm")

    # Hands
    add(edges, "LeftHandThumb1", "LeftHand")
    add(edges, "LeftHandThumb2", "LeftHandThumb1")
    add(edges, "LeftHandThumb3", "LeftHandThumb2")
    add(edges, "LeftHandIndex1", "LeftHand")
    add(edges, "LeftHandIndex2", "LeftHandIndex1")
    add(edges, "LeftHandIndex3", "LeftHandIndex2")
    add(edges, "LeftHandMiddle1", "LeftHand")
    add(edges, "LeftHandMiddle2", "LeftHandMiddle1")
    add(edges, "LeftHandMiddle3", "LeftHandMiddle2")
    add(edges, "LeftHandRing1", "LeftHand")
    add(edges, "LeftHandRing2", "LeftHandRing1")
    add(edges, "LeftHandRing3", "LeftHandRing2")
    add(edges, "LeftHandPinky1", "LeftHand")
    add(edges, "LeftHandPinky2", "LeftHandPinky1")
    add(edges, "LeftHandPinky3", "LeftHandPinky2")

    add(edges, "RightHandThumb1", "RightHand")
    add(edges, "RightHandThumb2", "RightHandThumb1")
    add(edges, "RightHandThumb3", "RightHandThumb2")
    add(edges, "RightHandIndex1", "RightHand")
    add(edges, "RightHandIndex2", "RightHandIndex1")
    add(edges, "RightHandIndex3", "RightHandIndex2")
    add(edges, "RightHandMiddle1", "RightHand")
    add(edges, "RightHandMiddle2", "RightHandMiddle1")
    add(edges, "RightHandMiddle3", "RightHandMiddle2")
    add(edges, "RightHandRing1", "RightHand")
    add(edges, "RightHandRing2", "RightHandRing1")
    add(edges, "RightHandRing3", "RightHandRing2")
    add(edges, "RightHandPinky1", "RightHand")
    add(edges, "RightHandPinky2", "RightHandPinky1")
    add(edges, "RightHandPinky3", "RightHandPinky2")

    return edges


def get_skeleton_connections(data_format: str, num_joints: int) -> list[tuple[int, int]]:
    """Return skeleton edges (child_idx, parent_idx) for a given data_format."""
    if data_format == "smplh":
        return [(c, p) for (c, p) in get_smplh_skeleton_connections() if c < num_joints and p < num_joints]

    demo_joints = DEMO_JOINTS_REGISTRY.get(data_format)
    if demo_joints is None:
        return []

    if data_format == "mocap":
        return [(c, p) for (c, p) in get_mocap_skeleton_connections(demo_joints) if c < num_joints and p < num_joints]

    return []


def score_connections(human_joints: np.ndarray, connections: list[tuple[int, int]], max_frames: int = 10) -> float:
    """Heuristic score: smaller mean bone length means more plausible connections."""
    if not connections:
        return float("inf")

    frames = human_joints[: min(max_frames, human_joints.shape[0])]
    dists = []
    for child_idx, parent_idx in connections:
        # Euclidean distance in joint space
        diff = frames[:, child_idx, :] - frames[:, parent_idx, :]
        dists.append(np.linalg.norm(diff, axis=-1))  # (F,)

    # Average distance across edges and frames
    return float(np.mean([float(np.mean(di)) for di in dists]))


def load_demo_data(demo_file: str, data_format: str):
    """Load demo data for different formats."""
    # Default path: original InterMimic SMPL-H .pt format
    if data_format == "smplh" and demo_file.endswith(".pt"):
        human_joints, object_poses = load_intermimic_data(demo_file)
        return human_joints, object_poses

    # Generic NumPy-based formats (e.g., lafan, mocap) stored as .npy or .npz.
    data = np.load(demo_file, allow_pickle=True)

    # Handle npz containers: try common key names first.
    if isinstance(data, np.lib.npyio.NpzFile):
        for key in ("human_joints", "joints", "positions"):
            if key in data:
                human_joints = np.asarray(data[key])
                break
        else:
            # Fallback: use the first array in the container
            first_key = next(iter(data.files))
            human_joints = np.asarray(data[first_key])
        object_poses = data.get("object_poses", None)
        if object_poses is not None:
            object_poses = np.asarray(object_poses)
    else:
        # Plain .npy array
        human_joints = np.asarray(data)
        object_poses = None

    # Ensure shape is (T, num_joints, 3) if possible.
    if human_joints.ndim == 2 and human_joints.shape[1] % 3 == 0:
        T = human_joints.shape[0]
        J = human_joints.shape[1] // 3
        human_joints = human_joints.reshape(T, J, 3)

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
    print(f"Using data_format: {cfg.data_format}")
    human_joints, object_poses = load_demo_data(cfg.demo_file, cfg.data_format)
    
    T, num_joints, _ = human_joints.shape
    print(f"Loaded {T} frames with {num_joints} joints")

    expected_joints = DEMO_JOINTS_REGISTRY.get(cfg.data_format)
    if expected_joints is not None:
        if len(expected_joints) != num_joints:
            print(
                f"[demo_player] [WARN] Expected {len(expected_joints)} joints for data_format='{cfg.data_format}', "
                f"but file contains {num_joints} joints. Skeleton connections may be incorrect."
            )
        else:
            print(f"[demo_player] Expected joint count matches registry: {num_joints}")
    
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
        connections = get_skeleton_connections(cfg.data_format, num_joints)

        # Heuristic: for mocap, the joint ordering in the file sometimes doesn't match
        # our registry; choose between smplh-topology and mocap-topology by which yields
        # shorter average bone edges (more human-like).
        if cfg.data_format == "mocap" and connections:
            # get the average bone length for mocap and smplh connections.
            mocap_score = score_connections(human_joints, connections)
            smplh_connections = get_skeleton_connections("smplh", num_joints)
            smplh_score = score_connections(human_joints, smplh_connections)
            if smplh_connections and smplh_score < mocap_score:
                print(
                    f"[demo_player] mocap topology picked SMPLH (mocap_score={mocap_score:.4f}, "
                    f"smplh_score={smplh_score:.4f})"
                )
                connections = smplh_connections
            else:
                print(
                    f"[demo_player] mocap topology picked MOCAP (mocap_score={mocap_score:.4f}, "
                    f"smplh_score={smplh_score:.4f})"
                )

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
