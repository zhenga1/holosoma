#!/usr/bin/env python3
# viser_player.py
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import tyro
import viser  # type: ignore[import-not-found]  # pip install viser
import yourdfpy  # type: ignore[import-untyped]  # pip install yourdfpy
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

src_root = Path(__file__).resolve().parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
from holosoma_retargeting.config_types.viser import ViserConfig  # noqa: E402
from holosoma_retargeting.src.viser_utils import create_motion_control_sliders  # noqa: E402


def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    # expected: qpos [T, ?], and optional fps
    qpos = data["qpos"]
    fps = int(data["fps"]) if "fps" in data else 30
    diag_human = data["diag_human_mapped_world"] if "diag_human_mapped_world" in data else None
    diag_robot = data["diag_robot_mapped_world"] if "diag_robot_mapped_world" in data else None
    return qpos, fps, diag_human, diag_robot


def make_player(
    config: ViserConfig,
    qpos: np.ndarray,
    diag_human_mapped_world: np.ndarray | None = None,
    diag_robot_mapped_world: np.ndarray | None = None,
    fps: int | None = None,
):
    """
    qpos layout (MuJoCo order):
      [0:3]   robot base position (xyz)
      [3:7]   robot base quat (wxyz)
      [7:7+R] robot joint positions (R = actuated dof)
      [end-7:end-4] (optional) object position (xyz)
      [end-4:end]   (optional) object quat (wxyz)

    We'll infer R from the robot URDF's actuated joints in ViserUrdf.
    """
    server = viser.ViserServer()
    n_frames = int(qpos.shape[0])

    # Root frames
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    object_root = server.scene.add_frame("/object", show_axes=False)

    # URDFs (using yourdfpy so meshes show up)
    robot_urdf_y = yourdfpy.URDF.load(config.robot_urdf, load_meshes=True, build_scene_graph=True)
    vr = ViserUrdf(server, urdf_or_path=robot_urdf_y, root_node_name="/robot")

    vo = None
    if config.object_urdf:
        object_urdf_y = yourdfpy.URDF.load(config.object_urdf, load_meshes=True, build_scene_graph=True)
        vo = ViserUrdf(server, urdf_or_path=object_urdf_y, root_node_name="/object")

    # A tiny grid
    server.scene.add_grid("/grid", width=config.grid_width, height=config.grid_height, position=(0.0, 0.0, 0.0))

    # Figure robot DOF from actuated limits in ViserUrdf
    joint_limits = vr.get_actuated_joint_limits()
    robot_dof = len(joint_limits)

    # Use fps from config if not provided, otherwise use the one from npz file
    actual_fps = fps if fps is not None else config.fps

    # Set initial mesh visibility
    vr.show_visual = config.show_meshes
    if vo is not None:
        vo.show_visual = config.show_meshes

    # ---------- Additional GUI controls (mesh visibility) ----------
    with server.gui.add_folder("Display"):
        show_meshes_cb = server.gui.add_checkbox("Show meshes", initial_value=config.show_meshes)

    @show_meshes_cb.on_update
    def _(_):
        vr.show_visual = bool(show_meshes_cb.value)
        if vo is not None:
            vo.show_visual = bool(show_meshes_cb.value)

    diag_handles = []

    def _draw_keypoints(points: np.ndarray, name: str, rgba=(0.0, 0.0, 1.0, 1.0)):
        sphere = trimesh.primitives.Sphere(radius=0.02)
        vertices = sphere.vertices
        faces = sphere.faces
        color = tuple(int(c * 255) for c in rgba[:3])
        opacity = float(rgba[3])
        return server.scene.add_batched_meshes_simple(
            f"/{name}",
            vertices=vertices,
            faces=faces,
            batched_positions=np.asarray(points),
            batched_wxyzs=np.tile(np.array([1, 0, 0, 0]), (points.shape[0], 1)),
            batched_colors=color,
            opacity=opacity,
        )

    def _clear_diag_handles():
        for h in diag_handles:
            try:
                h.remove()
            except Exception:
                pass
        diag_handles.clear()

    def _should_log_frame(frame_idx: int, num_frames: int) -> bool:
        if not config.retarget_diagnostics or num_frames <= 0:
            return False
        last = num_frames - 1
        stride = int(config.retarget_diagnostics_stride)
        if stride <= 0:
            return frame_idx in (0, last)
        return frame_idx == 0 or frame_idx == last or (frame_idx % stride == 0)

    replay_last_diag_frame = {"idx": -1}

    def _on_frame_change(frame_idx: int, _q_frame: np.ndarray) -> None:
        if frame_idx == replay_last_diag_frame["idx"]:
            return
        replay_last_diag_frame["idx"] = frame_idx
        if not _should_log_frame(frame_idx, n_frames):
            _clear_diag_handles()
            return
        if diag_human_mapped_world is None or diag_robot_mapped_world is None:
            _clear_diag_handles()
            return
        if frame_idx >= len(diag_human_mapped_world) or frame_idx >= len(diag_robot_mapped_world):
            _clear_diag_handles()
            return
        h_world = diag_human_mapped_world[frame_idx]
        r_world = diag_robot_mapped_world[frame_idx]
        mean_corr_err = float(np.mean(np.linalg.norm(r_world - h_world, axis=1)))
        print(f"[viser_player replay diag] frame {frame_idx}/{n_frames - 1}  mean||r-h||_world(m)={mean_corr_err:.4g}")
        _clear_diag_handles()
        diag_handles.append(_draw_keypoints(h_world, "diag_human_corr", rgba=(1.0, 0.55, 0.1, 1.0)))
        diag_handles.append(_draw_keypoints(r_world, "diag_robot_corr", rgba=(0.55, 0.1, 1.0, 1.0)))

    # ---------- Use reusable motion control sliders from viser_utils ----------
    create_motion_control_sliders(
        server=server,
        viser_robot=vr,
        robot_base_frame=robot_root,
        motion_sequence=qpos,
        robot_dof=robot_dof,
        viser_object=vo if config.assume_object_in_qpos else None,
        object_base_frame=object_root if config.assume_object_in_qpos else None,
        contains_object_in_qpos=config.assume_object_in_qpos,
        initial_fps=actual_fps,
        initial_interp_mult=config.visual_fps_multiplier,
        loop=config.loop,
        on_frame_change=_on_frame_change,
    )
    if config.retarget_diagnostics:
        has_saved_diag = (
            diag_human_mapped_world is not None
            and diag_robot_mapped_world is not None
            and len(diag_human_mapped_world) == n_frames
            and len(diag_robot_mapped_world) == n_frames
        )
        print(
            f"[viser_player] retarget_diagnostics=on stride={config.retarget_diagnostics_stride} "
            f"(saved diag points: {'yes' if has_saved_diag else 'no'})"
        )
        if not has_saved_diag:
            print(
                "[viser_player] Missing diag arrays in NPZ. Re-run retarget with "
                "--retargeter.retarget-diagnostics to save demo/robot correspondence points."
            )
    print(
        f"[viser_player] Loaded {n_frames} frames | robot_dof={robot_dof} | "
        f"object={'yes' if (config.object_urdf and config.assume_object_in_qpos) else 'no'}"
    )
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")
    return server


def main(cfg: ViserConfig) -> None:
    """Main function for viser player."""
    qpos, fps, diag_human, diag_robot = load_npz(cfg.qpos_npz)
    make_player(
        config=cfg,
        qpos=qpos,
        diag_human_mapped_world=diag_human,
        diag_robot_mapped_world=diag_robot,
        fps=fps,
    )

    # keep process alive
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    cfg = tyro.cli(ViserConfig)
    main(cfg)
