# Berkeley Humanoid (myrobot)

## Floor contact (why a `ground` geom alone may “do nothing”)

1. **Named ground plane** — Use a worldbody `<geom name="ground" type="plane" .../>` so filters that look for the substring `"ground"` can see the floor.

2. **Distance threshold** — Non-penetration only runs for pairs whose distance is within `collision_detection_threshold` (default **0.1 m**). SMPL/OMOMO scaling often places feet **more than 10 cm** above `z=0` at the first solve, so **no floor constraints were applied** even with a plane. The retargeter code now widens the query for ground pairs and adds explicit **ground ↔ foot collision geom** candidates; you can still raise the cap with CLI, e.g.  
   `--retargeter.collision-detection-threshold 0.5`

## Tuning

- **`robot_height`** in `config_types/robot.py` sets `calculate_scale_factor(...)` for SMPL-H / OMOMO. Measure your model’s standing height (m) and set `myrobot` accordingly if poses look too squat or stretched.
- **`JOINTS_MAPPINGS`** for `("smplh", "myrobot")` is a draft; poor pelvis/foot mapping also hurts standing quality.
