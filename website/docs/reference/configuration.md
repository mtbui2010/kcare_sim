# Configuration

All in `sim/config/`.

| file | what |
|---|---|
| `task_open_drawer_house.yaml` | the house task: cabinet style, `scene: house`, `camera_profile: home`, kcare ready pose (head, lift, arm angles), cabinet placement relative to the gripper |
| `task_open_drawer.yaml` | the same for the office scene |
| `randomization_house.yaml` | per-episode ranges (placement, drawer target + prompts, head jitter, lift, light, textures) + `house_sites` |
| `randomization_real.yaml` | the office version (fixed evaluation seeds depend on its draw order — append, do not reorder) |
| `camera_real.yaml` | per camera: intrinsics, distortion, noise, exposure target, saturation, colour gain (office fit) |
| `camera_home.yaml` | colour overrides fitted at home (`calibrate_home.py`) |
| `camera_sensor.yaml` | motion blur, rolling shutter, white-balance mix, depth models |
| `house_layout.yaml` | written by `house_scene.py`: rooms, walls, spots, objects, bounds, start pose |
| `mjcf.yaml`, `sim_urdf.yaml` | robot model: actuators, gains, cameras; masses and collision strategy |

## Task file (excerpt)

```yaml
cabinet: A_home                 # cabinet.py STYLES key
scene: house                    # render training scenes in the house
camera_profile: home            # camera_home.yaml over camera_real.yaml
mode: left                      # kcare robot mode
ready_pose:
  head_pan_offset: -0.12        # calibrated head zero (rad)
  head: {ry: -41, rz: {left: -70, right: 70}}
  lift: {inputs: 0.5, range: [0.56, 1.1], mobile_height: 0.43}
placement:
  standoff: 0.15                # TCP to handle bar [m]
  dforward: 0.08                # matched to the robot: wrist camera to drawer front 0.445 m
  dshift: 0.035
```

## Camera sensor file (excerpt)

| key | meaning | head (Femto) | wrist (D405) |
|---|---|---|---|
| `exposure_ms` | shutter time for motion blur (jittered ×0.6–1.4) | 12 | 10 |
| `readout_ms` | rolling-shutter readout | 20 | 0 (global shutter) |
| `depth.type` | depth model | tof | stereo |
| `depth.range_m` | valid range | 0.5–3.86 | 0.07–6.0 |
| noise | | `sigma_mm: [1.0, 0.0012]` (a + b·z) | `sigma_disp_px: 0.07`, baseline 0.018 m |

Shutter times were not measured on the robot; they are labelled so in the file.
