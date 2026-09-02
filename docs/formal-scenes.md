# Formal 3D Household Scenes

This document records the scene assembly facts for `household_v1`. The asset gallery is only an asset and camera check, not a training result; only a complete Episode video subsequently executed by a loaded learned policy counts as task reproduction.

## Scene Inventory

| Scene | Real-world scale and primary furniture | Manipulable objects | Target/movable furniture |
|---|---|---|---|
| `living_room_3d/v1` | 6.0 m × 5.0 m; leather sofa, solid-wood coffee table, wicker storage basket, rug | rubber duck, soccer ball | storage basket with collision bodies for the base and four walls |
| `dining_room_3d/v1` | 6.4 m × 5.4 m; solid-wood round table, three dining chairs, sideboard | ceramic cup, wooden plate | separate target volumes for the cup holder and plate holder |
| `kitchen_3d/v1` | 6.8 m × 5.6 m; three wooden cabinet units, island, worktop | two cleaning-agent bottles with different meshes/masses | drawer with left and right compartments; 0.42 m non-actuated slide rail |

All external meshes undergo Y-up to Z-up conversion, metric scaling, and XY centering; see
`assets/manifests/household_v1_sources.json` and the lock file. The formal MJCF declares the following for every manipulable object:

- a high-detail visible mesh with UVs/textures and collisions disabled;
- independent simplified collision geometry, mass, and friction parameters;
- a free joint managed by MuJoCo; task code may not write its pose during an Episode.

The kitchen drawer has only a slide joint with limits, damping, and friction; it has no actuator, weld, or equality constraint. The current stage proves only its physical structure; subsequent training/evaluation must also provide contact logs and raw video showing the gripper contacting and opening the drawer.

## Formal Training and Evaluation Runtime

`hwr.adapters.mujoco.formal_household_backend.MujocoFormalHouseholdDualArmBackend`
connects the three asset sets above to a unified 16-dimensional dual-arm runtime. Both the training and final-evaluation entry points load
`configs/tasks/formal_3d_v1.json` directly and no longer substitute simplified basket, tray, or drawer proxy tasks for the formal scenes.

Each task contains 4 training instructions and 3 non-overlapping evaluation rewrites. Training randomization covers object mass/friction, lighting,
materials, RGB/depth noise, camera extrinsics, focal length, actuator scaling, and action/observation latency; final evaluation automatically uses
wider out-of-distribution ranges. Randomization changes only the environment; it does not generate actions or task stages for the policy.

Success requires all manipulable objects to remain stable within their respective target volumes for 2 seconds, zero severe collisions, genuine two-finger
contact at each gripper, and at least 0.5 seconds of simultaneous contact by both arms. The kitchen additionally requires the non-actuated drawer to be physically opened to 0.30 m. All
fields enter only environment rewards, termination, and read-only audits; the Actor still sees only the four cameras, proprioception, action history, and the raw instruction.

## Reproduction Asset Gallery

```bash
python -m pip install -e '.[sim3d,assets3d,video]'
python scripts/verify_3d_assets.py
python -m hwr.apps.render_formal_scenes --output artifacts/formal-scenes.png
```

The left side of each gallery row is an independent evidence camera; the right side is the robot-head RGB view of the same model at the same physical time. The image is explicitly marked `not a trained rollout` and cannot replace a policy closed-loop video.
