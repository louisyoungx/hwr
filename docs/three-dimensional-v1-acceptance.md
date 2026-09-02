# Household Embodied-Intelligence 3D Realistic Training Platform V1: Implementation and Acceptance Contract

> Status: In progress
> Date: 2026-08-10
> Principle: Missing evidence is equivalent to incomplete; 2D results do not count toward this contract.

See [Formal 3D Visual Training Runbook](formal-visual-training.md) for the formal visual data, training commands, and current execution status.

## 1. Definition of Done

V1 is not “being able to open a 3D window”; it is a locally reproducible visual-to-action training loop. The goal may be marked complete only when all eight gate groups A–H in this document have direct evidence.

| Gate | Required deliverable | Authoritative evidence | Automatic check |
|---|---|---|---|
| A Ownership boundary | The engine exists only in adapters; simulation and future real hardware share Observation/Action/RuntimeBackend | Import graph, interface tests | `check_architecture.py` |
| B 3D scenes | Three metric scenes for the kitchen, dining room, and living-room storage; meshes, materials, textures, lighting, and occlusion | Asset manifest, scene inspection report, raw screenshots | `verify_3d_assets.py` |
| C 3D robot | Four-wheel chassis, two 6-DOF arms with two-finger grippers, head RGB-D, and left/right wrist RGB; mass/inertia/collision/limits | Compiled model inventory and dynamics tests | `verify_robot_model.py` |
| D Real physics | Gravity, collisions, friction, and joint constraints; contact grasping; no teleportation, dynamic object-pose writes, or weld-based grasping | Contact logs, model audit, source scan, perturbation tests | `verify_physics_integrity.py` |
| E Non-privileged policy | Inference reads only camera payloads, proprioceptive state, action history, and task instructions | Observation allowlist tests, policy-input manifest | `verify_policy_inputs.py` |
| F Local training | Local online sampling, reinforcement learning, registration, and reload without experts or demonstrations; evaluation executes only the Actor | Training manifest, data-source audit, device information, model hash, action-source logs | `verify_training_lineage.py` |
| G Three-task evaluation | Three specified tasks, each with 40 isolated random seeds and unseen instruction rewrites, success rate and Wilson lower bound ≥70%, zero severe collisions; dual-arm-required tasks pass the single-arm-lockout ablation | Per-Episode reports, aggregate report, and ablation report | `evaluate_foundation_world_model.py` |
| H Raw evidence | Third-person plus head/left-wrist/right-wrist views, unedited; final state stable for ≥2 seconds | MP4, frame timeline, camera samples, hashes | `verify_replay_evidence.py` |

The check-script names are frozen delivery interfaces; the files may be absent before their corresponding stages are implemented, but human explanations may not substitute for them in the final state.

## 2. Formal Tasks

### `tidy_living_room_3d/v1`

- The robot starts at the room entrance and navigates around the coffee table or sofa;
- picks up a rubber duck and a mini soccer ball from the floor and places both objects in the wicker basket;
- success checks only the objects’ target volumes, velocities, and stability duration; it provides no grasp points or transport-action labels;
- before success, the left and right grippers must each form genuine two-finger contact, with at least 0.5 seconds of simultaneous contact.

### `clear_dining_table_3d/v1`

- A ceramic cup and wooden plate are placed on the tabletop, with a blue cup holder and orange plate holder as their respective targets;
- both objects must enter their respective target volumes and remain stable for at least 2 seconds;
- before success, the left and right grippers must each form genuine two-finger contact, with at least 0.5 seconds of simultaneous contact;
- if the same success determination can still be met after locking one arm, the formal ablation fails.

### `store_kitchen_items_3d/v1`

- The yellow and pink cleaning-agent bottles must enter the drawer’s left and right compartments, respectively;
- the drawer has a non-actuated slide rail and must be opened by robot contact to at least 0.30 m;
- before success, the left and right grippers must each form genuine two-finger contact, with at least 0.5 seconds of simultaneous contact;
- cabinet-door/drawer motion must be caused by robot contact or grasping; the task script may not set joint positions directly.

All three tasks must use the mobile chassis and both arms. Stationary tabletop grasping, sequential two-hand completion without concurrent contact, or meeting the same success gate after locking one arm does not pass. The environment defines only observations, rewards, termination, safety, and legal transformations; it provides no action answers.

## 3. Scene and Asset Gates

Every formal scene must satisfy all of the following:

- be metric, with clear real-world dimensions for rooms, passages, tables, and target containers;
- contain at least 3 pieces of static furniture, 1 target container, and 2 manipulable objects;
- use meshes or non-trivial procedural meshes for visible furniture and manipulable objects;
- include at least the floor, walls, and recognizable textured materials such as wood, fabric, and ceramic;
- configure ambient light and at least two directional or positional light sources;
- show normal perspective, occlusion, and shadows in the rendered result;
- keep visual meshes separate from simplified collision meshes, with their scale error constrained by the manifest;
- record the source URL, license, original hash, processed hash, and metric scaling for every external asset.

Renaming a set of untextured boxes/cylinders/spheres as “furniture” does not pass acceptance. Basic geometric primitives may be used as invisible collision bodies or small connectors, but they cannot constitute all visible content in a formal scene.

## 4. Robot Gates

The robot model must contain:

- four independently visible wheel bodies and wheel-ground contact collision bodies;
- chassis drive capable of differential motion;
- six revolute arm joints on each side, with per-joint position, velocity, and torque limits;
- one two-finger gripper on each side, each with two movable fingers, contact surfaces, and force limits;
- a head RGB camera with a co-located depth camera, plus left and right wrist RGB cameras;
- positive mass, positive-definite inertia, and collision geometry for every dynamic link;
- self-collision allowlists/blocklists, inter-arm collision constraints, and chassis-arm safety limits.

The model checker must read these facts from the compiled engine model rather than merely checking whether a string appears in the XML.

## 5. Physical Integrity and Anti-Cheating

### Allowed

- set randomized initial poses before and after `reset`;
- have the controller convert Actor actions into joint-position, velocity, or torque targets;
- have the success checker, rewarder, and training-time Critic read physical state only;
- have the environment define rewards, termination, success/safety outcomes, and legal environment transformations; the trainer must not require target relabeling;
- have the curriculum scheduler adjust initial-state difficulty and the randomization distribution based on closed-loop success rate.

### Prohibited

- write the position, orientation, or velocity of a manipulable object during an Episode;
- bind, attract, weld, or copy an object to a gripper based on end-effector distance;
- switch directly to “grasped/placed” based on action values;
- use a rule-based expert, human teleoperation, teacher policy, or action script to generate formal training labels;
- initialize the formal Actor from an expert or behavioral-cloning checkpoint;
- have the Critic, rewarder, or task script output actions, stages, or intermediate plans to the Actor;
- have the evaluation policy read entity IDs, ground-truth poses, target vectors, task stages, or success state;
- mix rule-based expert actions into evaluation, retry failed steps, or cut failed segments;
- use a model, seed, or trajectory in the video that differs from the evaluation report.

The runtime audit records the source of every action, object-pose changes, finger-contact pairs, collision impulses, and task-determination inputs. Anti-cheating checks scan the code, model, and Episode audit records together.

## 6. Visual Observations and Training

At each step, the formal policy may use only:

- head RGB;
- metric depth from the head;
- left and right wrist RGB;
- six-axis joint position/velocity for each arm;
- left and right gripper position and force;
- chassis odometry and IMU;
- versioned task instruction;
- a limited history of executed actions.

Training-time privileged state and policy inputs are stored separately. Enforce an allowlist when exporting Actor tensors; any extra field must fail the build. The training lineage must prove that no expert Episodes, human actions, teacher checkpoints, or behavioral-cloning initialization were used; after reloading the final checkpoint, every evaluation action `source` must be a learned-policy version.

## 7. Randomization and Evaluation

Each formal task must use at least 40 seeds unseen during training and natural-language rewrites disjoint from the training set. Each seed simultaneously determines and records:

- robot initial pose;
- manipulable-object positions and orientations;
- texture or material variants;
- light intensity, color temperature, or position;
- object mass and contact friction within reasonable ranges;
- camera noise or depth-missing ratio;
- camera extrinsic translation/rotation, focal-length ratio, and depth-measurement noise;
- actuator scale error, action latency, and observation latency.

Evaluation perturbation ranges must be wider than the training ranges, with at least one end outside the training range; every parameter, instruction text, and seed for training and evaluation must be written to the audit record.

Entry criteria:

- success rate of at least 70%;
- zero severe collisions;
- every success passes a 2-second stability window;
- after the normal dual-arm condition reaches the success gate, rerun the same evaluation set with the left arm locked and with the right arm locked; neither success rate may reach 10%;
- successful dual-arm Episodes contain a necessary concurrent-operation window proven by physical contact; movement of the joints alone is not a substitute;
- retain per-Episode reasons, step counts, contact statistics, and collision statistics for all 40 rounds in each scene;
- retain the training-seed and evaluation-seed sets in the report and verify that they are disjoint.

## 8. Video and Auditable Artifacts

Each formal task must produce at least one successful replay and one failed replay (if there is no failure in 20 rounds, produce another successful seed). Videos must:

- be recorded by the evaluation process simultaneously from third-person, head RGB, and wrist RGB views;
- run from a stable view before reset through the 2-second stability window after task determination;
- not change the action frequency, skip frames to hide failures, or splice different Episodes together;
- overlay task, seed, checkpoint hash, simulation time, and action source;
- have a sidecar JSON recording the corresponding Episode step and per-view frame hash for every frame.

Large models, datasets, and videos may be ignored by Git, but small manifests, aggregate reports, acceptance thresholds, and reproduction commands must be version-controlled.

## 9. Stages and Git Commits

1. ADR, interfaces, and acceptance matrix;
2. local MuJoCo installation and off-screen RGB/depth smoke test;
3. four-wheel dual-six-axis robot model, 16-dimensional action, and post-compilation audit;
4. pure contact grasping and stability determination;
5. three-scene assets, licensing, and verification;
6. procedural tasks, rewards, termination, legal environment transformations, and autonomous replay;
7. expert-free Actor-Critic training and registration;
8. 20-seed evaluation, dual-arm ablation, video, and anti-cheating audit;
9. clean-environment reproduction and final item-by-item audit.

Each stage must run the relevant tests and `scripts/check_python_size.py` and be committed separately after passing. Completing a stage does not mean that the overall goal is complete.
