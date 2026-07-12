# Berkeley Humanoid Lite

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)
[![License](https://img.shields.io/badge/license-CC%20BY--SA%204.0-orange.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

**[Website](http://lite.berkeley-humanoid.org/)** | **[arXiv](https://arxiv.org/abs/2504.17249)** | **[Paper](https://lite.berkeley-humanoid.org/static/paper/demonstrating-berkeley-humanoid-lite.pdf)** | **[Video](https://youtu.be/dIdJGkMDFl4?si=SRD7HhQQbhM3JCRA)** | **[Documentation](https://berkeley-humanoid-lite.gitbook.io/berkeley-humanoid-lite-docs)** | **[Releases](https://berkeley-humanoid-lite.gitbook.io/docs/releases)**


Berkeley Humanoid Lite is an open-source, sub-$5,000 humanoid robot featuring modular 3D-printed gearboxes and widely available components, designed to democratize and advance humanoid robotics research.

This project is built on the values of open-source, accessibility, and customization, and it's continuously evolving. We welcome your feedback, issues, and pull requests in GitHub or joining our Discord.

## Overview

This repository is the workspace for the Berkeley Humanoid Lite project that contains everything we need, including policy training, sim2sim validation, real-world deployment, motion capture, and teleoperated manipulation controls.

Functionalities are organized into several submodules. We arrange the directory structure following the Isaac Lab convention, where each submodule can be installed as an extension:

- `source/berkeley_humanoid_lite/` contains the IsaacLab environment and task definitions.

- `source/berkeley_humanoid_lite_assets/` contains robot descriptions (URDF, MJCF, and USD) and the script to export these description files from Onshape project.

- `source/berkeley_humanoid_lite_lowlevel/` contains the lowlevel code running on the real robot. Only contents inside this folder is required to deploy to the real robot.

Except a few edge cases, all the commands should be invoked from the root directory of this repository. The entry points of different flows are collected in the `scripts/` directory.


## Local v1.2.3 Sim-to-Real Training Tasks

This working branch establishes Berkeley Humanoid Lite v1.2.3 as the current training baseline:

```text
Velocity-Lilgreen-Humanoid-v0    legacy upstream-style baseline
Velocity-Lilgreen-Stand-v0       v1.2.3 residual-action standing baseline
Velocity-Lilgreen-Hardware-v0    v1.2.3 residual-action staged locomotion curriculum
```

The v1.2.3 hardware-aligned tasks use action contract v3:

```text
a_raw
  -> clamp to [-1, +1]
  -> q_target = q_default + 0.20 * a_bounded
  -> clip to calibrated physical joint limits
```

Both Stand-v0 and Hardware-v0 use the same 45-D actor observation, 12-D action order, 50 Hz policy/action rate over 200 Hz physics, and residual mapping so a selected standing checkpoint can continue into the locomotion curriculum without changing action semantics. The measured deterministic q_default root-body COM height is `0.4899105727672577 m` and is used by both standing-height rewards and diagnostics.

See:

- `V1_2_3_ACTION_CONTRACT.md`
- `V1_2_3_CHANGES.md`
- `V1_2_3_BASELINE_GUIDE.md`

The v1.2/v1.2.1/v1.2.2 documents remain in the repository as historical records of the path to the current baseline.


## Getting Started

Please refer to our [Documentation](https://berkeley-humanoid-lite.gitbook.io/docs) to get started with software and hardware setup.

The latest release of CAD model and 3D print files can be accessed from the [Release](https://berkeley-humanoid-lite.gitbook.io/docs/releases) page.


## Contributing

We wholeheartedly welcome contributions from the community to make this robot platform more mature and useful for everyone. We appreciate any kind of contributions, including bug reports, feature requests, or code contributions.

Also, please reach out to us to tell us about your projects and how you are using this robot platform. We would love to feature your work on our website and social media.

## License

The code in this repository is licensed under [MIT License](https://opensource.org/license/mit). See the [LICENSE](LICENSE) file for details.

Other assets are under [Creative Commons Attribution-ShareAlike 4.0 International <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt="">](https://creativecommons.org/licenses/by-sa/4.0).

## Version history notes

The historical v1.2.1 and v1.2.2 notes remain available in `V1_2_1_SMOKE_FIX.md`, `V1_2_2_CHANGES.md`, and `V1_2_2_EXPERIMENT_GUIDE.md`.

## v1.2.3 residual baseline

v1.2.3 makes the symmetric `q_default`-centered ±0.20 rad residual mapping the canonical action contract for Stand-v0 and Hardware-v0, uses the measured nominal root-body COM height, separates stable-standing success from posture conformity, adds standing-only per-joint action/torque diagnostics, and exports explicit action-contract v3 metadata.


## v1.3.0 ST3215 actuator-aware branch

The repository preserves the v1.2.3 Stand-v0 and Hardware-v0 tasks as frozen residual-action baselines and adds:

- `Velocity-Lilgreen-Stand-ST3215-v0`
- `Velocity-Lilgreen-Hardware-ST3215-v0`

The new tasks preserve action contract v3 and the 45-D/12-D policy interface while inserting a Track-2-data-driven ST3215 response model between the residual policy target and the PhysX joint drive. See `V1_3_0_ST3215_TASKS.md`.

## v1.4.0 loaded ST3215 branch

v1.4.0 starts from the supplied v1.3.1 source tree, preserving its updated physical
joint limits and all earlier task registrations. It adds:

- `Velocity-Lilgreen-Stand-ST3215-Loaded-v0`
- `Velocity-Lilgreen-Hardware-ST3215-Loaded-v0`

The new tasks retain action contract v3 and the 45-D/12-D policy interface. They keep
the v1.3.1 suspended single-joint Stage-A response model and add a conservative loaded
standing-transition envelope calibrated from `track1_loaded_actuator_extension_v2`.
Knee pitch receives direction-conditioned crouch/stand-return treatment; other families
use conservative symmetric loaded envelopes unless the data strongly supports direction
conditioning. See `V1_4_0_ST3215_LOADED_TASKS.md`.

## v1.4.1 Hardware curriculum

v1.4.1 adds `Velocity-Lilgreen-Hardware-ST3215-Loaded-v1`, a safer Hardware-only curriculum for continuing the successful v1.4.0 loaded Stand checkpoint into locomotion. It preserves the v1.4.0 action contract and actuator model while slowing the command/disturbance expansion and strengthening standing/knee/action-health retention. See `V1_4_1_HARDWARE_CURRICULUM.md`.
