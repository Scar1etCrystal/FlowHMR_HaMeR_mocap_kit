# A user-friendly motion capture solution


## Indroduction
This is a mocap kit combining two revolutionary researchs: [FLowHMR](https://github.com/flowhmr/flowhmr) and [HaMeR](https://github.com/geopavlakos/hamer).

The project can't be a reality without their exceptional works. **Heartfelt thanks for the researchers behind!**

## Hardware Requirements
1.NVIDIA 30/40/50 series GPU with at least 8GiB of VRAM (≥30 series and ≥10GiB VRAM recommended)
2.More than 25GB of disk space.

## How to use
- 1.Download the lacking model weights and environments (see `WEIGHTS_AND_ENV.md`)
- 2.Click `Open_Console.bat`
- 3.Browse a video and wait for finishing. The VRAM usage will increase as the video gets longer. For videos with moving camera, use lower `Camera_interval` .

- 4.Import `final.fbx` motion file to Blender, C4D, Maya, etc.

## Tips
- 1.For low VRAM users(8GB), try to only capture static-camera videos, lower batch size, and enable `static camera`.


