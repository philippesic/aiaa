# AI Antialiasing

AIAA is a techinique that uses a neural network to simulate the high-quality antialiasing of SSAA4x as a fast post process.

The model is currently trained on a set of renders of ```demo_scene``` at both stadard and supersampled resolutions downscaled for antialiasing.

Limited training data also confines this model's accuracy to ```demo_scene```.

# How to Run

There is no official implementation yet. For now, clone the repository and run ```main.py``` which will walk you through generating data, training, and running.

Run ```check_version.py``` to check Cuda Compatability.

# Results

Below is a comparison of a frame aliased, supersampled, and AIAA at 1080p (Click to view images in full resolution).

<img src="output/1080/test/test.png" width="1920" height="1080">

*Alias*

<img src="renders/1080/antialias/demo_scene_frame0000_antialias.png" width="1920" height="1080">

*SSAA4x*

<img src="output/1080/result/result.png" width="1920" height="1080">

*AIAA*

# Requirements
- Cuda-capable GPU with compute > 7.0 (RTX 20-Series or Newer)
- Python 3.9 - 3.12

# Dependencies

- Pillow
- Torch 2.8.0 with Cuda 12.6
- Torchvision 0.23.0
- TQDM
- Triton-Windows 3.4.0.post20

# Changelog
- Tuned Model Quality
- Added ```main.py``` for execution
