# Video Fixer

This project reconstructs a video whose frames have been shuffled and may contain artifacts. It reorders the frames to restore the original video sequence.

## Approach

1. **Feature Extraction**:
   - Uses pretrained models (e.g., `mobilenet_v3_small`, `resnet18`) to extract features from video frames.

2. **Clustering**:
   - Frames are clustered based on feature similarity, and the most represented cluster is assumed to contain the original frames. The other are assumed to be artifacts and are removed. If artifacts don't get properly removed, it might be a good idea to lower the eps value in DBSCAN.

3. **Frame Reordering**:
   - Frames are reordered by calculating Euclidean distances between their features, iteratively selecting the closest frame. This gives a smooth video with a cut at the middle since we didn't iterate on the true first frame of the video.

4. **Identifying True Start and End**:
   - Four candidate frames (first/last frames and two frames around the cut) are identified to find the true start and end of the video. The two frames around the cut are found by their close distance since they are supposed to be consecutive. They are removed and we are only left with the first and last frame.

5. **Final Video Reconstruction**:
   - Two video candidates (starting from the true first or last frame) are reconstructed and compared using optical flow. The video with the highest optical flow is selected as the true video. The true video is reconstructed by iterating on the first frame.

## Installation

```pip -r requirements.txt```

## How to use

To use this program, simply use the following command in your python environment

```python3 main.py -i <your_videofile>```

This program uses pretrained models. You can choose the mobilenet_v3_small or the resnet18 model

```python3 main.py -i <your_video_file> -m <model>```

The default model is mobilenet_v3_small. To launch with resnet18, you can just type

```python3 main.py -i <your_video_file> -m resnet18```

Using resnet18 as a model usually gives better results than mobilenet_v3_small

## Demo

You can directly have a demo on the corrupted video provided in this repo.

```python3 main.py -i corrupted_video.mp4```

Results will generally be better if you use resnet18 instead of the base model

```python3 main.py -i corrupted_video.mp4 -m resnet18```

## GUI

You can also launch it in a GUI app

```streamlit run deployment.py```
