# Installation

```pip install numpy tqdm moviepy opencv-python scikit-learn torch torchvision```

# How to use

To use this program, simply use the following command in your python environment

```python3 main.py -i <your_videofile>```

This program uses pretrained models. You can choose among mobile mobilenet_v3_small, resnet18 or resnet50

```python3 main.py -i <your_video_file> -m <model>```

The default model is mobilenet_v3_small

# Demo

You can directly have a demo on the corrupted video provided in this repo

```python3 main.py -i corrupted_video.mp4```

# GUI

You can also launch it in a GUI app

```streamlit run deployment.py```
