import streamlit as st
import restoration
import tempfile

from moviepy import VideoFileClip, ImageSequenceClip
from torchvision.models import mobilenet_v3_small, resnet18


models = {
        "mobilenet_v3_small": mobilenet_v3_small,
        "resnet18": resnet18,
        }

# GUI
st.title("Video Fixer")
st.write("If your video has artifacts and has its frames shuffled for a reason or another, you can fix it here")


model = st.selectbox(
        "Choose your model",
        ("mobilenet_v3_small", "resnet18")
        )
upload = st.file_uploader("Drop your video here")

if upload is not None:
    # Save the uploaded bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(upload.getvalue())
        temp_video_path = temp_video.name
    
    try :
        with st.spinner("This might take around a minute...", show_time=True):
            restoration.VideoFixer(temp_video_path, models[model]).fix_video()
        st.video("fixed_video.mp4")
    except:
        st.error("we encountered an error :/")
        

