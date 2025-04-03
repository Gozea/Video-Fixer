from tqdm import tqdm
from moviepy import VideoFileClip, ImageSequenceClip
import cv2

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, resnet18


params = {
        "size": (224, 224),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
        }

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(params["size"]),
    transforms.Normalize(params["mean"], params["std"])
    ])


class VideoFixer():
    """
    A class to fix videos that have artifact issues or shuffled frames.
    It provides methods to extract frames, detect features, remove artifacts,
    cluster frames, arrange them in the correct order, and generate a fixed video.
    """

    def __init__(self, video_name, model=mobilenet_v3_small):
        """
        Initialize the VideoFixer class with the video input and model.

        Parameters:
        - video_name (str): Path to the input video file.
        - model (torch.nn.Module, optional): The model to use for feature extraction (default is 'mobilenet_v3_small').
        """
        self.video_name = video_name
        self.model = model

    def get_frames(self):
        """
        Extract all frames from the video input.

        Returns:
        - frames (list): A list of video frames.
        """
        video = VideoFileClip(self.video_name)
        frames = [frame for frame in video.iter_frames()]
        return frames

    def get_features(self):
        """
        Extract feature vectors from each frame of the video input using the model.

        Returns:
        - features (numpy.ndarray): A numpy array of extracted features from all frames.
        """
        frames = self.get_frames()

        model = self.model(weights="DEFAULT")
        model.eval()

        features = []
        for frame in tqdm(frames):
            x = transform(frame).unsqueeze(0)
            with torch.no_grad():
                feat = model(x)
                features.append(feat.flatten())
        features = torch.stack(features)
        features = features.numpy()
        return features

    def find_clusters(self, features):
        """
        Find clusters of similar frames based on extracted features.

        This method uses DBSCAN clustering to group frames with similar features
        based on their Euclidean distance.

        Parameters:
        - features (numpy.ndarray): A numpy array of frame features.

        Returns:
        - clusters (numpy.ndarray): A numpy array of cluster labels corresponding to each frame.
        """
        #clusters = KMeans(n_clusters=2, random_state=42).fit_predict(features)
        clusters = DBSCAN(eps=40).fit_predict(features)  # epsilon value found with K-distance between cluster
        unique, counts = np.unique(clusters, return_counts=True)
        return clusters

    def remove_artifacts(self, frames, features, clusters):
        """
        Remove frames that belong to outlier clusters, keeping the most frequent cluster.

        This method identifies the most common cluster and returns the frames and features
        that belong to this cluster, effectively removing outlier frames.

        Parameters:
        - frames (list): List of video frames.
        - features (numpy.ndarray): A numpy array of frame features.
        - clusters (numpy.ndarray): A numpy array of cluster labels.

        Returns:
        - correct_frames (list): A list of frames belonging to the most common cluster.
        - correct_features (numpy.ndarray): A numpy array of features for frames in the most common cluster.
        """
        labels, counts = np.unique(clusters, return_counts=True)
        most_represented_label = labels[np.argmax(counts)]
        correct_frames = [frame for frame, label in zip(frames, clusters) if label==most_represented_label]
        correct_features = features[clusters == most_represented_label]
        return correct_frames, correct_features


    def arrange_frames(self, correct_frames, correct_features, first_frame=0):
        """
        Arrange frames in a sequence based on the similarity of their features.

        This method builds a matrix of distances between frames and arranges them in the
        most logical sequence based on feature similarity, starting with a provided
        first frame.

        Parameters:
        - correct_frames (list): The list of frames that belong to the correct cluster.
        - correct_features (numpy.ndarray): Features of the correct frames.
        - first_frame (int, optional): The index of the first frame to start with (default is 0).

        Returns:
        - ordered_frames (list): A list of frames arranged in the correct order.
        - ordered_features (numpy.ndarray): Features of the frames arranged in the correct order.
        """
        # create a matrix of distance between each points
        neigh = NearestNeighbors(n_neighbors=len(correct_frames), algorithm='ball_tree').fit(correct_features) # get matrix of distance between each points
        distances, idx = neigh.kneighbors(correct_features)  # idx[:,1] is the closest neighbour

        # find incrementaly the closest frame to the last visited
        visited_frames = [first_frame]  # init first frame
        for i in range(len(correct_frames)-2):
            nearest_idx = 1
            neighbors_current_frame = idx[visited_frames[-1]]  # get neighbour of current frame
            while neighbors_current_frame[nearest_idx] in visited_frames:
                nearest_idx += 1  # if nearest neighbour already visited, we get the nearest+1
            visited_frames.append(int(neighbors_current_frame[nearest_idx]))
        return [correct_frames[frame] for frame in visited_frames], correct_features[visited_frames]

    def find_first_and_last_frames(self, features):
        """
        Find the first and last frames of the true video.

        This method identifies the four potential candidates for the first and last
        frames of the true video, which are the two ends of the rearranged sequence and two frames
        where there is a significant cut between them. The method eliminates the
        frames that are too close and returns the final first and last frames.

        Parameters:
        - features (numpy.ndarray): The features of the frames after arrangement.

        Returns:
        - candidates (list): A list of two indices, the first and last frames of the sequence.
        """
        # Get the four candidates
        candidates = [0, len(features)-1]
        highest_distance = 0
        for i in range(len(features)-1):
            # find the two consecutive frames with the highest distance
            current_distance = pairwise_distances(features[i].reshape(1, -1), features[i+1].reshape(1, -1), metric='euclidean')
            if current_distance > highest_distance:
                cut = [i, i+1]
                highest_distance = current_distance
        candidates += cut
        # We now have four candidates, the two with lowest distance should be consecutive and eliminated
        neigh = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(features[candidates])
        distance, idx = neigh.kneighbors(features[candidates])
        nearest_distance = distance[:,1]
        consecutive_frames = idx[nearest_distance==np.min(nearest_distance)][:,0]  # the consecutive are the one with closest distance
        candidates = [candidates[i] for i in range(len(candidates)) if i not in consecutive_frames]
        return candidates

    def optical_flow(self, frame1, frame2):
        """
        Compute the optical flow between two frames.

        This method calculates the optical flow between two frames to estimate the motion
        of objects between the frames.

        Parameters:
        - frame1 (ndarray): The first frame.
        - frame2 (ndarray): The second frame.

        Returns:
        - magnitude (float): The mean magnitude of the optical flow.
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return np.mean(magnitude)

    def fix_video(self, output_file="fixed_video.mp4"):
        """
        Fix the video by removing artifacts and reordering the frames.

        This method processes the video by extracting frames, obtaining their features,
        clustering them, removing artifacts, and rearranging them in the correct order.
        It then computes optical flow to select the best candidate for the final video
        and writes the fixed video to the output file.

        Parameters:
        - output_file (str, optional): The path to save the fixed video (default is 'fixed_video.mp4').

        Returns:
        - None: The method writes the fixed video directly to the specified output file.
        """
        frames = self.get_frames()
        features = self.get_features()
        clusters = self.find_clusters(features)
        real_frames, real_features = self.remove_artifacts(frames, features, clusters)
        frames_with_cut, features_with_cut = self.arrange_frames(real_frames, real_features)
        
        # we find the first and last frame of the true video
        two_candidate_frames = self.find_first_and_last_frames(features_with_cut)
        frames_candidate_2, features_candidate_2 = self.arrange_frames(frames_with_cut, features_with_cut, first_frame=two_candidate_frames[1])
        frames_candidate_1, features_candidate_1 = self.arrange_frames(frames_with_cut, features_with_cut, first_frame=two_candidate_frames[0])

        # compute optical flow
        flow1 = self.optical_flow(frames_candidate_1[0], frames_candidate_1[len(frames_candidate_1)//2])
        flow2 = self.optical_flow(frames_candidate_2[0], frames_candidate_2[len(frames_candidate_2)//2])

        # select candidate for final video 
        video_candidates = [frames_candidate_1, frames_candidate_2]
        flow_candidates = [flow1, flow2]
        final_frames = [video_candidates[i] for i in range(len(video_candidates)) if flow_candidates[i] == np.max(flow_candidates)][0]
        
        final_video = ImageSequenceClip(final_frames, fps=24)
        final_video.write_videofile(output_file)
