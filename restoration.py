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
    def __init__(self, video_name, model=mobilenet_v3_small):
        self.video_name = video_name
        self.model = model

    def get_frames(self):
        video = VideoFileClip(self.video_name)
        frames = [frame for frame in video.iter_frames()]
        return frames

    def get_features(self):
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
        #clusters = KMeans(n_clusters=2, random_state=42).fit_predict(features)
        clusters = DBSCAN(eps=40).fit_predict(features)  # epsilon value found with K-distance between cluster
        unique, counts = np.unique(clusters, return_counts=True)
        return clusters

    def remove_artifacts(self, frames, features, clusters):
        labels, counts = np.unique(clusters, return_counts=True)
        most_represented_label = labels[np.argmax(counts)]
        correct_frames = [frame for frame, label in zip(frames, clusters) if label==most_represented_label]
        correct_features = features[clusters == most_represented_label]
        return correct_frames, correct_features


    def arrange_frames(self, correct_frames, correct_features, first_frame=0):
        # create a matrix of distance between each points
        neigh = NearestNeighbors(n_neighbors=len(correct_frames), algorithm='ball_tree').fit(correct_features) # get matrix of distance between each points
        distances, idx = neigh.kneighbors(correct_features)

        # find incrementaly the closest frame to the last visited
        visited_frames = [first_frame]  # init first frame
        for i in range(len(correct_frames)-2):
            nearest_idx = 1
            neighbors_current_frame = idx[visited_frames[-1]]
            while neighbors_current_frame[nearest_idx] in visited_frames:
                nearest_idx += 1
            visited_frames.append(int(neighbors_current_frame[nearest_idx]))
        return [correct_frames[frame] for frame in visited_frames], correct_features[visited_frames]

    def find_first_and_last_frames(self, features):
        """
        After using arrange_frame the first time, we have 4 candidates to be the first or last frame of the original video.
        The four candidates are the first and last frames of the current sequences, and 2 frames that have a cut between them;
        We can eliminate 2 out of the 4 to only get the first and last frame.
        To do so, we have to identify the consecutive frames that have a cut (distance > threshold)
        """
        # Get the four candidates
        candidates = [0, len(features)-1]
        highest_distance = 0
        for i in range(len(features)-1):
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
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return np.mean(magnitude)

    def fix_video(self, output_file="fixed_video.mp4"):
        """
        Use this function to fix the video
        """
        frames = self.get_frames()
        features = self.get_features()
        clusters = self.find_clusters(features)
        real_frames, real_features = self.remove_artifacts(frames, features, clusters)
        frames_with_cut, features_with_cut = self.arrange_frames(real_frames, real_features)
        
        two_candidate_frames = self.find_first_and_last_frames(features_with_cut)
        frames_candidate_1, features_candidate_1 = self.arrange_frames(frames_with_cut, features_with_cut, first_frame=two_candidate_frames[0])
        frames_candidate_2, features_candidate_2 = self.arrange_frames(frames_with_cut, features_with_cut, first_frame=two_candidate_frames[1])

        # compute optical flow
        flow1 = self.optical_flow(frames_candidate_1[0], frames_candidate_1[len(frames_candidate_1)//2])
        flow2 = self.optical_flow(frames_candidate_2[0], frames_candidate_2[len(frames_candidate_2)//2])

        # select candidate for final video 
        video_candidates = [frames_candidate_1, frames_candidate_2]
        flow_candidates = [flow1, flow2]
        final_frames = [video_candidates[i] for i in range(len(video_candidates)) if flow_candidates[i] == np.max(flow_candidates)][0]
        
        final_video = ImageSequenceClip(final_frames, fps=24)
        final_video.write_videofile(output_file)
