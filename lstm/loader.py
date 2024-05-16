import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
# class SurgeryDataset(Dataset):
#     def __init__(self, root_dir, transform=None, interval = False):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.interval = interval
#         self.video_files = [f for f in os.listdir(root_dir) if f.endswith(".mp4")]

#     def __len__(self):
#         return len(self.video_files)

#     def __getitem__(self, index):
#         video_path = os.path.join(self.root_dir, self.video_files[index])
#         skill_level = self.extract_skill_level(video_path)

#         if not self.interval:
#             frames = self.load_frames(video_path)
#         else:
#             frames = self.load_frames_interval(video_path)
#         if self.transform:
#             frames = self.transform(frames)

#         return frames, skill_level

#     def load_frames(self, video_path, num_frames=32, target_size=(224, 224)):
#         cap = cv2.VideoCapture(video_path)
#         frames = []

#         while len(frames) < num_frames:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = cv2.resize(frame, target_size)

#             frames.append(frame)

#         cap.release()
#         return np.array(frames)
    
#     def load_frames_interval(self, video_path, num_frames=32, frame_interval=4, target_size=(224, 224)):
#         cap = cv2.VideoCapture(video_path)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         if total_frames < num_frames * frame_interval:
#             frame_indices = np.arange(0, total_frames, frame_interval)
#         else:
#             frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
#         frames = []
#         for index in frame_indices:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, index)
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = cv2.resize(frame, target_size)
#             frames.append(frame)
        
#         cap.release()
        
#         if len(frames) < num_frames:
#             padding_frames = np.zeros((num_frames - len(frames), *target_size, 3), dtype=np.uint8)
#             frames = np.concatenate((frames, padding_frames), axis=0)
        
#         return np.array(frames)
    
#     def extract_skill_level(self, video_path):
#         file_name = os.path.basename(video_path)
#         skill_level = file_name.split("_")[-1].split(".")[0]
#         skill_level = int(skill_level) -1
#         return skill_level


from PIL import Image

class SurgeryDataset(Dataset):
    def __init__(self, root_dir, transform=None, interval=False):
        self.root_dir = root_dir
        self.transform = transform
        self.interval = interval
        self.video_files = [f for f in os.listdir(root_dir) if f.endswith(".mp4")]
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, index):
        video_path = os.path.join(self.root_dir, self.video_files[index])
        skill_level = self.extract_skill_level(video_path)
        
        if not self.interval:
            frames = self.load_frames(video_path)
        else:
            frames = self.load_frames_interval(video_path)
        
        if self.transform:
            frames = [self.transform(Image.fromarray(frame)) for frame in frames]
        
        frames = torch.stack(frames)
        
        return frames, skill_level
    
    def load_frames(self, video_path, num_frames=32, target_size=(224, 224)):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
        
        cap.release()
        return frames
    
    def load_frames_interval(self, video_path, num_frames=32, frame_interval=4, target_size=(224, 224)):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < num_frames * frame_interval:
            frame_indices = np.arange(0, total_frames, frame_interval)
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) < num_frames:
            padding_frames = np.zeros((num_frames - len(frames), *target_size, 3), dtype=np.uint8)
            frames.extend(padding_frames)
        
        return frames
    
    def extract_skill_level(self, video_path):
        file_name = os.path.basename(video_path)
        skill_level = file_name.split("_")[-1].split(".")[0]
        skill_level = int(skill_level) - 1
        return skill_level