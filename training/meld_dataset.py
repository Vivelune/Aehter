from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import subprocess


class MELDDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        video_dir: str,
        max_frames: int = 30,
        max_text_len: int = 128,
    ):
        # Anchor everything to THIS file's location
        base_dir = Path(__file__).resolve().parent

        self.csv_path = (base_dir / csv_path).resolve()
        self.video_dir = (base_dir / video_dir).resolve()

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        if not self.video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {self.video_dir}")

        self.data = pd.read_csv(self.csv_path)

        self.max_frames = max_frames
        self.max_text_len = max_text_len

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.emotion_map = {
            "anger": 0,
            "disgust": 1,
            "sadness": 2,
            "joy": 3,
            "neutral": 4,
            "surprise": 5,
            "fear": 6,
        }

        self.sentiment_map = {
            "negative": 0,
            "neutral": 1,
            "positive": 2,
        }
        
        
    def __extract__audio__features(self, video_path: Path) -> Path:
        video_path = Path(video_path)
        audio_path = video_path.with_suffix(".wav")

        
        try:
            subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-anodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
             raise ValueError(f"Audio Error: {str(e)}")

    def __len__(self):
        return len(self.data)

    def _load_video_frames(self, video_path: Path) -> torch.Tensor:
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            while len(frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (224, 224))
                frame = frame.astype(np.float32) / 255.0  # correct normalization
                frames.append(frame)

        finally:
            cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {video_path}")

        # Pad or truncate
        if len(frames) < self.max_frames:
            pad = [np.zeros_like(frames[0])] * (self.max_frames - len(frames))
            frames.extend(pad)
        else:
            frames = frames[: self.max_frames]

        # (T, H, W, C) → (T, C, H, W)
        frames = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2)
        

        return frames

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        video_name = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
        video_path = self.video_dir / video_name

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        text = row["Utterance"]

        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt", 
        )

        # video_frames = self._load_video_frames(video_path)
        self.__extract__audio__features(video_path)

        emotion_label = self.emotion_map.get(row["Emotion"].lower(), -1)
        sentiment_label = self.sentiment_map.get(row["Sentiment"].lower(), -1)

        return {
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "video": video_frames,
            "emotion": torch.tensor(emotion_label, dtype=torch.long),
            "sentiment": torch.tensor(sentiment_label, dtype=torch.long),
        }


if __name__ == "__main__":
    dataset = MELDDataset(
        csv_path="../meld-dataset/MELD-RAW/MELD.Raw/dev/dev_sent_emo.csv",
        video_dir="../meld-dataset/MELD-RAW/MELD.Raw/dev/dev_splits_complete",
    )
    
    sample = dataset[0]
    print(sample)
