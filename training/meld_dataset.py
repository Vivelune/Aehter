from pathlib import Path
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import subprocess
from torch.utils.data import default_collate



def collate_fn(batch):
    # Remove failed samples (None)
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None

    return default_collate(batch)


class MELDDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        video_dir: str,
        max_text_len: int = 128,
    ):
        base_dir = Path(__file__).resolve().parent

        self.csv_path = (base_dir / csv_path).resolve()
        self.video_dir = (base_dir / video_dir).resolve()

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        if not self.video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {self.video_dir}")

        self.data = pd.read_csv(self.csv_path)
        self.max_text_len = max_text_len
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.emotion_map = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "joy": 3,
            "neutral": 4,
            "sadness": 5,
            "surprise": 6,
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
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            
            
        except Exception as e:
             raise ValueError(f"Audio Error: {str(e)}")
         
        audio_tensor = torch.zeros(1)
        if audio_path.exists():
            audio_path.unlink()

        return audio_tensor

    def __len__(self):
        return len(self.data)

    def _load_video_frames(self, video_path: Path) -> torch.Tensor:
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i", str(video_path),
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    str(audio_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            
            waveform, sample_rate = torchaudio.load(audio_path)
            
            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
                
                
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
            )(waveform)
                            
            
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
            
            if mel_spec.size(2) < 300:
                mel_spec = torch.nn.functional.pad(mel_spec, (0, 300 - mel_spec.size(2)))
            else:
                mel_spec = mel_spec[: , : , :300]
                
            return mel_spec.squeeze(0)
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed for {video_path}") from e

        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Install it with: brew install ffmpeg")

        finally:
        # 6️⃣ Cleanup
            if audio_path.exists():
                audio_path.unlink()
    
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        if isinstance(idx, torch.Tensor):
            idx=idx.item()
        row = self.data.iloc[idx]
        try:
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

            video_frames = self._load_video_frames(video_path)
            audio_features = self.__extract__audio__features(video_path)
            
            
            # Map Sentiment
            

            emotion_label = self.emotion_map.get(row["Emotion"].lower(), )
            sentiment_label = self.sentiment_map.get(row["Sentiment"].lower(), )

            return {
                    "text_inputs":{
                    "input_ids": text_inputs["input_ids"].squeeze(0),
                    "attention_mask": text_inputs["attention_mask"].squeeze(0),
                                    },
                "video": video_frames,
                'audio_features': audio_features,
                "emotion_label": torch.tensor(emotion_label, dtype=torch.long),
                "sentiment_label": torch.tensor(sentiment_label, dtype=torch.long),
            }
        except Exception as e:
            print(f"Error processing {video_path} : {str(e)}")
            return None


def collate_fn(batch):
    # Remove failed samples (None)
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None

    return default_collate(batch)





def prepare_dataloaders(train_csv, train_video_dir,dev_csv, dev_video_dir, test_csv, test_video_dir, batch_size=32):
    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)
    
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,   # keep 0 for now
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    
    
    return train_loader,dev_loader, test_loader
    
    
    
    
    
    

if __name__ == "__main__":
    train_loader, dev_loader, test_loader = prepare_dataloaders(
        "../meld-dataset/MELD-RAW/MELD.Raw/train/train_sent_emo.csv",
        "../meld-dataset/MELD-RAW/MELD.Raw/train/train_splits",
        "../meld-dataset/MELD-RAW/MELD.Raw/dev/dev_sent_emo.csv",
        "../meld-dataset/MELD-RAW/MELD.Raw/dev/dev_splits_complete",
         "../meld-dataset/MELD-RAW/MELD.Raw/test/test_sent_emo.csv",
        "../meld-dataset/MELD-RAW/MELD.Raw/test/output_repeated_splits_test",
    )
    
    for batch in train_loader:
        print(batch['text_inputs'])
        print(batch['video_frames'].shape)
        print(batch['audio_features'].shape)
        print(batch['emotion_label'])
        print(batch['sentiment_label'])
        break
