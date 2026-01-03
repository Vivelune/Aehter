from pathlib import Path
import subprocess
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torchaudio
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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

    def _extract_audio(self, video_path: Path) -> torch.Tensor:
        audio_path = video_path.with_suffix(".wav")


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
        row = self.data.iloc[idx]

        video_name = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
        video_path = self.video_dir / video_name

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        text_inputs = self.tokenizer(
            row["Utterance"],
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        audio_features = self._extract_audio(video_path)

        return {
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "audio_path": str(audio_features),
        }


if __name__ == "__main__":
    dataset = MELDDataset(
        csv_path="../meld-dataset/MELD-RAW/MELD.Raw/dev/dev_sent_emo.csv",
        video_dir="../meld-dataset/MELD-RAW/MELD.Raw/dev/dev_splits_complete",
    )

    sample = dataset[0]
    print(sample)
