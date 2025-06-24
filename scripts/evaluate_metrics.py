import os
import argparse
import torch
import torchvision.transforms as T
import numpy as np
from einops import rearrange
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import cv2

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.fvd import FrechetVideoDistance

def load_video_frames_from_folder(folder, image_exts={'.png', '.jpg', '.jpeg'}, max_frames=30, resize=(299, 299)):
    frames = []
    for fname in sorted(os.listdir(folder)):
        if Path(fname).suffix.lower() in image_exts:
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert("RGB")
            if resize:
                img = img.resize(resize)
            frame = T.ToTensor()(img)
            frames.append(frame)
            if len(frames) >= max_frames:
                break
    return torch.stack(frames)  # [T, C, H, W]

def load_video_from_mp4(video_path, max_frames=30, resize=(299, 299)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or len(frames) >= max_frames:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resize)
        frame = torch.from_numpy(frame).float() / 255.0  # [H, W, C]
        frame = frame.permute(2, 0, 1)  # [C, H, W]
        frames.append(frame)
    cap.release()
    return torch.stack(frames) if frames else None  # [T, C, H, W]

def load_all_videos(root_dir, is_video=True, max_frames=30, resize=(299, 299)):
    all_videos = []
    root_dir = Path(root_dir)
    print(f"Loading videos from: {root_dir}")
    for vid_dir in tqdm(sorted(root_dir.iterdir())):
        try:
            if is_video and vid_dir.suffix == '.mp4':
                video = load_video_from_mp4(str(vid_dir), max_frames=max_frames, resize=resize)
            elif not is_video and vid_dir.is_dir():
                video = load_video_frames_from_folder(str(vid_dir), max_frames=max_frames, resize=resize)
            else:
                continue

            if video is None or video.shape[0] < 2:
                continue

            video = rearrange(video, 't c h w -> c t h w')  # [C, T, H, W]
            all_videos.append(video)
        except Exception as e:
            print(f"Failed to load {vid_dir}: {e}")
    return torch.stack(all_videos)  # [B, C, T, H, W]

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load Data ===
    real_videos = load_all_videos(args.real_dir, is_video=args.real_is_video, max_frames=args.max_frames)
    fake_videos = load_all_videos(args.fake_dir, is_video=args.fake_is_video, max_frames=args.max_frames)

    print(f"Loaded {len(real_videos)} real and {len(fake_videos)} generated videos.")

    real_videos = real_videos.to(device)
    fake_videos = fake_videos.to(device)

    # === Calculate FVD ===
    if args.fvd:
        print("Calculating FVD...")
        fvd = FrechetVideoDistance(feature_extractor="i3d", reset_real_features=False).to(device)
        fvd.update(real_videos, real=True)
        fvd.update(fake_videos, real=False)
        fvd_score = fvd.compute().item()
        print(f"FVD: {fvd_score:.4f}")

    # === Calculate FID ===
    if args.fid:
        print("Calculating FID...")

        def extract_first_frame_batch(videos):
            return videos[:, :, 0]  # [B, C, H, W]

        real_imgs = extract_first_frame_batch(real_videos)
        fake_imgs = extract_first_frame_batch(fake_videos)

        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        fid.update(real_imgs, real=True)
        fid.update(fake_imgs, real=False)
        fid_score = fid.compute().item()
        print(f"FID (on first frames): {fid_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, required=True, help="Path to GT videos or image folders")
    parser.add_argument("--fake_dir", type=str, required=True, help="Path to generated videos or image folders")
    parser.add_argument("--real_is_video", action="store_true", help="If real_dir contains .mp4 files")
    parser.add_argument("--fake_is_video", action="store_true", help="If fake_dir contains .mp4 files")
    parser.add_argument("--max_frames", type=int, default=30, help="Max frames per video")
    parser.add_argument("--fid", action="store_true", help="Whether to compute FID")
    parser.add_argument("--fvd", action="store_true", help="Whether to compute FVD")
    args = parser.parse_args()
    main(args)