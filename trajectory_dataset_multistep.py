import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

class TrajectoryImageDatasetMultistep(Dataset):
    def __init__(self, root_dir, split="train", num_steps=3, transform=None, 
                 max_scenarios=None, max_episodes_per_scenario=None):
        self.split_dir = os.path.join(root_dir, split)
        self.num_steps = num_steps
        self.transform = transform or transforms.ToTensor()
        self.max_episodes_per_scenario = max_episodes_per_scenario

        # Find scenario dirs (one per TFRecord)
        scenario_dirs = sorted([d for d in os.listdir(self.split_dir) 
                                if os.path.isdir(os.path.join(self.split_dir, d))])
        if max_scenarios is not None:
            scenario_dirs = scenario_dirs[:max_scenarios]

        self.sequences = []

        print(f"Scanning {len(scenario_dirs)} scenario folders in '{self.split_dir}'...")
        for scenario_name in tqdm(scenario_dirs, desc="Scenarios"):
            scenario_dir = os.path.join(self.split_dir, scenario_name)

            # Each episode is a subfolder (1, 2, 3, ...)
            episode_dirs = sorted([d for d in os.listdir(scenario_dir) 
                                   if os.path.isdir(os.path.join(scenario_dir, d))])
            if self.max_episodes_per_scenario is not None:
                episode_dirs = episode_dirs[:self.max_episodes_per_scenario]

            for episode_name in tqdm(episode_dirs, desc=f"Scenario {scenario_name}", leave=False):
                episode_dir = os.path.join(scenario_dir, episode_name)
                
                image_files = []
                with os.scandir(episode_dir) as it:
                    for entry in it:
                        if entry.is_file() and entry.name.startswith("win") and entry.name.endswith(".png"):
                            image_files.append(entry.path)
                image_files.sort()

                if len(image_files) >= num_steps + 1:
                    self.sequences.append(image_files)

        # Build valid (sequence_idx, start_frame_idx) pairs
        self.indices = []
        for seq_idx, images in enumerate(self.sequences):
            for i in range(len(images) - num_steps):
                self.indices.append((seq_idx, i))

        print(f"Total sequences: {len(self.sequences)}, total steps: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        seq_idx, start_idx = self.indices[idx]
        image_paths = self.sequences[seq_idx][start_idx:start_idx + self.num_steps + 1]

        tensors = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            tensor = self.transform(image)  # (3, H, W), float32 in [0.0, 1.0]
            tensors.append(tensor)

        return tensors  # [t0, t1, ..., tN]
