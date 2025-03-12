import os
import json
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import timm
import torchvision.models as models

torch.backends.cudnn.benchmark = True

torch.manual_seed(42)
np.random.seed(42)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class FoodVolumeDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.volume_data = pd.read_csv(csv_file).apply(lambda x: x.astype(str).str.lower())
        self.transform = transform
        self.image_files = []
        self.root_dir = root_dir
        for group in os.listdir(root_dir):
            group_path = os.path.join(root_dir, group)
            for scene in os.listdir(group_path):
                scene_path = os.path.join(group_path, scene)
                transform_json = os.path.join(scene_path, 'transforms.json')
                if not os.path.exists(transform_json):
                    continue
                with open(transform_json, 'r') as f:
                    transform_matrices = {frame['file_path']: torch.tensor(frame['transform_matrix'][:3], dtype=torch.float32).flatten() for frame in json.load(f)['frames']}
                volume = self.volume_data.loc[(self.volume_data['Object_name'] == group.lower()) & (self.volume_data['Food_Type'] == scene.lower()), 'Volume'].values
                if volume.size > 0:
                    depth_path = os.path.join(scene_path, 'depths')
                    for file in os.listdir(depth_path):
                        if os.path.isfile(os.path.join(scene_path, 'masks', file)) and file in transform_matrices:
                            self.image_files.append((group, scene, file, float(volume[0]), transform_matrices[file]))

    def __getitem__(self, idx):
        group, scene, image_file, volume, trans_matrix = self.image_files[idx]
        def load_image(path, mode):
            return self.transform(Image.open(path).convert(mode)) if self.transform else T.ToTensor()(Image.open(path).convert(mode))

        rgb = load_image(os.path.join(self.root_dir, group, scene, 'images', image_file), 'RGB')
        depth = load_image(os.path.join(self.root_dir, group, scene, 'depths', image_file), 'RGB')
        #mask = (load_image(os.path.join(self.root_dir, group, scene, 'masks', image_file), 'RGB') > 0.5).float()

        return rgb, depth, torch.tensor(volume, dtype=torch.float32), trans_matrix

    def __len__(self):
        return len(self.image_files)


class VolumeEstimator(nn.Module):
    def __init__(self):
        super(VolumeEstimator, self).__init__()
        self.rgb_encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        #self.rgb_encoder = timm.create_model('inception_v4', pretrained=True)

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        self.rgb_proj = nn.Linear(1000, 256)
        #self.depth_proj = nn.Linear(768, 256)
        self.transform_proj = nn.Linear(256, 256)

        self.transform_encoder = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        self.cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, rgb, depth, transform_matrix):
        rgb_features = self.rgb_proj(self.rgb_encoder(rgb))
        depth_features = self.depth_encoder(depth)

        std = transform_matrix.std()
        if std > 1e-6:
            transform_matrix = (transform_matrix - transform_matrix.mean()) / std

        transform_features = self.transform_proj(self.transform_encoder(transform_matrix))

        rgb_features = rgb_features.unsqueeze(1)
        depth_features = depth_features.unsqueeze(1)
        transform_features = transform_features.unsqueeze(1)

        combined_features = torch.cat([rgb_features, transform_features], dim=1)
        attn_output, _ = self.cross_attention(depth_features, combined_features, combined_features)

        volume = self.regressor(attn_output.squeeze(1))
        return volume


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FoodVolumeDataset(args.data_path, args.csv_file, T.Compose([
        T.Resize((360, 480)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16, persistent_workers=True)

    model = VolumeEstimator().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_loss = float('inf')
    start_epoch = 0
    if args.resume and os.path.exists('best_model.pth'):
        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        training_loss = 0.0

        for rgb, depth, volume, transform in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            rgb, depth, volume, transform = rgb.to(device), depth.to(device), volume.to(device), transform.to(device)
            volume = volume.unsqueeze(1)

            optimizer.zero_grad()
            output = model(rgb, depth, transform)

            loss = criterion(output, volume)

            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        # evaluate the model
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for rgb, depth, volume, transform in tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
                rgb, depth, volume, transform = rgb.to(device), depth.to(device), volume.to(device), transform.to(device)
                output = model(rgb, depth, transform)
                test_loss += criterion(output, volume.unsqueeze(1)).item()

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'best_model.pth')
            print(f"Saved new best model at epoch {epoch+1} with test loss {test_loss:.4f}")

        scheduler.step()
        print(f"Epoch [{epoch+1}/{args.epochs}], Training Loss: {training_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.12f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file with volume labels')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')

    args = parser.parse_args()
