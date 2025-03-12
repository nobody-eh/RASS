import os
import json
import argparse
import torch
import torch.nn as nn
# import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
# from pose.convert_depth import compute_depth
import numpy as np
from tqdm import tqdm
import timm
# from functools import lru_cache

torch.backends.cudnn.benchmark = True

def find_transform_matrix(image_file, transform_matrices):
    for frame in transform_matrices:
        if image_file in frame['file_path']:
            return torch.tensor(frame['transform_matrix'][:3], dtype=torch.float32).flatten()
    return None

class FoodVolumeDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.volume_data = pd.read_csv(csv_file).apply(lambda x: x.astype(str).str.lower())
        self.transform = transform
        self.image_files = []

        self.cache = {}

        for group in os.listdir(root_dir):
            group_path = os.path.join(root_dir, group)
            for scene in os.listdir(group_path):
                scene_path = os.path.join(group_path, scene)
                transform_json = os.path.join(scene_path, 'transforms.json')
                if not os.path.exists(transform_json):
                    print(f"Transform json file not found={transform_json}. Skipping...")
                    continue

                with open(transform_json, 'r') as f:
                    transform_matrices = json.load(f)['frames']

                vol = self.volume_data.loc[(self.volume_data["Object_name"] == group.lower()) & (self.volume_data["Food_Type"] == scene.lower())]["Volume"].values
                if len(vol) == 0:
                    print("skip=", group, scene, vol)
                    continue

                depth_path = os.path.join(scene_path, 'depths')
                for file in os.listdir(depth_path):
                    if not os.path.isfile(os.path.join(scene_path, 'masks', file)):
                        continue
                    tm = find_transform_matrix(file, tuple(transform_matrices) if isinstance(transform_matrices, list) else transform_matrices)
                    if tm is None:
                        continue
                    self.image_files.append((group, scene, file, float(vol[0]), tm))
        print(f"Dataset size: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def load_image(self, path, mode='RGB'):
        return Image.open(path).convert(mode)

    # @lru_cache(maxsize=1000)
    def load_depth_masked(self, depth_path):
        return np.load(depth_path).astype(np.float32)
        # For faster training time, we do pre-compute for a masked
        # return compute_depth(depth_path, mask_path, False, 5, (0, 3))

    def __getitem__(self, idx):
        group, scene, image_file, volume, trans_matrix = self.image_files[idx]
        volume = torch.tensor(volume, dtype=torch.float32)

        # depth_path = os.path.join(self.root_dir, group, scene, 'depths_compute', image_file.split('.')[0] + '.npy')
        depth_path = os.path.join(self.root_dir, group, scene, 'depths', image_file)
        mask_path = os.path.join(self.root_dir, group, scene, 'masks', image_file)
        rgb_path = os.path.join(self.root_dir, group, scene, 'images', image_file)

        # masked_depth = self.load_depth_masked(depth_path)

        # masked_depth_img = Image.fromarray(masked_depth).convert('RGB')
        depth_img = self.load_image(depth_path, 'RGB')
        mask_img = self.load_image(mask_path, 'RGB')
        rgb_img = self.load_image(rgb_path, 'RGB')

        if self.transform:
            depth_img = self.transform(depth_img)
            mask_img = self.transform(mask_img)
            rgb_img = self.transform(rgb_img)

        mask_img = (mask_img > 0.5).float()
        masked_rgb = rgb_img * mask_img
        masked_depth = depth_img * mask_img

        return masked_rgb, masked_depth, volume, trans_matrix

class VolumeEstimator(nn.Module):
    def __init__(self):
        super(VolumeEstimator, self).__init__()
        # self.resnet = models.resnet34(pretrained=True)
        # self.resnet.fc = nn.Linear(512, 256)

        # EfficientNet backbone (you can choose the variant based on your speed/accuracy trade-off)
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
        self.efficientnet.classifier = nn.Linear(self.efficientnet.classifier.in_features, 256)

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

        self.transform_encoder = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True), num_layers=6
        )
        self.regressor = nn.Linear(256, 1)

    def forward(self, rgb, depth, transform_matrix):
        rgb_features = self.efficientnet(rgb)
        depth_features = self.depth_encoder(depth)

        std = transform_matrix.std()
        if std > 1e-6:
            transform_matrix = (transform_matrix - transform_matrix.mean()) / std

        transform_features = self.transform_encoder(transform_matrix)

        combined_features = torch.mul(rgb_features, depth_features)
        combined_features = torch.mul(combined_features, transform_features)
        # combined_features = torch.stack([rgb_features, depth_features, transform_features], dim=0)
        transformer_out = self.transformer(combined_features)
        volume = self.regressor(transformer_out.squeeze(0))
        return volume

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.Compose([
        T.Resize((360, 480)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = FoodVolumeDataset(root_dir=args.data_path, csv_file=args.csv_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16, persistent_workers=True)


    train_size = int(args.train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8)

    model = VolumeEstimator().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # resume the model if the args.resume is available
    checkpoint_path = 'best_model.pth'
    start_epoch = 0
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed training from epoch {start_epoch}")

    best_loss = float('inf')
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
        print(f"Epoch [{epoch+1}/{args.epochs}], Training Loss: {training_loss/len(dataloader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file with volume labels')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--train_size', type=float, default=0.8, help='Training size')
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")

    args = parser.parse_args()
    train(args)