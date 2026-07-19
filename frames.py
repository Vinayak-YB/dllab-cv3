import torch
from torchvision.utils import save_image
import os
from model import SequenceEncoderDecoder
from dataset import FramePredictionDataset
from torchvision import transforms

model_dir = 'models/lstm_1'
data_dir = 'physics-data-v3'

device = 'cuda'
model = SequenceEncoderDecoder.from_pretrained(model_dir, device=device)

transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])
val_dataset = FramePredictionDataset([os.path.join(data_dir, f'traj-{i}') for i in range(900, 910)], context=5, transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

target_frames = []
predicted_frames = []
for input_seq, target in val_loader:
    with torch.no_grad():
        pred = model(input_seq.to(device)).clamp(0, 1).cpu()
        predicted_frames.append(pred)
        target_frames.append(target.squeeze(1))

predicted_frames = torch.cat(predicted_frames, dim=0)
target_frames = torch.cat(target_frames, dim=0)

os.makedirs('videos', exist_ok=True)
for i in range(min(50, len(predicted_frames))):
    save_image(predicted_frames[i], f'videos/pred_{i:04d}.png')
    save_image(target_frames[i], f'videos/gt_{i:04d}.png')

print("Frames saved in 'videos' folder. Open them to compare!")