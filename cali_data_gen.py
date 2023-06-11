
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torch.onnx
import numpy as np
from torch import nn
import io
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# model = torchvision.models.resnet18(pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
model.to(device)
model.eval()

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Load the file names and labels from a text file
        self.file_names, self.labels = [], []
        with open(f'{root_dir}/{split}.txt', 'r') as f:
            for line in f:
                file_name, label = line.strip().split()
                self.file_names.append(f'{root_dir}/{file_name}')
                self.labels.append(int(label))
        
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        # Load the image and label
        file_name = self.file_names[index]
        image = Image.open(file_name).convert('RGB')
        label = self.labels[index]

        # Apply the transformation, if any
        if self.transform is not None:
            image = self.transform(image)

        return image, label

# Create the ImageNet dataset with normalization
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
# dataset = torchvision.datasets.ImageNet(root='/data/wqzhao/Datasets/imagenet', split='val', transform=transform)
# Create the data loader
dataset = ImageFolder('/data/wqzhao/quantization/Dipoorlet/cali_path', transform=transform)
loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
# loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
# images, labels = next(loader)
# images, labels = images.to(device), labels.to(device)

# Evaluate the model on the dataset
# Evaluate the model on the dataset
total = 0
correct_top1 = 0
correct_top5 = 0
for i, (image, labels) in enumerate(dataset):
    image = image.numpy()
    image.tofile(os.path.join("/data/wqzhao/quantization/Dipoorlet/cali_path/input", "{}.bin".format(i)))
    # images, labels = images.to(device), labels.to(device)
    # outputs = model(images)
    # _, predicted = torch.topk(outputs, k=5)
    # total += labels.size(0)
    # correct_top1 += (predicted[:, 0] == labels).sum().item()
    # correct_top5 += (predicted == labels.view(-1, 1)).sum().item()

    # # Print log information every 20 iterations
    # if (i+1) % 20 == 0:
    #     accuracy_top1 = 100 * correct_top1 / total
    #     accuracy_top5 = 100 * correct_top5 / total
    #     print(f'Iteration [{i+1}/{len(loader)}]\tTop-1 Accuracy: {accuracy_top1:.2f}%\tTop-4 Accuracy: {accuracy_top5:.2f}%')

# # Print the final top-1 and top-4 accuracy
# accuracy_top1 = 100 * correct_top1 / total
# accuracy_top4 = 100 * correct_top5 / total
# print(f'Final Top-1 Accuracy on ImageNet: {accuracy_top1:.2f}%')
# print(f'Final Top-5 Accuracy on ImageNet: {accuracy_top5:.2f}%')
