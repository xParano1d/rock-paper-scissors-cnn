import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir='dataset', batch_size=32, img_size=128, train_split=0.8):
   
    # 1. TRANSFORMACJE TRENINGOWE (Kłody pod nogi)
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        
        transforms.RandomRotation(degrees=90),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.7, 1.3)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        
        
        transforms.ToTensor(),
        
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 2. TRANSFORMACJE WALIDACYJNE (Czysty egzamin)
    # Żadnego wymazywania i obracania! Tylko to, co niezbędne dla wejścia sieci.
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 3. Wczytujemy zbiór podwójnie (PyTorch załaduje to sprytnie, nie bój się o RAM)
    full_train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    full_val_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)

    # Obliczamy wielkości
    total_size = len(full_train_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size

    # 4. Losujemy wspólną listę indeksów (stały seed, żeby uniknąć wycieku danych)
    indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(179582)).tolist()
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    # 5. Tworzymy ostateczne zbiory
    # train_dataset dostaje transformacje treningowe
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_idx)
    # val_dataset dostaje transformacje walidacyjne
    val_dataset = torch.utils.data.Subset(full_val_dataset, val_idx)

    # 6. Dataloadery z workerami na maksa
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    return train_loader, val_loader, full_train_dataset.classes