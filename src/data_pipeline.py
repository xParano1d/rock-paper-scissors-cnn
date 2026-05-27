import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

def get_data_loaders(data_dir='dataset', batch_size=32, img_size=128, train_split=0.8):
   
    #* IMAGE PROCESSING
    # Ujednolicamy rozmiar i zamieniamy na Tensory.
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # Normalizacja pikseli do przedziału [-1, 1] - sieci splotowe to uwielbiają
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


    #* Załadowanie całego datasetu z folderu głównego
    # ImageFolder automatycznie wykryje foldery 'paper', 'rock', 'scissors' jako klasy
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    #*Dynamiczny podział na Train i Validation
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size

    # Stały Seed, zawsze ten sam podział dataset'u
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(179582) 
    )

    #* Utworzenie DataLoaderów
    # shuffle=True dla treningu, żeby sieć nie uczyła się sekwencji na pamięć
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, full_dataset.classes


#! --- BLOK TESTOWY ---
if __name__ == "__main__":
    # parametry
    B_SIZE = 16
    IMG_SIZE = 128
    
    print("Ładowanie danych...")
    train_loader, val_loader, classes = get_data_loaders(
        data_dir='dataset', 
        batch_size=B_SIZE, 
        img_size=IMG_SIZE
    )
    
    print(f"Znalezione klasy: {classes}")
    print(f"Ilość paczek treningowych: {len(train_loader)} (po {B_SIZE} zdjęć)")
    print(f"Ilość paczek walidacyjnych: {len(val_loader)} (po {B_SIZE} zdjęć)")

    # test zgodnosci krztałtu pojedynczej paczki danych
    images, labels = next(iter(train_loader))
    print(f"\nKształt tensora obrazów: {images.shape}") 