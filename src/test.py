import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Importujemy architekturę z Twojego pliku
from cnn import RPS_CNN

def run_live_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Odpalam sprzęt do akcji: {device}")

    # 1. Wczytujemy Zwycięski Model (Złote Parametry)
    model = RPS_CNN(kernel_size=7, num_filters=32, dropout_rate=0.2).to(device)
    
    try:
        # Ładujemy "mózg" z dysku
        model.load_state_dict(torch.load('./model/rps_model_best.pth', weights_only=True))
        print("Mózg sieci załadowany pomyślnie!")
    except FileNotFoundError:
        print("BŁĄD: Nie mogę znaleźć 'rps_model_best.pth'. Upewnij się, że jest w tym samym folderze.")
        return

    model.eval() # Przełączamy na tryb testowy (bez usypiania neuronów!)

    # 2. Transformacje wejściowe (Tylko czysty egzamin, żadnych szumów)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Kolejność klas - ImageFolder domyślnie układa je alfabetycznie
    class_names = ['paper', 'rock', 'scissors']

    # 3. Pobieranie zdjęć z Twojego folderu
    folder_path = './customVal'
    if not os.path.exists(folder_path):
        print(f"Nie znaleziono folderu {folder_path}!")
        return

    # Wyciągamy pliki obrazów (max 16, pod siatkę 4x4)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:16]
    
    if not image_files:
        print("Folder customVal jest pusty lub nie ma w nim zdjęć jpg/png!")
        return

    # 4. Rysowanie wyników
    plt.figure(figsize=(16, 12)) # Powiększone okno, żeby zmieścić 16 zdjęć
    
    for idx, filename in enumerate(image_files):
        img_path = os.path.join(folder_path, filename)
        
        # Otwieramy zdjęcie
        image = Image.open(img_path).convert('RGB')
        
        # Przepuszczamy przez filtry (Resize, Tensor, Normalize)
        img_tensor = transform(image)
        
        # PyTorch oczekuje paczki (batch), więc dodajemy sztuczny wymiar: [1, 3, 128, 128]
        img_batch = img_tensor.unsqueeze(0).to(device)
        
        # Pytamy model o zdanie!
        with torch.no_grad():
            output = model(img_batch)
            
            # Zamieniamy surowe liczby na procenty (Softmax)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            
            # Wybieramy indeks z najwyższym procentem
            predicted_idx = torch.argmax(probabilities).item()
            predicted_class = class_names[predicted_idx]
            confidence = probabilities[predicted_idx].item() * 100

        # Dodajemy zdjęcie do siatki Matplotlib (matryca 4x4)
        plt.subplot(4, 4, idx + 1)
        plt.imshow(image)
        plt.title(f"{predicted_class.upper()}\nPewność: {confidence:.1f}%", 
                  color='green' if confidence > 80 else 'red', 
                  fontweight='bold')
        plt.axis('off')

    plt.suptitle("Test modelu w akcji - Zbiór customVal (16 zdjęć)", fontsize=22, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_live_test()