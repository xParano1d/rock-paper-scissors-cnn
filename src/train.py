import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt



from data_pipeline import get_data_loaders
from cnn import RPS_CNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")

# Tworząc model, od razu wysyłamy go na kartę graficzną
model = RPS_CNN().to(device)


# podział dataseta na train i validation
train_loader, val_loader, classes = get_data_loaders(batch_size=16)


# CrossEntropyLoss - klasyfikator (Kamień vs Papier vs Nożyce)
criterion = nn.CrossEntropyLoss()

# lr=0.001 to Learning Rate
optimizer = optim.AdamW(model.parameters(), lr=0.001)

print("Setup Ready.")


# --- LISTY NA DANE DO WYKRESÓW ---
EPOCHS = 20
train_losses = []
val_losses = []     # Nowa zmienna!
train_accuracies = [] # Nowa zmienna!
val_accuracies = []

for epoch in range(EPOCHS):
    
    # --- FAZA 1: TRENING ---
    model.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        predictions = model(images)
        loss = criterion(predictions, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item()
        
        # Zbieramy dane do Train Accuracy
        _, predicted_class = torch.max(predictions, 1)
        total_train += labels.size(0)
        correct_train += (predicted_class == labels).sum().item()
        
    # Obliczamy i zapisujemy wyniki dla Treningu (Z CAŁEJ EPOKI)
    avg_train_loss = running_train_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train
    
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)
    
    # --- FAZA 2: WALIDACJA (EGZAMIN) ---
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            
            # Zbieramy dane do Val Loss
            loss = criterion(predictions, labels)
            running_val_loss += loss.item()
            
            # Zbieramy dane do Val Accuracy
            _, predicted_class = torch.max(predictions, 1)
            total_val += labels.size(0)
            correct_val += (predicted_class == labels).sum().item()
            
    # Obliczamy i zapisujemy wyniki dla Walidacji (Z CAŁEJ EPOKI)
    avg_val_loss = running_val_loss / len(val_loader)
    val_acc = 100 * correct_val / total_val
    
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)
    
    # RAPORT DO KONSOLI
    print(f"Epoka [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")


# --- FAZA 3: WIZUALIZACJA (WYKRESY) ---
print("Rysowanie wykresów...")
plt.figure(figsize=(14, 6))

# Lewy wykres: Błąd (Loss)
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Trening (Loss)', color='red', marker='o')
plt.plot(val_losses, label='Walidacja (Loss)', color='orange', marker='s')
plt.title('Krzywa błędu (Loss) - Trening vs Walidacja')
plt.xlabel('Epoka')
plt.ylabel('Błąd (CrossEntropyLoss)')
plt.grid(True)
plt.legend()

# Prawy wykres: Dokładność (Accuracy)
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Trening (Accuracy)', color='blue', marker='o')
plt.plot(val_accuracies, label='Walidacja (Accuracy)', color='cyan', marker='s')
plt.title('Skuteczność modelu (Accuracy) - Trening vs Walidacja')
plt.xlabel('Epoka')
plt.ylabel('Accuracy [%]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()