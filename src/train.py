import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

from data_pipeline import get_data_loaders
from cnn import RPS_CNN

def train_and_evaluate(
    batch_size=128, 
    lr=0.001, 
    epochs=20, 
    kernel_size=3, 
    num_filters=16, 
    dropout_rate=0.2,
    plot_results=False
):
    filename = f"Batch_{batch_size}_LR_{lr}_Kernel_{kernel_size}_Filters_{num_filters}_Dropout_{dropout_rate}"
    print("\n--- Start Treningu | "+filename.replace("_", " ")+" ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Tworzymy model przepuszczając parametry z funkcji
    model = RPS_CNN(
        kernel_size=kernel_size, 
        num_filters=num_filters, 
        dropout_rate=dropout_rate
    ).to(device)

    # 2. Ładujemy dane przepuszczając batch_size
    train_loader, val_loader, classes = get_data_loaders(batch_size=batch_size)

    # 3. Sędzia i Nauczyciel
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Krok schedulera dostosowany do liczby epok
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/1000)

    # --- LISTY NA DANE DO WYKRESÓW ---
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    best_val_acc = 0.0

    for epoch in range(epochs):
        
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
            
            _, predicted_class = torch.max(predictions, 1)
            total_train += labels.size(0)
            correct_train += (predicted_class == labels).sum().item()
            
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
                
                loss = criterion(predictions, labels)
                running_val_loss += loss.item()
                
                _, predicted_class = torch.max(predictions, 1)
                total_val += labels.size(0)
                correct_val += (predicted_class == labels).sum().item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        scheduler.step()

        # RAPORT DO KONSOLI
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoka [{epoch+1}/{epochs}]\t|  Train Acc: {train_acc:.2f}%  |  Val Acc: {val_acc:.2f}%  |  LR: {current_lr:.6f}")
        # --- ZAPISYWANIE NAJLEPSZEGO MODELU ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"🌟 UWAGA: Nowy rekord walidacji! Zapisuję mózg sieci do pliku ({best_val_acc:.2f}%)")
            torch.save(model.state_dict(), 'rps_model_best.pth')

    # --- FAZA 3: WIZUALIZACJA ---
    if plot_results:
        print("Rysowanie wykresów...")
        plt.figure(figsize=(14, 6))

        # Lewy wykres: Błąd (Loss)
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Trening (Loss)', color='red', linewidth=2)
        plt.plot(val_losses, label='Walidacja (Loss)', color='orange', linewidth=2)
        plt.title('Krzywa błędu (Loss) - Trening vs Walidacja')
        plt.xlabel('Epoka')
        plt.ylabel('Błąd (CrossEntropyLoss)')
        plt.grid(True, linestyle='--', alpha=0.7) # Delikatniejsza siatka w tle
        plt.legend()

        # Prawy wykres: Dokładność (Accuracy)
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Trening (Accuracy)', color='blue', linewidth=2)
        plt.plot(val_accuracies, label='Walidacja (Accuracy)', color='cyan', linewidth=2)
        plt.title('Skuteczność modelu (Accuracy) - Trening vs Walidacja')
        plt.xlabel('Epoka')
        plt.ylabel('Accuracy [%]')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.tight_layout()
        plt.show()
        plt.savefig(f"./figure/{filename}.png", dpi=300, bbox_inches="tight");

    # Na sam koniec funkcja musi "wypluć" ostateczny wynik walidacji, żebyśmy mogli go zapisać w tabeli
    return val_accuracies[-1]


# --- TARCZA OCHRONNA I TEST ---
# Ten kod odpali się TYLKO, gdy ręcznie uruchomisz train.py w terminalu.
# Jeśli eksperymenty.py zaimportują ten plik, ten blok zostanie zignorowany.
if __name__ == "__main__":
    print("Odpalam testowy trening:")
    # Odpalamy funkcję i każemy jej narysować wykres po 10 epokach
    train_and_evaluate(
        epochs=200,              # Zostawiamy 55, sprawdziło się super
        batch_size=64,          # Zwycięzca
        lr=0.0005,              # Zwycięzca
        kernel_size=7,          # Zwycięzca
        num_filters=32,         # Zwycięzca
        dropout_rate=0.2,       # Zwycięzca
        plot_results=True       # Chcemy zobaczyć ten piękny wykres na koniec
    )

    