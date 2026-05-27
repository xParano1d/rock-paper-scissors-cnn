import torch
import torch.nn as nn
import torch.optim as optim


from data_pipeline import get_data_loaders
from cnn import RPS_CNN


# podział dataseta na train i validation
train_loader, val_loader, classes = get_data_loaders(batch_size=16)

# model
model = RPS_CNN()


# CrossEntropyLoss - klasyfikator (Kamień vs Papier vs Nożyce)
criterion = nn.CrossEntropyLoss()

# lr=0.001 to Learning Rate
optimizer = optim.AdamW(model.parameters(), lr=0.001)

print("Setup Ready.")


# Ustawiamy model w tryb nauki
model.train()

# Wyciągamy TYLKO JEDNĄ paczkę zdjęć z taśmociągu (żeby przetestować mechanizm)
for images, labels in train_loader:

    # KROK 1: Uczeń zgaduje wynk
    # Podajemy 16 zdjęć do modelu, a on wypluwa 3 liczby (prawdopodobieństwa) dla każdego
    predictions = model(images)

    # KROK 2: Sędzia ocenia błąd
    # Porównujemy to co zgadł model (predictions) z prawdziwymi odpowiedziami z folderu (labels)
    loss = criterion(predictions, labels)

    # KROK 3: Wymazanie brudnopisu
    # PyTorch domyślnie dodaje do siebie stare błędy. Musimy je wyzerować przed korektą!
    optimizer.zero_grad()

    # KROK 4: Obliczenie winy (Wsteczna Propagacja)
    # Sieć oblicza skomplikowane pochodne i sprawdza, który filtr zawiódł najbardziej
    loss.backward()

    # KROK 5: Nauczyciel poprawia ucznia
    # Optymalizator fizycznie zmienia parametry (wagi) modelu na nieco lepsze
    optimizer.step()

    # Wypisujemy wynik i przerywamy pętlę po 1 paczce!
    print(f"Błąd (Loss) dla pierwszej paczki wynosi: {loss.item():.4f}")
    break