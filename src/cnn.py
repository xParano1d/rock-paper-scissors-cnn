import torch
import torch.nn as nn

class RPS_CNN(nn.Module):
    def __init__(self, kernel_size=3, num_filters=16, hidden_neurons=64, dropout_rate=0.2):
        super(RPS_CNN, self).__init__()
        
        # --- Zmienne wyciągnięte do eksperymentów badawczych ---
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.hidden_neurons = hidden_neurons
        self.dropout_rate = dropout_rate
        
        #* WARSTWA KONWOLUCYJNA
        # padding='same' zachowuje rozmiar krawędzi obrazu (wymóg z projektu na 5.0)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.num_filters, kernel_size=self.kernel_size, padding='same')
        # Funkcja aktywacji SiLU (zamiast ReLU) - lepsze wyniki w widzeniu maszynowym
        self.silu1 = nn.SiLU() 
        # MaxPool zmniejsza rozdzielczość o połowę (ze 128x128 na 64x64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #* WARSTWA KONWOLUCYJNA 
        # (Podwajamy liczbę filtrów)
        self.conv2 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.num_filters * 2, kernel_size=self.kernel_size, padding='same')
        self.silu2 = nn.SiLU()
        # MaxPool znów zmniejsza rozdzielczość (z 64x64 na 32x32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #* KLASYFIKATOR (Warstwy w pełni połączone)
        # Obliczamy ile pikseli wyjdzie z ostatniego basenu (32 * 32 * liczba kanałów)
        flattened_size = 32 * 32 * (self.num_filters * 2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened_size, self.hidden_neurons)
        self.silu3 = nn.SiLU()
        # Dropout chroni przed uczeniem się obrazków "na blachę"
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.hidden_neurons, 3) # Zwraca wektor 3 prawdop. (kamień, papier, nożyce)

    def forward(self, x):
        # Przepływ danych przez sieć (tzw. Forward Pass)
        x = self.pool1(self.silu1(self.conv1(x)))
        x = self.pool2(self.silu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.silu3(self.fc1(x)))
        x = self.fc2(x)
        return x

#! --- BLOK TESTOWY ---
if __name__ == "__main__":
    # Budujemy pusty model
    model = RPS_CNN()
    print("Architektura sieci załadowana:")
    print(model)
    
    # Tworzymy fałszywy tensor udający jedną paczkę danych z Twojego data_pipeline
    dummy_input = torch.randn(16, 3, 128, 128)
    output = model(dummy_input)
    
    print(f"\nKształt wejściowy z taśmociągu: {dummy_input.shape}")
    print(f"Kształt wyjściowy z modelu: {output.shape} -> [rozmiar_paczki, liczba_klas]")