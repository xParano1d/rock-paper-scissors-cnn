import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

from train import train_and_evaluate

def run_grid_search():
    # Upewniamy się, że folder na wykresy istnieje, żeby skrypt się nie wywalił na zapisie
    if not os.path.exists("./figure"):
        os.makedirs("./figure")

    # --- LABORATORIUM ---
    # ZMIENIAMY TYLKO 2 PIERWSZE ZMIENNE! Reszta zostaje po 1 opcji.
    param_grid = {
        'batch_size': [64, 128, 256],     # OŚ X (Wielkość paczki ładownana do VRAM)
        'lr': [0.005, 0.001, 0.0005],     # OŚ Y (Krok uczenia)
        'kernel_size': [7],               # ZABLOKOWANE (Nasz Mistrz)
        'num_filters': [32],              # ZABLOKOWANE (Nasz Mistrz)
        'dropout_rate': [0.2]             # ZABLOKOWANE (Nasz Mistrz)
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    x_key, y_key = keys[0], keys[1]
    x_vals, y_vals = param_grid[x_key], param_grid[y_key]
    
    # Macierz wyników dla osi Z (wypełniona zerami)
    Z = np.zeros((len(y_vals), len(x_vals)))
    
    print(f"=== ODPALAM {len(combinations)} EKSPERYMENTÓW DLA: {x_key} vs {y_key} ===")
    
    for idx, params in enumerate(combinations):
        print(f"\n[Eksperyment {idx+1}/{len(combinations)}] Parametry: {params}")
        
        val_acc = train_and_evaluate(
            epochs=55, 
            plot_results=True, 
            **params
        )
        
        print(f"-> Wynik (Val Acc): {val_acc:.2f}%")
        
        # Zapis do macierzy na podstawie indeksów wartości
        x_idx = x_vals.index(params[x_key])
        y_idx = y_vals.index(params[y_key])
        Z[y_idx, x_idx] = val_acc

    # --- RYSOWANIE I ZAPIS WYKRESU 3D ---
    print("\nEksperymenty zakończone. Generowanie i zapisywanie wykresu...")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(x_vals, y_vals)
    surf = ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='k', alpha=0.8)
    
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_zlabel('Accuracy [%]')
    ax.set_title(f'Zależność Accuracy od {x_key} i {y_key}')
    fig.colorbar(surf, shrink=0.5, aspect=0.5)
    
    # Dynamiczna nazwa pliku na podstawie badanych parametrów
    plot_filename = f"{x_key}_vs_{y_key}.png"
    plt.savefig(f"./figure/{plot_filename}", dpi=300, bbox_inches="tight")
    
    print(f"Pomyślnie zapisano wykres jako: ./figure/{plot_filename}")

if __name__ == "__main__":
    run_grid_search()