import itertools
import os
import csv
import torch
import gc

from train import train_and_evaluate

def run_experiments():
    if not os.path.exists("./figure/experiments"):
        os.makedirs("./figure/experiments")

    
    csv_file = "wyniki_eksperymentow.csv"
    
    # 1. Tworzymy nagłówek TYLKO RAZ, jeśli plik nie istnieje
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Tylko nazwy kolumn, żadnych 'params'!
            writer.writerow(['Eksperyment', 'Batch_Size', 'LR', 'Kernel', 'Filters', 'Dropout', 'Train_Accuracy', 'Val_Accuracy'])

    # =====================================================================
    # EKSPERYMENT 1: ARCHITEKTURA (Zależność Kernela i Filtrów) - 20 testów
    # =====================================================================
    print("\n" + "="*50)
    print("ROZPOCZYNAM EKSPERYMENT 1: ARCHITEKTURA")
    print("="*50)
    
    param_grid_1 = {
        'batch_size': [64],                 # ZAMROŻONE
        'lr': [0.001],                      # ZAMROŻONE
        'dropout_rate': [0.2],              # ZAMROŻONE
        'kernel_size': [3, 5, 7, 15],           # ZMIENNA 1 (X)
        'num_filters': [8, 16, 32, 64, 128]         # ZMIENNA 2 (Y)
    }
    
    keys1, values1 = zip(*param_grid_1.items())
    comb1 = [dict(zip(keys1, v)) for v in itertools.product(*values1)]
    
    for params in comb1:
        print(f"\n[Eksperyment 1] Parametry: {params}")
        # Odbieramy obie wartości: treningową i walidacyjną
        tr_acc, val_acc = train_and_evaluate(epochs=50, plot_results=False, save_model=False, **params)
        print(f"-> Wynik (Val): {val_acc:.2f}% | (Train): {tr_acc:.2f}%")
        
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Eksperyment_1', 
                params['batch_size'], 
                params['lr'], 
                params['kernel_size'], 
                params['num_filters'], 
                params['dropout_rate'], 
                round(tr_acc, 2),
                round(val_acc, 2)
            ])
        torch.cuda.empty_cache()
        gc.collect()

    # =====================================================================
    # EKSPERYMENT 2: DYNAMIKA UCZENIA (Batch vs LR) - 56 testów
    # =====================================================================
    print("\n" + "="*50)
    print("ROZPOCZYNAM EKSPERYMENT 2: DYNAMIKA UCZENIA")
    print("="*50)
    
    param_grid_2 = {
        'kernel_size': [7],                 # ZAMROŻONE (ustaw swój najlepszy po 1 teście)
        'num_filters': [32],                # ZAMROŻONE (ustaw swój najlepszy po 1 teście)
        'dropout_rate': [0.2],              # ZAMROŻONE
        'batch_size': [8, 16,],        # ZMIENNA 1 (X)           8, 16, 32, 64, 128, 256, 512
        'lr': [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001] # ZMIENNA 2 (Y)
    }
    
    keys2, values2 = zip(*param_grid_2.items())
    comb2 = [dict(zip(keys2, v)) for v in itertools.product(*values2)]
    
    for params in comb2:
        print(f"\n[Eksperyment 2] Parametry: {params}")
        tr_acc, val_acc = train_and_evaluate(epochs=50, plot_results=False, save_model=False, **params)
        print(f"-> Wynik (Val): {val_acc:.2f}% | (Train): {tr_acc:.2f}%")

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Eksperyment_2', 
                params['batch_size'], 
                params['lr'], 
                params['kernel_size'], 
                params['num_filters'], 
                params['dropout_rate'], 
                round(tr_acc, 2),
                round(val_acc, 2)
            ])
        torch.cuda.empty_cache()
        gc.collect()

    # # =====================================================================
    # # EKSPERYMENT 3: REGULARYZACJA (Wpływ Dropoutu) - 11 testów
    # # =====================================================================
    # print("\n" + "="*50)
    # print("ROZPOCZYNAM EKSPERYMENT 3: REGULARYZACJA (DROPOUT)")
    # print("="*50)
    
    # dropouts = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # for d in dropouts:
    #     params = {'batch_size': 64, 'lr': 0.0005, 'kernel_size': 7, 'num_filters': 32, 'dropout_rate': d}
    #     print(f"\n[Eksperyment 3] Dropout: {d}")
    #     tr_acc, val_acc = train_and_evaluate(epochs=50, plot_results=False, save_model=False, **params)
    #     print(f"-> Wynik (Val): {val_acc:.2f}% | (Train): {tr_acc:.2f}%")

    #     with open(csv_file, mode='a', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow([
    #             'Eksperyment_3', 
    #             params['batch_size'], 
    #             params['lr'], 
    #             params['kernel_size'], 
    #             params['num_filters'], 
    #             params['dropout_rate'], 
    #             round(tr_acc, 2),
    #             round(val_acc, 2)
    #         ])
    #     torch.cuda.empty_cache()
    #     gc.collect()

    print("\n[SUKCES] Maszyna skończyła. Wyniki są w wyniki_eksperymentow.csv")

if __name__ == "__main__":
    run_experiments()