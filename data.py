from data import DeepfakeJPEGAIDataset, base_transform
from torch.utils.data import DataLoader

# Inizializza il dataset usando il percorso che abbiamo appena creato
test_dataset = DeepfakeJPEGAIDataset(root_dir=DATASET_PATH, target_bpp=None, transform=base_transform)

# Crea il DataLoader (raggruppa le immagini a pacchetti di 4)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

# Estrai un singolo "batch" (pacchetto) per vedere se funziona
batch_imgs, batch_labels = next(iter(test_loader))

print(f"Immagini caricate con successo!")
print(f"Shape del tensore immagini (Batch, Canali, Altezza, Larghezza): {batch_imgs.shape}")
print(f"Etichette del batch (0=Real, 1=Fake): {batch_labels}")