import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DeepfakeJPEGAIDataset(Dataset):
    def __init__(self, root_dir, target_bpp=None, transform=None):
        """
        root_dir: cartella base del dataset (deve contenere sottocartelle 'real' e 'fake')
        target_bpp: livello di compressione JPEG AI (es. 0.12, 0.25). Se None, carica le originali.
        """
        self.root_dir = root_dir
        self.target_bpp = target_bpp
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Gestione dei percorsi in base alla compressione
        suffix = "original" if target_bpp is None else f"bpp{target_bpp}"
        real_dir = os.path.join(root_dir, f"real_{suffix}")
        fake_dir = os.path.join(root_dir, f"fake_{suffix}")
        
        # Carica immagini Reali (Classe 0)
        if os.path.exists(real_dir):
            for img in os.listdir(real_dir):
                self.image_paths.append(os.path.join(real_dir, img))
                self.labels.append(0)
                
        # Carica immagini Fake (Classe 1)
        if os.path.exists(fake_dir):
            for img in os.listdir(fake_dir):
                self.image_paths.append(os.path.join(fake_dir, img))
                self.labels.append(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Trasformazione standard per le reti neurali in PyTorch
base_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])