import os
import random
import subprocess
import tempfile
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


# =====================================================================
# Compressione JPEG AI (Fase 1 - preparazione dataset)
# Wrapper attorno al JPEG AI reference software (tool esterno).
# Pipeline per immagine: originale -> encoder (target bpp) -> bitstream
#                                  -> decoder -> PNG ricostruita.
# Le PNG ricostruite vengono salvate in <classe>_bpp<X>, gli stessi nomi
# che DeepfakeJPEGAIDataset cerca con suffix = f"bpp{target_bpp}".
# =====================================================================

VALID_IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp')


def _list_images(root, recursive=True):
    """Elenca tutte le immagini sotto root (ricorsivo: gestisce sottocartelle per
    video/metodo tipiche dei dataset FF++)."""
    out = []
    if recursive:
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith(VALID_IMG_EXT):
                    out.append(os.path.join(dirpath, f))
    else:
        out = [os.path.join(root, f) for f in os.listdir(root)
               if f.lower().endswith(VALID_IMG_EXT)]
    return sorted(out)


def _save_rgb_png(src_path, out_path):
    """Apre un'immagine e la ri-salva come PNG RGB. Ritorna True se ok."""
    try:
        Image.open(src_path).convert("RGB").save(out_path)
        return True
    except Exception as e:
        print(f"  ⚠️ skip {src_path}: {e}")
        return False


def build_original_split(real_src, fake_src, dataset_path,
                         n_per_class=None, seed=42, recursive=True, paired=True):
    """
    Prepara real_original/ e fake_original/ a partire da due cartelle sorgente
    qualsiasi (es. quelle del dataset FF++ scaricato da Kaggle).

    - Cerca le immagini anche nelle sottocartelle (recursive=True).
    - paired=True: stessa selezione posizionale per real e fake, cosi' real_i e
      fake_i sono lo STESSO frame sorgente (sfondo identico, volto diverso). Salva
      la coppia solo se entrambe le immagini sono valide, per non rompere
      l'accoppiamento degli indici. Richiede che le due cartelle abbiano lo stesso
      ordinamento (tipico di FF++ Original vs Deepfakes).
      paired=False: campionamento indipendente per classe (nessuna corrispondenza).
    - Sottocampiona n_per_class immagini per classe (None = tutte) con seed fisso.
    - Ri-salva TUTTO come PNG RGB: entrambe le classi condividono lo stesso formato
      (neutralizza il confound 'formato/container' prima della compressione JPEG AI).
    """
    real = _list_images(real_src, recursive=recursive)
    fake = _list_images(fake_src, recursive=recursive)

    if paired:
        m = min(len(real), len(fake))
        idx = list(range(m))
        random.Random(seed).shuffle(idx)
        if n_per_class is not None:
            idx = idx[:n_per_class]
        real_out = os.path.join(dataset_path, "real_original")
        fake_out = os.path.join(dataset_path, "fake_original")
        os.makedirs(real_out, exist_ok=True)
        os.makedirs(fake_out, exist_ok=True)
        n = 0
        for k in idx:
            rp = os.path.join(real_out, f"real_{n:05d}.png")
            fp = os.path.join(fake_out, f"fake_{n:05d}.png")
            ok_r = _save_rgb_png(real[k], rp)
            ok_f = _save_rgb_png(fake[k], fp)
            if ok_r and ok_f:
                n += 1
            else:
                # rimuovi l'eventuale meta' salvata per mantenere la corrispondenza
                for pth in (rp, fp):
                    if os.path.exists(pth):
                        os.remove(pth)
        print(f"real_original/fake_original: {n} coppie -> {dataset_path}")
    else:
        rng = random.Random(seed)
        for cls, lst in [("real", real), ("fake", fake)]:
            rng.shuffle(lst)
            sel = lst if n_per_class is None else lst[:n_per_class]
            out_dir = os.path.join(dataset_path, f"{cls}_original")
            os.makedirs(out_dir, exist_ok=True)
            n = sum(_save_rgb_png(p, os.path.join(out_dir, f"{cls}_{i:05d}.png"))
                    for i, p in enumerate(sel))
            print(f"{cls}_original: {n} immagini -> {out_dir}")
    print("✅ Split 'original' pronto.")


def _jpegai_run(cmd, jpegai_dir, env_name="jpeg_ai_vm"):
    """Esegue un comando nell'env conda di JPEG AI, dentro la sua directory."""
    full = ["conda", "run", "-n", env_name] + cmd
    res = subprocess.run(full, cwd=jpegai_dir, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"Comando JPEG AI fallito: {' '.join(cmd)}\nSTDERR:\n{res.stderr[-2000:]}"
        )
    return res


def compress_image_jpegai(src_path, out_png_path, bpp, jpegai_dir,
                          profile="base", stream_dir="/tmp/jpegai_streams",
                          env_name="jpeg_ai_vm"):
    """
    Comprime una singola immagine a un dato bpp e salva la PNG ricostruita.

    src_path:    immagine di input (se non e' PNG viene convertita al volo,
                 l'encoder JPEG AI accetta solo PNG).
    out_png_path: dove salvare l'immagine ricostruita.
    bpp:         bit-per-pixel target (es. 0.12). --set_target_bpp vuole bpp*100.
    jpegai_dir:  cartella del JPEG AI reference software (i path cfg sono relativi).
    profile:     'simple', 'base' o 'high'.
    """
    os.makedirs(stream_dir, exist_ok=True)
    cfg = ["cfg/tools_off.json", f"cfg/profiles/{profile}.json"]

    # L'encoder vuole input PNG: converte se necessario
    if src_path.lower().endswith(".png"):
        png_in, is_tmp = src_path, False
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=stream_dir)
        tmp.close()
        Image.open(src_path).convert("RGB").save(tmp.name)
        png_in, is_tmp = tmp.name, True

    base = os.path.splitext(os.path.basename(src_path))[0]
    stream = os.path.join(stream_dir, f"{base}_bpp{bpp}.bin")
    try:
        _jpegai_run(
            ["python", "-m", "src.reco.coders.encoder", png_in, stream,
             "--set_target_bpp", str(int(round(bpp * 100))), "--cfg", *cfg],
            jpegai_dir, env_name,
        )
        _jpegai_run(
            ["python", "-m", "src.reco.coders.decoder", stream, out_png_path],
            jpegai_dir, env_name,
        )
    finally:
        if is_tmp and os.path.exists(png_in):
            os.remove(png_in)
        if os.path.exists(stream):
            os.remove(stream)


def compress_dataset_jpegai(dataset_path, bpp_list, jpegai_dir,
                            classes=("real", "fake"), profile="base",
                            env_name="jpeg_ai_vm"):
    """
    Comprime l'intero dataset a tutti i bpp richiesti.

    Per ogni classe legge <classe>_original/ e genera <classe>_bpp<X>/ con le
    immagini ricostruite. Riprende automaticamente (salta i file gia' fatti).

    dataset_path: cartella che contiene real_original/ e fake_original/.
    bpp_list:     lista di bpp (es. [0.12, 0.25, 0.50]). Devono combaciare con
                  il target_bpp passato poi a DeepfakeJPEGAIDataset.
    """
    for cls in classes:
        src_dir = os.path.join(dataset_path, f"{cls}_original")
        if not os.path.isdir(src_dir):
            print(f"⚠️  Manca {src_dir}, salto la classe '{cls}'.")
            continue
        images = [f for f in sorted(os.listdir(src_dir))
                  if f.lower().endswith(VALID_IMG_EXT)]
        print(f"=== Classe {cls}: {len(images)} immagini ===")

        for bpp in bpp_list:
            out_dir = os.path.join(dataset_path, f"{cls}_bpp{bpp}")
            os.makedirs(out_dir, exist_ok=True)
            print(f"  bpp={bpp} -> {out_dir}")
            for fname in images:
                src = os.path.join(src_dir, fname)
                out = os.path.join(out_dir, os.path.splitext(fname)[0] + ".png")
                if os.path.exists(out):
                    continue
                try:
                    compress_image_jpegai(src, out, bpp, jpegai_dir,
                                          profile=profile, env_name=env_name)
                except Exception as e:
                    print(f"    ❌ {fname}: {e}")
    print("✅ Compressione completata.")