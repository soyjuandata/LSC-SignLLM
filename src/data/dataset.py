import os
import torch
from torch.utils.data import Dataset
from glob import glob

class SignLanguageDataset(Dataset):
    """
    Dataset PyTorch para LSC-SignLLM.
    Lee los tensores de landmarks preprocesados y extrae la etiqueta de clase (glosa) del nombre de archivo.
    """
    def __init__(self, data_dir="data/processed/landmarks", max_frames=300):
        super().__init__()
        self.data_dir = os.path.abspath(data_dir)
        self.max_frames = max_frames
        
        # Buscar todos los tensores .pt en el directorio y subdirectorios
        self.file_paths = glob(os.path.join(self.data_dir, "**/*.pt"), recursive=True)
        
        if len(self.file_paths) == 0:
            print(f"[-] Warning: No se encontraron tensores en {self.data_dir}")
            
    def __len__(self):
        return len(self.file_paths)

    def _pad_tensor(self, tensor, max_len):
        """Pad o trunca el tensor a max_len frames."""
        T = tensor.shape[0]
        if T >= max_len:
            return tensor[:max_len]
        else:
            # Pad con ceros al final
            pad_size = list(tensor.shape)
            pad_size[0] = max_len - T
            padding = torch.zeros(*pad_size, dtype=tensor.dtype)
            return torch.cat([tensor, padding], dim=0)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # Extraer la glosa (ej: path/Volunteer_01/HOLA.pt -> HOLA)
        # Asumiendo estructura de directorios del script extract_landmarks.py
        basename = os.path.basename(file_path)
        gloss = os.path.splitext(basename)[0] # "HOLA"
        
        # Cargar los tensores de MediaPipe
        streams = torch.load(file_path, weights_only=True)
        
        # Asegurar longitud uniforme (Padding/Truncating)
        padded_streams = {
            "left_hand": self._pad_tensor(streams["left_hand"], self.max_frames),
            "right_hand": self._pad_tensor(streams["right_hand"], self.max_frames),
            "face": self._pad_tensor(streams["face"], self.max_frames),
            "pose": self._pad_tensor(streams["pose"], self.max_frames)
        }
        
        return padded_streams, gloss

if __name__ == "__main__":
    # Prueba del Dataset
    dataset = SignLanguageDataset()
    print(f"Total de muestras: {len(dataset)}")
    if len(dataset) > 0:
        streams, gloss = dataset[0]
        print(f"Muestra de prueba - Glosa: {gloss}")
        print(f"Shape left_hand: {streams['left_hand'].shape}")
