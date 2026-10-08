import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Asegurar que src/ está en el path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.landmark_encoder import LandmarkEncoder # Nuestro LightSignMoEProjector
from src.data.dataset import SignLanguageDataset

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("[-] Warning: La librería 'transformers' no está instalada. Se usarán embeddings simulados para pruebas.")

class InfoNCELoss(nn.Module):
    """
    Contrastive Loss (InfoNCE) para alinear tokens visuales con embeddings de texto.
    Acerca las secuencias (v, t) emparejadas y aleja las demás en el batch.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, visual_embeds, text_embeds):
        # visual_embeds: [B, D] (Pooling de los T_condensado tokens)
        # text_embeds: [B, D]
        
        # Normalización L2 (convertimos a float32 para evitar mismatch y ganar precisión)
        v_norm = F.normalize(visual_embeds.to(torch.float32), dim=-1)
        t_norm = F.normalize(text_embeds.to(torch.float32), dim=-1)
        
        # Similitud Coseno (Logits)
        logits = torch.matmul(v_norm, t_norm.transpose(0, 1)) / self.temperature
        
        # Labels: la diagonal principal representa los pares correctos
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        # Pérdida simétrica cruzada
        loss_v = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.transpose(0, 1), labels)
        
        return (loss_v + loss_t) / 2.0

def train_epoch(model, dataloader, optimizer, criterion, tokenizer, text_model, device):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (streams, glosses) in enumerate(pbar):
        # Mover tensores a la GPU/CPU
        streams = {k: v.to(device) for k, v in streams.items()}
        
        # 1. Forward Pass Visual (Proyector MoE)
        # Salida: [B, T_condensado, d_llm]
        visual_tokens = model(streams)
        
        # Agrupamiento temporal temporal (Average Pooling) para la contrastiva
        # Convertimos [B, T_cond, D] a [B, D] para emparejarlo con 1 palabra
        visual_embeds = torch.mean(visual_tokens, dim=1) 
        
        # 2. Forward Pass Textual (Congelado)
        if HAS_TRANSFORMERS and tokenizer is not None and text_model is not None:
            # Tokenizar glosa
            inputs = tokenizer(glosses, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                # Obtener embeddings textuales (última capa oculta)
                outputs = text_model(**inputs, output_hidden_states=True)
                # Tomamos el embedding del último token no de padding
                text_embeds = outputs.hidden_states[-1][:, -1, :] 
        else:
            # Modo Simulación: Generar embeddings aleatorios fijos por glosa
            B, D = visual_embeds.shape
            text_embeds = torch.randn(B, D, device=device)
        
        # 3. Pérdida Contrastiva
        loss = criterion(visual_embeds, text_embeds)
        
        # 4. Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Usando dispositivo: {device}")
    
    # Parámetros
    batch_size = 4
    epochs = 10
    lr = 1e-4
    model_name = "google/gemma-4-E2B-it-qat-w4a16-ct" # Usando la configuración E2B de la familia Gemma 4
    
    # 1. Cargar LLM Congelado (Solo si transformers está disponible)
    tokenizer = None
    text_model = None
    if HAS_TRANSFORMERS:
        try:
            print(f"[*] Descargando/Cargando {model_name} de Hugging Face...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Para la función de pérdida InfoNCE solo necesitamos generar embeddings, no autoregresión
            text_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            text_model.eval() # Siempre congelado
            print("[+] Gemma cargado y congelado.")
        except Exception as e:
            print(f"[-] Error cargando {model_name}: {e}")
            print("    Asegúrate de estar logueado con 'huggingface-cli login'.")
            print("    Ejecutando en modo de simulación de embeddings...")
    
    # 2. Cargar Proyector MoE (Entrenable)
    print("[*] Instanciando LightSignMoEProjector...")
    # Asegurar que d_llm coincide con Gemma (2b tiene 2048, 7b tiene 3072, etc)
    # Por defecto usamos 2048 para gemma-2b
    if text_model:
        if hasattr(text_model.config, "text_config"):
            d_llm = text_model.config.text_config.hidden_size
        else:
            d_llm = text_model.config.hidden_size
    else:
        d_llm = 2048
    projector = LandmarkEncoder(d_llm=d_llm).to(device)
    
    # 3. Dataset y DataLoader
    dataset = SignLanguageDataset()
    if len(dataset) == 0:
        print("[-] Error: No hay datos para entrenar. Genera los tensores con extract_landmarks.py primero.")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 4. Optimizador y Loss
    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr)
    criterion = InfoNCELoss(temperature=0.07).to(device)
    
    # 5. Training Loop
    print("\n[*] Iniciando Fase 1: Entrenamiento de Alineación (Contrastive Loss)")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(projector, dataloader, optimizer, criterion, tokenizer, text_model, device)
        print(f"Epoch {epoch}/{epochs} | Avg Loss: {loss:.4f}")
        
        # Guardar Checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(projector.state_dict(), f"checkpoints/moe_projector_epoch_{epoch}.pth")
        
    print("[+] Entrenamiento Completado. Pesos guardados en checkpoints/")

if __name__ == "__main__":
    main()
