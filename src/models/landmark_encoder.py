import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1dBlock(nn.Module):
    """
    Bloque Convolucional Causal 1D para extraer coherencia temporal local 
    sin sesgo de futuro (evitando look-ahead).
    """
    def __init__(self, d_model=512, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        # Padding causal manual: agregamos (kernel_size - 1) a la izquierda en el forward pass
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1
        )
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        x: [B, T, d_model]
        """
        # Cambiar dimensiones para PyTorch Conv1D: [B, d_model, T]
        x_t = x.transpose(1, 2)
        
        # Padding causal manual por la izquierda
        padded = F.pad(x_t, (self.kernel_size - 1, 0))
        
        # Convolución
        out = self.conv(padded)
        
        # Regresar a formato original [B, T, d_model]
        out = out.transpose(1, 2)
        
        # Conexión residual + normalización + activación
        return self.activation(self.norm(out + x))


class SlidingPerceiverResampler(nn.Module):
    """
    Sliding Perceiver Resampler: Aplica una ventana temporal móvil (w) y stride (s)
    para comprimir fotogramas continuos en tokens latentes densos inyectables en el LLM.
    """
    def __init__(self, d_model=512, d_llm=2560, num_heads=8, window_size=4, stride=2):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.d_llm = d_llm
        
        # Query Latente Entrenable (Z_queries): 1 token por ventana temporal
        # Mapea las características temporales locales a 1 único vector de entrada para el LLM
        self.z_query = nn.Parameter(torch.randn(1, 1, d_llm))
        
        # Capa de proyección para alinear d_model (512) con d_llm (2560)
        self.feat_proj = nn.Linear(d_model, d_llm)
        
        # Atención Cruzada Multi-Head
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_llm,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(d_llm)
        self.norm2 = nn.LayerNorm(d_llm)
        
        # FFN Feed-Forward Network simple para el Perceiver
        self.ffn = nn.Sequential(
            nn.Linear(d_llm, d_llm * 4),
            nn.GELU(),
            nn.Linear(d_llm * 4, d_llm)
        )

    def forward(self, x):
        """
        x: [B, T, d_model] - Características del Landmark Encoder
        Retorna: [B, T_condensado, d_llm] - Visual Tokens listos para el LLM
        """
        B, T, _ = x.shape
        
        # Proyectar características a la dimensión del LLM (2560)
        x_proj = self.feat_proj(x) # [B, T, d_llm]
        
        # Generar las ventanas temporales deslizantes (Sliding Window)
        windows = []
        for t in range(0, T - self.window_size + 1, self.stride):
            # Tomar sub-secuencia de la ventana: [B, window_size, d_llm]
            win_feat = x_proj[:, t:t+self.window_size, :]
            windows.append(win_feat)
            
        if not windows:
            # Salvaguarda para secuencias extremadamente cortas
            win_feat = F.pad(x_proj, (0, 0, 0, max(0, self.window_size - T)))
            windows.append(win_feat)
            
        # Apilar ventanas temporales a lo largo de una dimensión de lote extendido para paralelizar
        # Cada ventana se procesará independientemente por la atención cruzada
        num_windows = len(windows)
        
        # Concatenar a nivel de lote: [B * num_windows, window_size, d_llm]
        flat_windows = torch.cat(windows, dim=0) 
        
        # Expandir la query latente Z a nivel de lote extendido: [B * num_windows, 1, d_llm]
        queries = self.z_query.expand(flat_windows.shape[0], -1, -1)
        
        # Atención Cruzada: Query (Z) busca información en Keys/Values (flat_windows)
        # Q: [B_ext, 1, d_llm], K: [B_ext, w, d_llm], V: [B_ext, w, d_llm]
        attn_out, _ = self.cross_attn(
            query=queries,
            key=flat_windows,
            value=flat_windows
        )
        
        # Residuos y Normalización
        out = self.norm1(attn_out + queries)
        out = self.norm2(self.ffn(out) + out) # [B * num_windows, 1, d_llm]
        
        # Re-organizar a dimensiones originales: [B, num_windows, d_llm]
        condensed_tokens = out.view(B, num_windows, self.d_llm)
        
        return condensed_tokens


class VisibilityAwareRouter(nn.Module):
    def __init__(self, face_dim=1404, hands_dim=126, pose_dim=99):
        super().__init__()
        # +3 por los 3 valores de visibilidad (Face, Hands, Pose)
        total_dim = face_dim + hands_dim + pose_dim + 3 
        self.net = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 3) # Un peso para: Face, Hands, Pose
        )
        
    def forward(self, face_f, hands_f, pose_f, face_v, hands_v, pose_v):
        x = torch.cat([face_f, hands_f, pose_f, face_v, hands_v, pose_v], dim=-1)
        logits = self.net(x)
        weights = F.softmax(logits, dim=-1)
        return weights # [B, T, 3]

class ExpertStream(nn.Module):
    """
    Procesa un flujo de datos (Rostro, Manos o Pose). 
    Proyecta espacialmente y luego aplica convolución causal temporal.
    """
    def __init__(self, in_dim, proj_dim=512):
        super().__init__()
        self.spatial_proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )
        self.temporal_conv = CausalConv1dBlock(d_model=proj_dim, kernel_size=3)
        
    def forward(self, x):
        spatial_feat = self.spatial_proj(x)
        temp_feat = self.temporal_conv(spatial_feat)
        return temp_feat

class LightSignMoEProjector(nn.Module):
    """
    Arquitectura Mixture-of-Parts para LSC-SignLLM.
    Orquesta el enrutamiento dinámico, procesamiento experto y compresión temporal.
    """
    def __init__(self, d_model=512, d_llm=2560, window_size=4, stride=2):
        super().__init__()
        
        # Dimensiones basadas en MediaPipe (extract_landmarks.py)
        # Face: 468*3 = 1404, Hands: (21+21)*3 = 126, Pose: 33*3 = 99
        self.face_dim = 1404
        self.hands_dim = 126
        self.pose_dim = 99
        
        # Expertos
        self.expert_face = ExpertStream(self.face_dim, d_model)
        self.expert_hands = ExpertStream(self.hands_dim, d_model)
        self.expert_pose = ExpertStream(self.pose_dim, d_model)
        
        # Router
        self.router = VisibilityAwareRouter(self.face_dim, self.hands_dim, self.pose_dim)
        
        # Compresión temporal y proyección
        self.resampler = SlidingPerceiverResampler(
            d_model=d_model,
            d_llm=d_llm,
            window_size=window_size,
            stride=stride
        )
        
    def _infer_visibility(self, tensor):
        """
        Calcula una visibilidad "dummy" basada en si el tensor son puros ceros.
        MediaPipe devuelve ceros cuando no detecta el articulador.
        tensor: [B, T, num_points, 3]
        """
        B, T, _, _ = tensor.shape
        magnitude = torch.sum(torch.abs(tensor), dim=(2, 3), keepdim=True) # [B, T, 1, 1]
        vis = (magnitude > 1e-6).float().view(B, T, 1) # [B, T, 1]
        return vis

    def forward(self, streams):
        B, T = streams["pose"].shape[0], streams["pose"].shape[1]
        
        # Aplanar los tensores [B, T, pts, 3] -> [B, T, dim]
        face_f = streams["face"].view(B, T, -1)
        # Manos concatenadas (Izquierda y Derecha)
        hands_f = torch.cat([streams["left_hand"].view(B, T, -1), streams["right_hand"].view(B, T, -1)], dim=-1)
        pose_f = streams["pose"].view(B, T, -1)
        
        # Inferir visibilidad sobre la marcha
        face_v = self._infer_visibility(streams["face"])
        left_v = self._infer_visibility(streams["left_hand"])
        right_v = self._infer_visibility(streams["right_hand"])
        # Para las manos, tomamos el promedio de confianza de ambas manos
        hands_v = (left_v + right_v) / 2.0
        pose_v = self._infer_visibility(streams["pose"])
        
        # Enrutamiento Dinámico
        g_weights = self.router(face_f, hands_f, pose_f, face_v, hands_v, pose_v)
        
        # Procesamiento en Paralelo de Expertos
        out_face = self.expert_face(face_f)
        out_hands = self.expert_hands(hands_f)
        out_pose = self.expert_pose(pose_f)
        
        # Gated Fusion Junction
        fused = (
            g_weights[:, :, 0].unsqueeze(-1) * out_face +
            g_weights[:, :, 1].unsqueeze(-1) * out_hands +
            g_weights[:, :, 2].unsqueeze(-1) * out_pose
        )
        
        # Compresión Temporal y Mapeo al Latente de Gemma
        visual_tokens = self.resampler(fused)
        return visual_tokens

# Alias para mantener compatibilidad con el pipeline de carga/entrenamiento actual
LandmarkEncoder = LightSignMoEProjector

if __name__ == "__main__":
    print("[*] Iniciando prueba de sanidad del LightSignMoEProjector...")
    
    simulated_batch = {
        "left_hand": torch.randn(2, 10, 21, 3),
        "right_hand": torch.randn(2, 10, 21, 3),
        "face": torch.randn(2, 10, 468, 3),
        "pose": torch.randn(2, 10, 33, 3)
    }
    
    # Simulamos oclusión temporal (la mano izquierda sale de cámara)
    simulated_batch["left_hand"][:, 5:, :, :] = 0.0
    
    model = LandmarkEncoder()
    output = model(simulated_batch)
    
    print(f"[+] Entrada: 10 frames de landmarks Multi-Stream.")
    print(f"[+] Salida de Visual Tokens condensados: {output.shape}")
    print(f"[+] ¿Coincide la dimensión final con Gemma 4 E4B (2560)?: {'SÍ' if output.shape[2] == 2560 else 'NO'}")
    print(f"[+] ¿Compresión temporal aplicada exitosamente?: {'SÍ' if output.shape[1] < 10 else 'NO'}")
