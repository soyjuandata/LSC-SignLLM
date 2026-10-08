import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import re

# Inicializar MediaPipe Holistic
mp_holistic = mp.solutions.holistic

def get_numerical_sort_key(filename):
    """Retorna los números dentro del nombre del archivo para ordenación numérica natural."""
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else filename

def normalize_hand(hand_landmarks):
    """
    Normalización local Multi-Stream para una Mano:
    - Centrado en la muñeca (Landmark 0).
    - Escalado por la distancia entre la muñeca (0) y la articulación MCP del dedo medio (9).
    """
    if hand_landmarks is None:
        return np.zeros((21, 3), dtype=np.float32)
    
    # Extraer coordenadas
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    
    wrist = coords[0]
    mcp_middle = coords[9]
    
    # Calcular factor de escala (evitar división por cero con epsilon)
    scale = np.linalg.norm(wrist - mcp_middle) + 1e-8
    
    # Mapeo normalizado
    normalized = (coords - wrist) / scale
    return normalized

def normalize_face(face_landmarks):
    """
    Normalización local Multi-Stream para el Rostro (468 keypoints):
    - Centrado en la punta de la nariz (Landmark 1).
    - Escalado por la distancia interpupilar aproximada (Landmarks 33 y 263 - lagrimales/esquinas de los ojos).
    """
    if face_landmarks is None:
        return np.zeros((468, 3), dtype=np.float32)
    
    coords = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark], dtype=np.float32)
    
    nose_tip = coords[1]
    left_eye_corner = coords[33]
    right_eye_corner = coords[263]
    
    scale = np.linalg.norm(left_eye_corner - right_eye_corner) + 1e-8
    
    normalized = (coords - nose_tip) / scale
    return normalized

def normalize_pose(pose_landmarks):
    """
    Normalización local Multi-Stream para la Pose (33 keypoints):
    - Centrado en el punto medio entre los hombros (11 y 12).
    - Escalado por la distancia euclidiana entre ambos hombros.
    """
    if pose_landmarks is None:
        return np.zeros((33, 3), dtype=np.float32)
    
    coords = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks.landmark], dtype=np.float32)
    
    left_shoulder = coords[11]
    right_shoulder = coords[12]
    
    midpoint = (left_shoulder + right_shoulder) / 2.0
    scale = np.linalg.norm(left_shoulder - right_shoulder) + 1e-8
    
    normalized = (coords - midpoint) / scale
    return normalized

def process_sequence(image_paths, holistic_tracker):
    """Procesa una secuencia de imágenes (frames de una seña) y retorna tensores normalizados por stream."""
    seq_left_hand = []
    seq_right_hand = []
    seq_face = []
    seq_pose = []
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # MediaPipe requiere formato RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = holistic_tracker.process(img_rgb)
        
        # Aplicar normalizaciones locales e independientes por stream
        seq_left_hand.append(normalize_hand(results.left_hand_landmarks))
        seq_right_hand.append(normalize_hand(results.right_hand_landmarks))
        seq_face.append(normalize_face(results.face_landmarks))
        seq_pose.append(normalize_pose(results.pose_landmarks))
        
    if not seq_pose:
        return None
        
    # Convertir a tensores de PyTorch
    return {
        "left_hand": torch.tensor(np.array(seq_left_hand), dtype=torch.float32),   # [T, 21, 3]
        "right_hand": torch.tensor(np.array(seq_right_hand), dtype=torch.float32), # [T, 21, 3]
        "face": torch.tensor(np.array(seq_face), dtype=torch.float32),             # [T, 468, 3]
        "pose": torch.tensor(np.array(seq_pose), dtype=torch.float32)              # [T, 33, 3]
    }

def run_extraction():
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/LSC70"))
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed/landmarks"))
    
    if not os.path.exists(raw_dir):
        print(f"[-] Error: No se encontró la carpeta del dataset en: {raw_dir}")
        return
        
    print(f"[*] Iniciando extracción Multi-Stream con MediaPipe Holistic...")
    
    # Agrupar imágenes por secuencia/seña
    # Las secuencias están contenidas en carpetas como: raw_dir/Sección/Volunteer_XX/Sign_YY/
    sequences = {}
    
    for root, dirs, files in os.walk(raw_dir):
        # Si la carpeta contiene imágenes, es una secuencia
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            # Ordenar numéricamente para preservar la secuencia temporal correcta (frame_1, frame_2...)
            image_files.sort(key=get_numerical_sort_key)
            full_paths = [os.path.join(root, f) for f in image_files]
            
            # Obtener identificador único (ej: LSC70W/Volunteer_01/HOLA)
            rel_path = os.path.relpath(root, raw_dir)
            sequences[rel_path] = full_paths
            
    print(f"[+] Se detectaron {len(sequences)} secuencias de señas en total.")
    
    # Inicializar el procesador MediaPipe
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        for seq_name, img_paths in tqdm(sequences.items(), desc="Procesando secuencias de LSC70"):
            tensors = process_sequence(img_paths, holistic)
            
            if tensors is not None:
                # Guardar en data/processed/landmarks/
                output_path = os.path.join(processed_dir, f"{seq_name}.pt")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                torch.save(tensors, output_path)
                
    print(f"[+] ¡Extracción Multi-Stream completada con éxito!")
    print(f"[+] Los tensores normalizados se guardaron en: {processed_dir}")

if __name__ == "__main__":
    run_extraction()
