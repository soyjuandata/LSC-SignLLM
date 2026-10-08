import cv2
import mediapipe as mp
import numpy as np
import time

# Acceder a las utilidades a través del objeto principal 'mp'
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def draw_hud(image, scale_lh, scale_rh, fps):
    """Dibuja un HUD (Heads-Up Display) futurista y limpio sobre el feed de video."""
    h, w, _ = image.shape
    # Fondo semi-transparente para lecturas matemáticas
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (320, 140), (15, 23, 42), -1) # Slate-900
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Texto HUD
    cv2.putText(image, "LSC-SignLLM :: TELEMETRIA LIVE", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1, cv2.LINE_AA) # Sky-400
    cv2.line(image, (20, 45), (300, 45), (51, 65, 85), 1) # Border
    
    cv2.putText(image, f"FPS: {fps:.1f}", (20, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (241, 245, 249), 1, cv2.LINE_AA)
    
    # Mostrar factores de escala locales en tiempo real
    status_lh = f"Escala Mano Izq: {scale_lh:.4f}" if scale_lh > 0 else "Escala Mano Izq: NO DETECTADA"
    color_lh = (74, 222, 128) if scale_lh > 0 else (248, 113, 113) # Green vs Red
    cv2.putText(image, status_lh, (20, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_lh, 1, cv2.LINE_AA)
    
    status_rh = f"Escala Mano Der: {scale_rh:.4f}" if scale_rh > 0 else "Escala Mano Der: NO DETECTADA"
    color_rh = (74, 222, 128) if scale_rh > 0 else (248, 113, 113)
    cv2.putText(image, status_rh, (20, 115), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rh, 1, cv2.LINE_AA)
    
    # Marco exterior estético
    cv2.rectangle(image, (0, 0), (w, h), (51, 65, 85), 2)

def get_hand_scale(hand_landmarks):
    """Calcula el factor de escala local de la mano en tiempo real."""
    if hand_landmarks is None:
        return 0.0
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = coords[0]
    mcp_middle = coords[9]
    return np.linalg.norm(wrist - mcp_middle)

def main():
    import sys
    import os
    import glob

    print("[*] Iniciando capturador en tiempo real de LSC-SignLLM...")
    print("[*] Presiona 'q' en la ventana de video para salir.")
    
    # Intentar abrir la cámara
    cap = cv2.VideoCapture(0)
    use_mock = False
    mock_frames = []
    mock_idx = 0
    
    if not cap.isOpened():
        print("[!] No se pudo acceder a la cámara web. Activando modo DEMO con frames del dataset...")
        # Buscar un directorio de ejemplo en el dataset
        base_search = "data/raw/LSC70/LSC70/LSC70W/Per09/ANNOS"
        if os.path.exists(base_search):
            # Obtener y multiplicar los frames para que el video dure más (bucle x10)
            mock_frames = sorted(glob.glob(os.path.join(base_search, "*.jpg"))) * 10
            if mock_frames:
                use_mock = True
                print(f"[+] Modo DEMO activado. Reproduciendo secuencialmente {len(mock_frames)} frames de: {base_search}")
            else:
                print("[-] Error: No se encontraron imágenes en el directorio de ejemplo.")
                return
        else:
            print(f"[-] Error: Directorio de dataset '{base_search}' no encontrado para simular video.")
            return
        
    prev_time = 0
    
    # Inicializar rastreador Holistic
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        
        while True:
            if use_mock:
                # Leer la siguiente imagen de la secuencia mock
                img_path = mock_frames[mock_idx]
                frame = cv2.imread(img_path)
                if frame is None:
                    print("[-] Error cargando imagen mock.")
                    break
                mock_idx = (mock_idx + 1) % len(mock_frames)
                time.sleep(0.1) # Simular 10 FPS
                success = True
            else:
                success, frame = cap.read()
                if not success:
                    print("[-] Saltando fotograma vacío...")
                    continue
                
            # Espejar la imagen para un feedback natural
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # MediaPipe requiere RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)
            
            # Calcular FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0.0
            prev_time = curr_time
            
            # Extraer escalas locales para visualización interactiva
            scale_lh = get_hand_scale(results.left_hand_landmarks)
            scale_rh = get_hand_scale(results.right_hand_landmarks)
            
            # --- DIBUJAR ESQUELETOS (MediaPipe) ---
            # Rostro
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                
            # Pose
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
            # Mano Izquierda
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
                )
                
            # Mano Derecha
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
                )
            
            # Dibujar telemetría HUD
            draw_hud(frame, scale_lh, scale_rh, fps)
            
            # Inicializar VideoWriter si no está inicializado
            if 'out' not in locals() or out is None:
                output_path = "tests/output_demo.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # Bajar los FPS a 5.0 para que se vea más lento y claro
                out = cv2.VideoWriter(output_path, fourcc, 5.0, (w, h))
                print(f"[+] Inicializado grabador de video en: {output_path}")
            
            # Guardar el frame procesado en el archivo de video
            out.write(frame)
            
            # Mostrar ventana (Desactivado para WSL2 para evitar core dumps)
            # if "DISPLAY" in os.environ:
            #     try:
            #         cv2.imshow("LSC-SignLLM :: Real-time Visualizer", frame)
            #         if cv2.waitKey(5) & 0xFF == ord('q'):
            #             break
            #     except cv2.error:
            #         pass
            # else:
            
            # Simular 10 FPS de procesamiento
            time.sleep(0.05)
                
            # En modo demo, si terminamos la secuencia, podemos parar tras guardar todo
            if use_mock and mock_idx == 0:
                print("[+] Se ha completado una reproducción completa de la secuencia DEMO.")
                break
                
    if 'out' in locals() and out is not None:
        out.release()
        print("[+] Guardado video completo con telemetría en: tests/output_demo.mp4")
        
    cap.release()
    # cv2.destroyAllWindows()
    print("[+] Capturador cerrado exitosamente.")
    
    # Abrir el video automáticamente usando el reproductor predeterminado de Windows
    if 'out' in locals() and out is not None and os.path.exists("tests/output_demo.mp4"):
        import subprocess
        print("[*] Abriendo video en el reproductor de Windows...")
        # Convierte el path para que cmd de Windows lo entienda y lo abra
        subprocess.call(['cmd.exe', '/c', 'start', '', 'tests\\output_demo.mp4'])

if __name__ == "__main__":
    main()
