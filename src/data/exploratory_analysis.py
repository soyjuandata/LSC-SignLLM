import os
import cv2
import json
from tqdm import tqdm
from collections import defaultdict

def run_exploratory_analysis():
    # Define paths
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/LSC70"))
    report_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed"))
    report_json_path = os.path.join(report_dir, "exploratory_report.json")
    report_md_path = os.path.join(report_dir, "exploratory_report.md")

    os.makedirs(report_dir, exist_ok=True)

    if not os.path.exists(raw_dir):
        print(f"[-] Error: No se encontró la carpeta del dataset en: {raw_dir}")
        return False

    print(f"[*] Escaneando y analizando el dataset LSC70 en: {raw_dir} ...")

    # Mappings
    total_images_scanned = 0
    corrupt_images = []
    class_distribution = defaultdict(int)
    class_frame_counts = defaultdict(list)

    # Walk through the directories
    # The structure of LSC70 is typically:
    # LSC70/
    # ├── LSC70AN/
    # │   └── Volunteer_XX/
    # │       └── Sign_YY/
    # │           └── Image_ZZ.jpg
    # ...
    
    # We find all image files
    all_image_paths = []
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_image_paths.append(os.path.join(root, file))

    total_images = len(all_image_paths)
    print(f"[+] Se encontraron {total_images} imágenes en total.")

    if total_images == 0:
        print("[-] Error: No se encontraron imágenes JPG/PNG en el dataset.")
        return False

    # Check images for corruption and record distribution
    for img_path in tqdm(all_image_paths, desc="Validando integridad de imágenes"):
        total_images_scanned += 1
        
        # 1. Parse information from path
        # Let's extract the sign category
        # path is like: .../LSC70/LSC70W/Volunteer_01/HOLA/frame_1.jpg
        parts = img_path.replace("\\", "/").split("/")
        
        # Determine the section (LSC70AN, LSC70ANH, LSC70W)
        section = "Unknown"
        for s in ["LSC70AN", "LSC70ANH", "LSC70W"]:
            if s in parts:
                section = s
                break
                
        # The sign category is usually the directory immediately containing the frames, or the parent folder
        # Let's find the sign name. Let's assume it's parts[-2] (the folder name of the sign)
        # e.g., parts = [..., 'LSC70W', 'Volunteer_01', 'HOLA', 'frame_1.jpg'] -> sign = 'HOLA'
        sign_name = parts[-2] if len(parts) >= 2 else "Unknown"
        
        class_distribution[sign_name] += 1

        # 2. Check for corruption using OpenCV
        try:
            img = cv2.imread(img_path)
            if img is None or img.size == 0:
                corrupt_images.append(img_path)
        except Exception:
            corrupt_images.append(img_path)

    # 3. Compute stats
    num_corrupt = len(corrupt_images)
    print(f"\n[+] Análisis completado.")
    print(f"    - Imágenes escaneadas: {total_images_scanned}")
    print(f"    - Imágenes corruptas: {num_corrupt}")
    
    unique_signs = sorted(list(class_distribution.keys()))
    print(f"    - Clases únicas (Señas): {len(unique_signs)}")

    # 4. Generate JSON Report
    report_data = {
        "dataset_name": "LSC70",
        "total_images_scanned": total_images_scanned,
        "total_corrupt_images": num_corrupt,
        "corrupt_images_list": corrupt_images,
        "total_classes": len(unique_signs),
        "classes": unique_signs,
        "class_distribution": dict(class_distribution)
    }

    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)
    print(f"[+] Reporte JSON guardado en: {report_json_path}")

    # 5. Generate Markdown Report
    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write("# 📊 Reporte de Análisis Exploratorio - Dataset LSC70\n\n")
        f.write(f"Este reporte detalla la validación estructural y distribución de clases del dataset LSC70.\n\n")
        f.write("## 📈 Métricas Generales de Calidad\n\n")
        f.write(f"- **Total de imágenes escaneadas:** {total_images_scanned}\n")
        f.write(f"- **Imágenes corruptas detectadas:** {num_corrupt}\n")
        f.write(f"- **Total de clases únicas (Glosas LSC):** {len(unique_signs)}\n")
        f.write(f"- **Estado de Calidad:** {'✅ 100% Correcto (0 corruptas)' if num_corrupt == 0 else '⚠️ Archivos corruptos detectados'}\n\n")
        
        if num_corrupt > 0:
            f.write("### ⚠️ Lista de Archivos Corruptos\n")
            for corrupt_img in corrupt_images[:10]:
                f.write(f"- `{os.path.basename(corrupt_img)}` (`{corrupt_img}`)\n")
            if num_corrupt > 10:
                f.write(f"- ... y {num_corrupt - 10} más.\n")
            f.write("\n")
            
        f.write("## 🗂️ Distribución de Clases (Muestras por Seña)\n\n")
        f.write("| No. | Glosa de Seña (Clase) | Total Frames / Imágenes |\n")
        f.write("| --- | --------------------- | ----------------------- |\n")
        for i, (sign, count) in enumerate(sorted(class_distribution.items(), key=lambda x: x[1], reverse=True), 1):
            f.write(f"| {i} | `{sign}` | {count} |\n")
            
    print(f"[+] Reporte Markdown guardado en: {report_md_path}")
    return num_corrupt == 0

if __name__ == "__main__":
    run_exploratory_analysis()
