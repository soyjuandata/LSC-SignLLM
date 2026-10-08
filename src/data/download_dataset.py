import os
import zipfile
import requests
from tqdm import tqdm

def download_dataset():
    # Define URL and Paths
    url = "https://data.mendeley.com/public-files/datasets/9ssyn8tff5/files/5a25e9cb-d67b-4590-bb90-e1784a90a578/file_downloaded"
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))
    zip_path = os.path.join(raw_dir, "LSC70.zip")
    extract_dir = os.path.join(raw_dir, "LSC70")

    # 1. Create target directories if they don't exist
    print(f"[*] Creando directorios en: {raw_dir}")
    os.makedirs(raw_dir, exist_ok=True)

    # 2. Download dataset
    print(f"[*] Iniciando descarga de LSC70 desde Mendeley Data...")
    print(f"[*] URL: {url}")
    
    # We streaming to show tqdm progress bar
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        # Download block size
        block_size = 1024 * 1024 # 1 MB
        
        with open(zip_path, 'wb') as file, tqdm(
            desc="Descargando LSC70.zip",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
                
        print(f"[+] Descarga completada con éxito. Guardado en: {zip_path}")
        
    except Exception as e:
        print(f"[-] Error durante la descarga: {str(e)}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False

    # 3. Unzip files
    print(f"[*] Descomprimiendo dataset LSC70...")
    try:
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files to show extraction progress
            file_list = zip_ref.namelist()
            with tqdm(desc="Extrayendo archivos", total=len(file_list), unit="archivo") as bar:
                for file_info in file_list:
                    zip_ref.extract(file_info, extract_dir)
                    bar.update(1)
                    
        print(f"[+] Dataset descomprimido con éxito en: {extract_dir}")
        
        # Optionally clean up the zip file to save space
        print(f"[*] Limpiando archivo temporal LSC70.zip...")
        os.remove(zip_path)
        print(f"[+] Limpieza completada.")
        return True
        
    except Exception as e:
        print(f"[-] Error durante la descompresión: {str(e)}")
        return False

if __name__ == "__main__":
    download_dataset()
