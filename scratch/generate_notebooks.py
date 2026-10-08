import os
import json

def create_notebook_1(output_path):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 📘 Guía de Estudio 1: Entendiendo los Landmarks de MediaPipe y su Normalización\n",
                "\n",
                "En esta primera guía interactiva, aprenderás:\n",
                "1. Cómo están estructurados los datos de entrada (landmarks 3D).\n",
                "2. Por qué necesitamos la **Normalización Multi-Stream**.\n",
                "3. Cómo manipular y visualizar estos datos paso a paso en PyTorch.\n",
                "\n",
                "---"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Concepto Base: ¿Qué es un Landmark?\n",
                "\n",
                "Un **landmark** es un punto clave en el espacio 3D ($x, y, z$) detectado por MediaPipe Holistic. Representa articulaciones, puntos del rostro o partes del cuerpo.\n",
                "\n",
                "En nuestro proyecto, procesamos **4 streams (flujos) independientes**:\n",
                "- **Mano Izquierda (Left Hand):** 21 landmarks ($21 \\times 3$ coordenadas).\n",
                "- **Mano Derecha (Right Hand):** 21 landmarks ($21 \\times 3$ coordenadas).\n",
                "- **Rostro (Face):** 468 landmarks ($468 \\times 3$ coordenadas).\n",
                "- **Pose (Cuerpo/Hombros):** 33 landmarks ($33 \\times 3$ coordenadas).\n",
                "\n",
                "Cuando guardamos una seña completa de $T$ frames (imágenes), guardamos un diccionario con 4 tensores de dimensiones:\n",
                "- `left_hand`: $[T, 21, 3]$\n",
                "- `right_hand`: $[T, 21, 3]$\n",
                "- `face`: $[T, 468, 3]$\n",
                "- `pose`: $[T, 33, 3]$"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "import os\n",
                "import glob\n",
                "\n",
                "# Vamos a buscar un archivo de landmarks real de nuestro dataset procesado\n",
                "data_dir = \"../../data/processed/landmarks\"\n",
                "file_paths = glob.glob(os.path.join(data_dir, \"**/*.pt\"), recursive=True)\n",
                "\n",
                "if len(file_paths) == 0:\n",
                "    print(\"[-] ¡No se encontraron archivos procesados! Asegúrate de haber ejecutado extract_landmarks.py\")\n",
                "else:\n",
                "    sample_path = file_paths[0]\n",
                "    print(f\"[+] Cargando archivo de ejemplo: {sample_path}\")\n",
                "    landmarks = torch.load(sample_path, weights_only=True)\n",
                "    \n",
                "    # Mostrar las dimensiones\n",
                "    for stream_name, tensor in landmarks.items():\n",
                "        print(f\"Stream: {stream_name:<12} | Dimensiones: {list(tensor.shape)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. ¿Por qué normalizar? (La matemática explicada)\n",
                "\n",
                "Si una persona firma una seña parada cerca de la cámara vs. lejos de la cámara, los números de las coordenadas de MediaPipe cambian drásticamente. Lo mismo pasa si se mueve a la izquierda o derecha.\n",
                "\n",
                "Para lograr **invarianza espacial** (que el modelo entienda el gesto sin importar la distancia o posición de la cámara), aplicamos normalización local por stream:\n",
                "\n",
                "#### 🖐️ Normalización de Mano\n",
                "1. **Centrado:** Restamos la coordenada de la muñeca (Landmark 0) a todos los demás puntos de la mano. Así, la muñeca se vuelve el origen $(0,0,0)$.\n",
                "2. **Escalado:** Calculamos la distancia euclidiana entre la muñeca (0) y el nudillo del dedo medio (9), y dividimos todas las coordenadas por esta distancia. Esto hace que si la mano está cerca o lejos, siempre tenga el mismo tamaño normalizado.\n",
                "\n",
                "$$\\mathbf{p}_{norm} = \\frac{\\mathbf{p} - \\mathbf{p}_{wrist}}{\\|\\mathbf{p}_{wrist} - \\mathbf{p}_{middle}\\| + \\epsilon}$$"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 📝 Ejercicio 1: Implementar normalización de mano de forma manual\n",
                "\n",
                "A continuación, tienes un tensor de mano artificial con 21 landmarks sin normalizar. Completa la función de normalización y valida que la muñeca (índice 0) quede exactamente en `[0.0, 0.0, 0.0]` y que las dimensiones se mantengan."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "\n",
                "# 21 landmarks aleatorios simulando coordenadas reales de una mano frente a la cámara\n",
                "mano_sucia = np.random.rand(21, 3) * 10.0 + 5.0 # valores entre 5.0 y 15.0\n",
                "\n",
                "def normalizar_mano_ejercicio(coords):\n",
                "    \"\"\"\n",
                "    Coords: numpy array de shape (21, 3)\n",
                "    \"\"\"\n",
                "    # TODO: Escribe tu código aquí\n",
                "    # 1. Obtén la muñeca (coords[0])\n",
                "    # 2. Obtén el nudillo del dedo medio (coords[9])\n",
                "    # 3. Calcula la distancia euclidiana entre muñeca y nudillo\n",
                "    # 4. Resta la muñeca de coords y divide por la distancia (añade 1e-8 para evitar dividir por 0)\n",
                "    pass\n",
                "\n",
                "# --- Validación ---\n",
                "mano_limpia = normalizar_mano_ejercicio(mano_sucia)\n",
                "if mano_limpia is not None:\n",
                "    print(f\"Posición de la muñeca normalizada (debe ser [0,0,0]): {mano_limpia[0]}\")\n",
                "    print(f\"Distancia normalizada muñeca-dedo medio (debe ser 1.0): {np.linalg.norm(mano_limpia[0] - mano_limpia[9]):.4f}\")\n",
                "else:\n",
                "    print(\"[-] Por favor, completa la función primero.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Solución sugerida para validar el ejercicio:\n",
                "```python\n",
                "def normalizar_mano_ejercicio(coords):\n",
                "    wrist = coords[0]\n",
                "    middle = coords[9]\n",
                "    scale = np.linalg.norm(wrist - middle) + 1e-8\n",
                "    return (coords - wrist) / scale\n",
                "```"
            ]
        }
    ]
    
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

def create_notebook_2(output_path):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 📘 Guía de Estudio 2: Compresión Temporal y Proyector MoE (Mixture of Experts)\n",
                "\n",
                "En esta segunda guía interactiva, aprenderás:\n",
                "1. Cómo convertimos coordenadas físicas a la dimensión del LLM (2560 para Gemma).\n",
                "2. Cómo el **Soft-Gating MoE** pondera la importancia de cada stream (por ejemplo, ignorando la mano si está oculta).\n",
                "3. Cómo el **Sliding Perceiver Resampler** comprime secuencias largas en un número fijo de tokens visuales.\n",
                "\n",
                "---"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. El Reto: Reducir y Adaptar Dimensiones\n",
                "\n",
                "El LLM (Gemma 4 E2B) no entiende coordenadas $(x, y, z)$. Él entiende vectores densos de **2560 dimensiones** (los llamados *tokens embeddings*).\n",
                "\n",
                "Además, una seña puede durar 30 frames (1 segundo) o 150 frames (5 segundos). Pero los LLMs necesitan un número **fijo y pequeño de tokens** para no saturar su contexto.\n",
                "\n",
                "Nuestra arquitectura hace lo siguiente:\n",
                "1. **Proyección Lineal:** Proyecta cada stream (mano, pose, rostro) de forma independiente a la dimensión intermedia (ej. 256).\n",
                "2. **MoE Router (Mixture of Experts):** Calcula dinámicamente qué stream es relevante en cada instante temporal.\n",
                "3. **Sliding Perceiver Resampler:** Aplica atención cruzada para resumir $T$ frames a solo $M$ tokens visuales fijos (ej. 4 tokens), y los proyecta finalmente a la dimensión del LLM (2560)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Inspeccionando el LightSignMoEProjector\n",
                "\n",
                "Vamos a instanciar el proyector que está definido en `src/models/landmark_encoder.py` y ver cómo fluyen los datos a través de él."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "import os\n",
                "import torch\n",
                "\n",
                "# Asegurar que la raíz del proyecto está en el path\n",
                "sys.path.append(os.path.abspath(\"../../\"))\n",
                "from src.models.landmark_encoder import LandmarkEncoder\n",
                "\n",
                "# Instanciamos el proyector con la dimensión de Gemma 4 E2B (2560)\n",
                "projector = LandmarkEncoder(d_llm=2560)\n",
                "print(projector)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 📝 Ejercicio 2: Pasar un Batch de Landmarks simulado y Validar la Salida\n",
                "\n",
                "Vamos a simular una seña de 60 frames con un batch de tamaño 2. \n",
                "Completa el script para:\n",
                "1. Definir tensores simulados con las formas correctas.\n",
                "2. Alimentarlos al proyector.\n",
                "3. Validar que la forma final de salida sea exactamente `[Batch, 4, 2560]` (donde 4 es el número de tokens visuales de salida)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "\n",
                "B = 2   # Batch size\n",
                "T = 60  # 60 frames de video\n",
                "\n",
                "# TODO: Completa las dimensiones correctas usando torch.randn\n",
                "# Tip: Mano tiene 21 landmarks, rostro 468, pose 33, y todos tienen x, y, z (3 coordenadas)\n",
                "streams = {\n",
                "    \"left_hand\": torch.randn(B, T, 21, 3),\n",
                "    \"right_hand\": None, # Reemplaza None por el shape correcto\n",
                "    \"face\": None,       # Reemplaza None por el shape correcto\n",
                "    \"pose\": None        # Reemplaza None por el shape correcto\n",
                "}\n",
                "\n",
                "# Correr el proyector\n",
                "try:\n",
                "    output = projector(streams)\n",
                "    print(f\"[+] ¡Éxito! Shape final de los tokens visuales: {list(output.shape)}\")\n",
                "    assert list(output.shape) == [B, 4, 2560], \"Dimensiones incorrectas. Deben ser [2, 4, 2560]\"\n",
                "except Exception as e:\n",
                "    print(f\"[-] Error durante la ejecución: {e}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Solución sugerida para validar las dimensiones:\n",
                "```python\n",
                "streams = {\n",
                "    \"left_hand\": torch.randn(B, T, 21, 3),\n",
                "    \"right_hand\": torch.randn(B, T, 21, 3),\n",
                "    \"face\": torch.randn(B, T, 468, 3),\n",
                "    \"pose\": torch.randn(B, T, 33, 3)\n",
                "}\n",
                "```"
            ]
        }
    ]
    
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

def create_notebook_3(output_path):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 📘 Guía de Estudio 3: Pérdida Contrastiva InfoNCE y Alineación de Espacios\n",
                "\n",
                "En esta tercera guía interactiva, aprenderás:\n",
                "1. Qué significa \"alinear representaciones\" visuales con representaciones de texto.\n",
                "2. Cómo funciona la **Pérdida InfoNCE (Contrastive Loss)** matemáticamente.\n",
                "3. Cómo calcular y optimizar esta pérdida usando PyTorch.\n",
                "\n",
                "---"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. El Concepto de Alineación Semántica\n",
                "\n",
                "Imagínate que tenemos dos mundos:\n",
                "- **Mundo Visual:** Los embeddings que genera nuestro proyector de landmarks de señas.\n",
                "- **Mundo Textual:** Los embeddings que genera el LLM para las palabras escritas (ej: \"HOLA\", \"ADIOS\").\n",
                "\n",
                "Inicialmente, el proyector genera vectores aleatorios porque no está entrenado. Por lo tanto, el vector del video de \"HOLA\" y la palabra de texto \"HOLA\" están muy lejos en el espacio matemático.\n",
                "\n",
                "La pérdida contrastiva **InfoNCE**:\n",
                "1. Toma un par de ejemplo coincidente (Video de HOLA, Texto de HOLA) -> Par positivo.\n",
                "2. Toma pares no coincidentes (Video de HOLA, Texto de ADIOS) -> Par negativo.\n",
                "3. Modifica los pesos del proyector para **maximizar la similitud coseno** del par positivo, y **minimizar** la similitud de los pares negativos."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Similitud Coseno e InfoNCE\n",
                "\n",
                "La similitud coseno entre dos vectores normalizados $\\mathbf{v}$ y $\\mathbf{t}$ es:\n",
                "\n",
                "$$\\text{sim}(\\mathbf{v}, \\mathbf{t}) = \\frac{\\mathbf{v} \\cdot \\mathbf{t}}{\\|\\mathbf{v}\\| \\|\\mathbf{t}\\|}$$"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "import torch.nn.functional as F\n",
                "\n",
                "# Simulación de 2 muestras de video y 2 muestras de texto en el batch (Dimensión = 2560)\n",
                "B, D = 2, 2560\n",
                "visual_embeds = torch.randn(B, D)\n",
                "text_embeds = torch.randn(B, D)\n",
                "\n",
                "# Normalización L2\n",
                "v_norm = F.normalize(visual_embeds, dim=-1)\n",
                "t_norm = F.normalize(text_embeds, dim=-1)\n",
                "\n",
                "# Multiplicación matricial para obtener similitudes entre todos\n",
                "similarity_matrix = torch.matmul(v_norm, t_norm.transpose(0, 1))\n",
                "print(\"Matriz de similitud original (aleatoria):\\n\", similarity_matrix)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 📝 Ejercicio 3: Implementar la pérdida InfoNCE básica\n",
                "\n",
                "Implementa la función de pérdida InfoNCE y observa cómo varía al cambiar los vectores para que sean más similares o diferentes."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def calcular_infonce(visual, text, temperature=0.07):\n",
                "    # TODO:\n",
                "    # 1. Normaliza los vectores visual y text usando F.normalize\n",
                "    # 2. Calcula la matriz de similitud usando multiplicación matricial y dividiendo por la temperatura\n",
                "    # 3. Crea las etiquetas target (deben ser torch.arange(B))\n",
                "    # 4. Calcula la pérdida de cross_entropy en ambos ejes (visual a texto, y texto a visual) y saca el promedio\n",
                "    pass\n",
                "\n",
                "# --- Validación ---\n",
                "loss = calcular_infonce(visual_embeds, text_embeds)\n",
                "if loss is not None:\n",
                "    print(f\"[+] Pérdida calculada: {loss.item():.4f}\")\n",
                "else:\n",
                "    print(\"[-] Por favor, completa la función primero.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Solución sugerida para validar la pérdida:\n",
                "```python\n",
                "def calcular_infonce(visual, text, temperature=0.07):\n",
                "    v_norm = F.normalize(visual, dim=-1)\n",
                "    t_norm = F.normalize(text, dim=-1)\n",
                "    logits = torch.matmul(v_norm, t_norm.transpose(0, 1)) / temperature\n",
                "    labels = torch.arange(logits.shape[0], device=logits.device)\n",
                "    loss_v = F.cross_entropy(logits, labels)\n",
                "    loss_t = F.cross_entropy(logits.transpose(0, 1), labels)\n",
                "    return (loss_v + loss_t) / 2.0\n",
                "```"
            ]
        }
    ]
    
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

if __name__ == "__main__":
    # Obtener la ruta absoluta de la raíz del proyecto (un nivel arriba de scratch/)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    study_guide_dir = os.path.join(project_root, "notebooks", "study_guide")
    os.makedirs(study_guide_dir, exist_ok=True)
    
    create_notebook_1(os.path.join(study_guide_dir, "01_understanding_landmarks.ipynb"))
    create_notebook_2(os.path.join(study_guide_dir, "02_moe_projection_and_perceiver.ipynb"))
    create_notebook_3(os.path.join(study_guide_dir, "03_contrastive_alignment_infonce.ipynb"))
    print(f"[+] Cuadernos de estudio generados con éxito en: {study_guide_dir}")
