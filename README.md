# LSC-SignLLM

**Multimodal Translation System for Colombian Sign Language (LSC)**  
Local LLMs · Edge Computer Vision · Offline Speech Synthesis · 3D Avatar

> A 100% offline, privacy-first system bridging communication barriers between deaf and hearing communities in Colombia.

---

## Overview

**LSC-SignLLM** is an open-source, privacy-preserving translator for Colombian Sign Language (Lengua de Senas Colombianas - LSC).

All processing runs **locally on-device** using Edge AI and Local Large Language Models (LLMs). Proyección continua de fotogramas usando un **Sliding Perceiver Resampler** hacia el espacio latente del LLM.

---

## System Architecture

### Sign-to-Speech Pipeline

```mermaid
flowchart LR
    A["Webcam Input"] --> B["MediaPipe Holistic"]
    B --> C["Spatial Normalizer"]
    C --> D["Sign-LLM Perceiver"]
    D --> E["Local LLM (Gemma 4 E4B)"]
    E --> F["Piper TTS"]
    F --> G["Audio Output"]

    style A fill:#1e293b,stroke:#475569,color:#f1f5f9
    style B fill:#1e293b,stroke:#475569,color:#f1f5f9
    style C fill:#134e4a,stroke:#0f766e,color:#f0fdf4
    style D fill:#134e4a,stroke:#0f766e,color:#f0fdf4
    style E fill:#312e81,stroke:#4338ca,color:#e0e7ff
    style F fill:#78350f,stroke:#b45309,color:#fef3c7
    style G fill:#78350f,stroke:#b45309,color:#fef3c7
```

### Speech-to-Sign Pipeline

```mermaid
flowchart LR
    A["Microphone Input"] --> B["Whisper ASR (Local)"]
    B --> C["Local LLM (Ollama)"]
    C --> D["Sign Dictionary"]
    D --> E["Hermite Spline Engine"]
    E --> F["3D Avatar (WebGL)"]

    style A fill:#78350f,stroke:#b45309,color:#fef3c7
    style B fill:#78350f,stroke:#b45309,color:#fef3c7
    style C fill:#312e81,stroke:#4338ca,color:#e0e7ff
    style D fill:#134e4a,stroke:#0f766e,color:#f0fdf4
    style E fill:#4a1d96,stroke:#6d28d9,color:#f5f3ff
    style F fill:#4a1d96,stroke:#6d28d9,color:#f5f3ff
```

### Full Stack Overview

```mermaid
flowchart TD
    Browser["Browser / Client"]

    subgraph EdgeLayer["Edge Layer (Client-Side)"]
        MP["MediaPipe Holistic (JS/WebGL)"]
        THR["Three.js Avatar Renderer (WebGL)"]
    end

    subgraph BackendLayer["Local Backend (FastAPI)"]
        PERC["Perceiver Resampler (PyTorch)"]
        LLM["Ollama: Gemma 4 E4B"]
        ASR["Whisper ASR"]
        TTS["Piper TTS"]
    end

    Browser --> MP
    MP --> PERC
    PERC --> LLM
    LLM --> TTS
    LLM --> THR
    ASR --> LLM

    style EdgeLayer fill:#0f172a,stroke:#334155,color:#f8fafc
    style BackendLayer fill:#0f172a,stroke:#334155,color:#f8fafc
    style Browser fill:#1e293b,stroke:#475569,color:#f1f5f9
    style MP fill:#1e293b,stroke:#475569,color:#f1f5f9
    style THR fill:#4a1d96,stroke:#6d28d9,color:#f5f3ff
    style PERC fill:#134e4a,stroke:#0f766e,color:#f0fdf4
    style LLM fill:#312e81,stroke:#4338ca,color:#e0e7ff
    style ASR fill:#78350f,stroke:#b45309,color:#fef3c7
    style TTS fill:#78350f,stroke:#b45309,color:#fef3c7
```

---

## Tech Stack

| Layer | Technology | License |
|:------|:-----------|:--------|
| Vision / Landmark Extraction | MediaPipe Holistic | Apache 2.0 |
| Multimodal Projection | PyTorch + Sliding Perceiver | BSD |
| Language Translation | Gemma 4 E4B via Ollama | Gemma / Apache 2.0 |
| Speech Transcription | Whisper (local) | MIT |
| Speech Synthesis | Piper TTS | MIT |
| Backend API | FastAPI | MIT |
| 3D Avatar Renderer | Three.js (WebGL) | MIT |
| Training Dataset | LSC70 — Universidad del Cauca | CC BY 4.0 |

---

## License

This project is open-source and licensed under the **Apache License 2.0**.  
All integrated datasets and models are verified for commercial use.

---

*Built to empower Colombian Sign Language communication — bringing state-of-the-art AI to the edge.*
