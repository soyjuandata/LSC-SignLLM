# 🌟 LSC-SignLLM

> **Multimodal Translation System for Colombian Sign Language (LSC) using Local LLMs, Edge Computer Vision, and Speech Synthesis.**
> A 100% offline, privacy-first, edge-computing assistant bridging communication barriers between deaf and hearing communities.

---

## 🏗️ Multimodal System Architecture

LSC-SignLLM operates a bidirectional, multimodal pipeline orchestrating spatial 3D keypoints, causal language modeling, local speech processing, and dynamic 3D rendering.

```mermaid
flowchart TB
    %% Styling Classes
    classDef client fill:#1e293b,stroke:#0f172a,stroke-width:2px,color:#f8fafc;
    classDef edgeModel fill:#0f766e,stroke:#115e59,stroke-width:2px,color:#f8fafc;
    classDef llm fill:#4338ca,stroke:#3730a3,stroke-width:2px,color:#f8fafc;
    classDef speech fill:#b45309,stroke:#92400e,stroke-width:2px,color:#f8fafc;
    classDef webgl fill:#6d28d9,stroke:#5b21b6,stroke-width:2px,color:#f8fafc;

    %% ----------------------------------------------------
    %% SIGN-TO-SPEECH PIPELINE
    %% ----------------------------------------------------
    subgraph S2S [Sign-to-Speech Flow (Sordo ──► Oyente)]
        direction LR
        Camera[Webcam Feed]:::client -->|30 FPS| MediaPipe[MediaPipe Holistic]:::client
        MediaPipe -->|3D Coords| Norm[Spatial Normalizer]:::edgeModel
        Norm -->|Scale-Invariant Vector| CTC[CTC Classifier Model]:::edgeModel
        CTC -->|Predicted LSC Glosses| TranslateOllama[Ollama: Gemma 3 / Qwen]:::llm
        TranslateOllama -->|Natural Spanish Text| Piper[Piper TTS Engine]:::speech
        Piper -->|Acoustic Modality| AudioOut[Local Audio Output]:::speech
    end

    %% ----------------------------------------------------
    %% SPEECH-TO-SIGN PIPELINE
    %% ----------------------------------------------------
    subgraph S2V [Speech-to-Sign Flow (Oyente ──► Sordo)]
        direction LR
        Mic[Microphone Input]:::speech -->|Audio Stream| Whisper[Whisper ASR Local]:::speech
        Whisper -->|Spanish Text| RevOllama[Ollama: Gemma 3 / Qwen]:::llm
        RevOllama -->|LSC Gloss Sequence| WebGL[WebGL Avatar Engine]:::webgl
        WebGL -->|Splines Hermite Cubic C¹| Avatar[3D Character Animation]:::webgl
    end

    %% Formatting
    style S2S fill:#0f172a,stroke:#334155,stroke-width:2px,color:#f8fafc
    style S2V fill:#0f172a,stroke:#334155,stroke-width:2px,color:#f8fafc
```

### 1. Stage 1: Spatial Invariance & Classification (Vision Modality)
* **Real-time Body-Tracking:** MediaPipe Holistic runs on client WebGL/JS, distributing the computing load.
* **Math Normalization:** Keypoints are localized using a mid-shoulder coordinate frame to eliminate scale and camera distance variations:
  $$p'_i = \frac{p_i - \mathbf{p}_{\text{midpoint}}}{\| p_{\text{shoulder\_L}} - p_{\text{shoulder\_R}} \|_2}$$
* **Temporal Mapping:** A lightweight PyTorch network trained with **Connectionist Temporal Classification (CTC) loss** converts variable-length coordinate frames into discrete LSC gloss sequences.

### 2. Stage 2: Causal Translation & Synthesis (NLP & Speech Modality)
* **Grammar Transformation:** A local LLM translates unstructured LSC glosses (`AGUA TOMAR QUERER YO`) into syntactically correct, conversational Spanish (`Quiero tomar agua`).
* **Offline Audio Output:** Highly efficient Piper TTS synthesizes natural-sounding speech locally, staying within a total latency budget of $\le 500\text{ ms}$.

### 3. Stage 3: Bidirectional Kinematics (Avatar Rendering Modality)
* **Interpolation:** To prevent robotic or sudden joint movements, transitions between signs in the dictionary are smoothed using **Cubic Hermite Splines**, ensuring $C^1$ continuity:
  $$p(t) = (2t^3 - 3t^2 + 1)p_A + (t^3 - 2t^2 + t)m_A + (-2t^3 + 3t^2)p_B + (t^3 - t^2)m_B$$
* **High-Fidelity Canvas:** Dynamic, lightweight 3D character rendering is achieved via WebGL and Three.js directly in the browser.

---

## 🔒 Security, Compliance & License

LSC-SignLLM is designed for enterprise deployment (SaaS B2B) and complies fully with data safety and licensing requirements:

* **100% Offline Edge Computing:** Ingestion, vison tracking, LLM translation, and speech generation run entirely on local nodes, protecting sensitive client-employee interaction.
* **Permissive Licenses:** Built exclusively on commercial-friendly components (**Apache License 2.0** / **MIT**).
* **CC BY 4.0 Verified Datasets:** Trained on approved local datasets (e.g., LSC70 by Universidad del Cauca), avoiding high-risk datasets with non-commercial restrictions (`CC BY-NC 4.0`).

---

*LSC-SignLLM is proudly built to empower Colombian sign language translation, bringing state-of-the-art AI directly to the edge.*
