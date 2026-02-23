# 🧠 OmniMLX
**The Universal AI Backend for macOS (Apple Silicon)**

![Platform](https://img.shields.io/badge/Platform-M1%20%7C%20M2%20%7C%20M3%20%7C%20M4-gray?logo=apple) ![License](https://img.shields.io/badge/License-MIT-blue) ![Status](https://img.shields.io/badge/Status-Production-success)

**Stop reinventing the wheel for every AI project.**

OmniMLX is a "One-Stop-Shop" that turns your Mac into a local AI Server Farm. It exposes standard OpenAI-compatible APIs for **LLMs, Speech-to-Text, Text-to-Speech, and Vision**.

Run it in the background, and connect **any** tool to it.

## 🚀 Why OmniMLX?

### 🔗 The "Universal Integrator"
OmniMLX isn't just a chat app. It's a **Backend** for your entire workflow.
*   **VS Code (Cline/Roo Code):** Point your coding assistant to `http://localhost:8080`.
*   **Obsidian/Notes:** Use the "Ears" port (`8081`) for local dictation.
*   **Custom Python Scripts:** Build agents that can See, Hear, and Speak without needing 4 different libraries.
*   **Home Automation:** Run a local TTS "Mouth" (`8082`) for your smart home.

### ⚡️ Features
*   **🎛 4-Sense Dashboard:** Dedicated tabs for Brain, Ears, Mouth, and Eyes.
*   **🔍 Built-in Model Browser:** Search HuggingFace and download MLX models directly within the app.
*   **☁️ Hybrid Gateway:** Flip a switch to route difficult tasks to GPT-4o (Cloud) while keeping easy tasks Local (Privacy).
*   **🔋 Zero-Config:** Comes pre-loaded with Llama 3.2, Whisper v3, and Kokoro. Just click "Start."
<img width="646" height="616" alt="Screenshot 2026-02-22 at 9 50 19 PM" src="https://github.com/user-attachments/assets/b49e8627-95d0-4775-a068-df3947528fc0" />
<img width="646" height="616" alt="Screenshot 2026-02-22 at 9 50 07 PM" src="https://github.com/user-attachments/assets/a40509d9-d591-4344-a04c-aca78a4d436f" />
<img width="646" height="616" alt="Screenshot 2026-02-22 at 9 49 56 PM" src="https://github.com/user-attachments/assets/ab29f371-a044-4713-b6d2-1abfe184f0e3" />
<img width="646" height="616" alt="Screenshot 2026-02-22 at 9 49 47 PM" src="https://github.com/user-attachments/assets/ca378d3c-06b2-4ee9-9f54-16b2e4b55215" />
<img width="646" height="616" alt="Screenshot 2026-02-22 at 9 49 39 PM" src="https://github.com/user-attachments/assets/93c2b181-66a9-43b4-9cc6-230b9982f4f1" />
<img width="646" height="616" alt="Screenshot 2026-02-22 at 9 49 22 PM" src="https://github.com/user-attachments/assets/4ce9d348-11d3-4355-83b2-7c50213e001b" />


---


## 🛠 Quick Start

1.  **Clone & Install**
    ```bash
    git clone https://github.com/YOUR_USERNAME/OmniMLX.git
    cd OmniMLX
    chmod +x setup.sh
    ./setup.sh
    ```

2.  **Run**
    Double-click **`OmniMLX_Launcher.command`**.

3.  **Connect Your Tools**
    OmniMLX mimics the OpenAI API standard.

    | Service | Port | Endpoint |
    | :--- | :--- | :--- |
    | **Brain** (LLM) | `8080` | `POST /v1/chat/completions` |
    | **Ears** (STT) | `8081` | `POST /v1/audio/transcriptions` |
    | **Mouth** (TTS) | `8082` | `POST /v1/audio/speech` |
    | **Eyes** (Vision) | `8083` | `POST /v1/chat/completions` |

---

## 🔌 Integration Examples

**Python Client (using openai lib):**
```python
from openai import OpenAI

# Connect to OmniMLX Brain
client = OpenAI(base_url="http://localhost:8080/v1", api_key="mlx")

response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

**VS Code / Cline Settings:**
*   API Provider: OpenAI Compatible
*   Base URL: `http://localhost:8080/v1`
*   Model ID: `local-model` (or whatever is running)

---

## 📦 What's Included?

*   **Search:** Integrated HuggingFace Model finder.
*   **Launcher:** Native MacOS .command launcher.
*   **Management:** Auto-download and cache management via MLX.

## 📜 License

MIT License. Built with [Apple MLX](https://github.com/ml-explore/mlx).
