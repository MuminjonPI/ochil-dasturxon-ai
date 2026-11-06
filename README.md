# 🍽️ Ochil Dasturxon AI — Uzbek Dish Classifier

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX_Runtime_Web-005CED?logo=onnx&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![Netlify](https://img.shields.io/badge/Netlify-00C7B7?logo=netlify&logoColor=white)

A web app that recognises four Uzbek dishes — **lag'mon, manti, palov (plov) and somsa** — from a photo.
Built with **transfer learning**: a ResNet18 pretrained on ImageNet, fine-tuned on Uzbek dish images.

**Live demo:** https://ochil-dasturxon-ai.netlify.app — runs entirely in your browser, your photo never leaves your device.

## How it works

1. **Data:** 4 classes from the [Central Asian Food Dataset (CAFD)](https://huggingface.co/datasets/issai/Central_Asian_Food_Dataset)
   by ISSAI — train 1,932 / val 415 / test 405 images. CAFD's three lagman classes
   (fried, with soup, without soup) were merged into one `lagman` class.
2. **Model:** `timm` ResNet18 with ImageNet weights; the last layer was replaced with `Linear(512 → 4)`
   and the whole network was fine-tuned (Adam, lr = 3e-4, 10 epochs, batch 32, random horizontal flip).
3. **Web app (live):** the model is exported to ONNX and runs in the browser with ONNX Runtime Web — a static site hosted on Netlify (`web/`).
4. **Python app:** the same design as a [Streamlit](https://streamlit.io) app (`app.py`) for running locally.

## Tech stack

| Area | Tools |
|---|---|
| Language | Python, JavaScript, HTML/CSS |
| Deep learning | PyTorch, timm (ResNet18), torchvision |
| Data & evaluation | NumPy, Pillow, scikit-learn, Matplotlib, Hugging Face Hub (dataset download) |
| Training environment | Jupyter Notebook on Google Colab (T4 GPU) |
| Model export | ONNX, ONNX Runtime (Python) for checking |
| Web app | ONNX Runtime Web (in-browser inference), HTML/CSS/JS |
| Python app | Streamlit |
| Hosting & tools | Netlify, Git & GitHub |

## Results

| Set | Accuracy | Macro F1 |
|---|---|---|
| Validation | 0.995 | 0.994 |
| Test | 0.993 | 0.993 |

The 3 test-set mistakes were unusual images (fried lagman with minced meat, raw manti packed on a tray,
a plov photo mostly showing salad). Full training, evaluation and error analysis are in `HW_62.ipynb`.

## Project files

| File | What it is |
|---|---|
| `web/index.html` | Live website (HTML/CSS/JS, runs the model with ONNX Runtime Web) |
| `web/model.onnx` | Model in ONNX format for the browser (~45 MB) |
| `netlify.toml` | Netlify settings (publishes the `web` folder) |
| `export_onnx.py` | Converts the PyTorch model to ONNX and checks the outputs match |
| `app.py` | Streamlit version of the app (page design and layout) |
| `classifier.py` | Model code: loads ResNet18 and predicts one image |
| `uzbek_dishes_fp16.pt` | Fine-tuned model weights (float16, ~22 MB; same accuracy as float32) |
| `requirements.txt` | Python packages for the app |
| `HW_62.ipynb` | Training notebook (Google Colab) |

## Run locally

**Website version:**

```bash
python -m http.server 8000 --directory web
# open http://localhost:8000
```

**Streamlit version:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations

- The model always picks one of the 4 dishes — it has no "other" class.
- The 4 dishes look quite different, so the task is easier than fine-grained food classification.
- Single training run; results can vary slightly between runs.

## Credits

- Dataset: Karabay, A., Bolatov, A., Varol, H. A., & Chan, M.-Y. (2023).
  *A Central Asian Food Dataset for Personalized Dietary Interventions.* Nutrients, 15(7), 1728.
  Licensed CC BY-NC 4.0 — this project is for non-commercial, educational use.
- Course: Najot Ta'lim Machine Learning, Lesson 62 (Transfer learning).
