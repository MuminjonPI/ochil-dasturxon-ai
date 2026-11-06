# 🍽️ Ochil Dasturxon AI — Uzbek Dish Classifier

A web app that recognises four Uzbek dishes — **lag'mon, manti, palov (plov) and somsa** — from a photo.
Built with **transfer learning**: a ResNet18 pretrained on ImageNet, fine-tuned on Uzbek dish images.

**Live demo:** *https://ochil-dasturxon-ai.netlify.app*

## How it works

1. **Data:** 4 classes from the [Central Asian Food Dataset (CAFD)](https://huggingface.co/datasets/issai/Central_Asian_Food_Dataset)
by ISSAI — train 1,932 / val 415 / test 405 images. CAFD's three lagman classes
(fried, with soup, without soup) were merged into one `lagman` class.
2. **Model:** `timm` ResNet18 with ImageNet weights; the last layer was replaced with `Linear(512 → 4)`
and the whole network was fine-tuned (Adam, lr = 3e-4, 10 epochs, batch 32, random horizontal flip).
3. **App:** [Streamlit](https://streamlit.io) — upload a photo, get the predicted dish and the probability of each class.

## Results

|Set|Accuracy|Macro F1|
|-|-|-|
|Validation|0.995|0.994|
|Test|0.993|0.993|

The 3 test-set mistakes were unusual images (fried lagman with minced meat, raw manti packed on a tray,
a plov photo mostly showing salad). Full training, evaluation and error analysis are in `HW\_62.ipynb`.

## Project files

|File|What it is|
|-|-|
|`app.py`|Streamlit web app (page design and layout)|
|`classifier.py`|Model code: loads ResNet18 and predicts one image|
|`uzbek\_dishes\_fp16.pt`|Fine-tuned model weights (float16, \~22 MB; same accuracy as float32)|
|`requirements.txt`|Python packages for the app|
|`HW\_62.ipynb`|Training notebook (Google Colab)|

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations

* The model always picks one of the 4 dishes — it has no "other" class.
* The 4 dishes look quite different, so the task is easier than fine-grained food classification.
* Single training run; results can vary slightly between runs.

## Credits

* Dataset: Karabay, A., Bolatov, A., Varol, H. A., \& Chan, M.-Y. (2023).
*A Central Asian Food Dataset for Personalized Dietary Interventions.* Nutrients, 15(7), 1728.
Licensed CC BY-NC 4.0 — this project is for non-commercial, educational use.
* Course: Najot Ta'lim Machine Learning, Lesson 62 (Transfer learning).

