# 🧠 Brain Tumor MRI Image Classification

Deep-learning pipeline that classifies brain MRI images into tumor types (e.g. glioma, meningioma, pituitary, no tumor):
a **custom CNN built from scratch**, **transfer learning** with pretrained ImageNet models (MobileNetV2, ResNet50V2,
EfficientNetB0, optional InceptionV3), a model comparison, and a **Streamlit app** for real-time predictions.

> ⚠️ Educational decision-support prototype. **Not a medical device** and not for diagnosis.

## Project structure
```
app.py                    Streamlit web app (upload MRI -> tumor type + confidence)
src/data.py               scan/clean dataset (drop corrupt files + duplicates), stratified split, tf.data pipeline
src/models.py             custom CNN + transfer-learning models, augmentation, fine-tuning helper
src/train.py              train -> checkpoint (.h5) -> evaluate -> compare -> export best model
src/evaluate.py           metrics, confusion matrix, training curves, comparison chart
notebooks/train_on_colab.ipynb        run everything on a free Colab GPU using the dataset in Google Drive
notebooks/Brain_Tumor_MRI_Capstone.ipynb   full capstone notebook (EDA with 15 charts, hypothesis tests, 3 tuned models, Grad-CAM)
models/                   trained .h5 models + class_names.json (created by training)
results/                  metrics, confusion matrices, history plots, model_comparison.csv/png
```

## Dataset
*Brain Tumor MRI Multi-Class Dataset.* The **parent folder of each image is its class**:
```
Tumour/Training/glioma/*.jpg ...  Tumour/Testing/glioma/*.jpg ...      (or simply Tumour/<class>/*.jpg)
```
Corrupt files and exact duplicates are removed automatically; if the dataset has a `Testing` folder it is used as the
test set, otherwise a stratified 70/15/15 split is made. Validation is a stratified 15 % slice of the training data.

## Workflow (matches the project brief)
1. **Understand data** – class counts/imbalance, resolutions, sample images (capstone notebook).
2. **Preprocessing** – images resized to 224x224. The custom CNN normalises to **0-1** (`Rescaling(1/255)`); pretrained
   models use the scaling their ImageNet weights expect ([-1, 1] or raw 0-255), done *inside* the model so the app needs no extra code.
3. **Augmentation** – rotation, horizontal + vertical flip, zoom, brightness, shifts (training only).
4. **Custom CNN** – 4 blocks of (Conv-BatchNorm-ReLU)x2 + MaxPool + Dropout, global average pooling, dense head.
5. **Transfer learning** – frozen ImageNet backbone + new dense head, then fine-tuning of the top 30 layers at a low learning rate.
   (ResNet**50V2** is used because its preprocessing can be saved inside a portable `.h5`; it is the improved ResNet50.)
6. **Training** – `EarlyStopping`, `ModelCheckpoint` (best `val_loss` -> `models/<name>.h5`), `ReduceLROnPlateau`, class weights.
7. **Evaluation** – accuracy, macro precision/recall/F1, confusion matrix, accuracy/loss curves.
8. **Comparison** – accuracy + F1 + parameters + file size + inference time per image; best model chosen on **validation macro-F1** and copied to `models/best_model.h5`.
9. **Deployment** – Streamlit app.

## Quick start
### A) Colab + Google Drive (recommended – needs a GPU)
1. Put this folder and the dataset in your Drive, open `notebooks/train_on_colab.ipynb` in Colab, set `PROJECT` / `DATA_SRC`, choose a GPU runtime, *Run all*.

### B) Local / any machine
```bash
pip install -r requirements.txt
python -m src.train --data_dir path/to/Tumour --epochs 25 --ft_epochs 10      # GPU strongly recommended
streamlit run app.py
```
Useful flags: `--models custom_cnn,mobilenetv2,resnet50v2,efficientnetb0,inceptionv3`, `--img_size 224`, `--batch 32`.
Fast CPU smoke test (no ImageNet weights): `python -m src.train --data_dir data --img_size 64 --epochs 1 --ft_epochs 1 --no_pretrained`.

## Results
Run training, then paste `results/model_comparison.csv` here:

| model | accuracy | precision | recall | F1 | params (M) | size (MB) | ms / image |
|---|---|---|---|---|---|---|---|
| custom_cnn | | | | | | | |
| mobilenetv2 | | | | | | | |
| resnet50v2 | | | | | | | |
| efficientnetb0 | | | | | | | |

## Deliverables checklist
- [x] Custom CNN and pretrained models saved as `.h5` (`models/`)
- [x] Streamlit application (`app.py`)
- [x] Training / evaluation / deployment scripts + notebooks
- [x] Model comparison (`results/model_comparison.*`)
- [x] README, modular commented code

## Notes
- `.h5` is Keras' legacy format (requested by the brief); the app also accepts `.keras` files. Loading a saved model needs `import src.models` first (registers the custom brightness layer) – `app.py` already does this.
- Trained `.h5` files are large; share them through GitHub Releases / Git LFS rather than normal commits.
