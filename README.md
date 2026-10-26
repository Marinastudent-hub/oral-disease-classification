# oral-disease-classification
Oral disease image classification with CNN, ResNet50, EfficientNet, DenseNet + Gradio demo
# 🦷 Oral Diseases Image Classification

A deep learning system that classifies intraoral images into 6 categories:
**Calculus, Caries, Gingivitis, Hypodontia, Tooth Discoloration, Ulcer**.
The project includes a custom CNN, three fine-tuned pretrained models, hyperparameter tuning, a comparative analysis, and a Gradio demo app.

> ⚠️ Educational project only. This is not a medical diagnosis tool.

## 🔗 Live demo
[Hugging Face Space](PUT_LINK_HERE)

## Dataset
[Oral Diseases (Kaggle, salmansajid05)](https://www.kaggle.com/datasets/salmansajid05/oral-diseases)
- 11,653 images from the 6 classification folders (the YOLO-annotated folder and the small "original" folders were excluded).
- 1,604 exact duplicate files removed.
- Stratified split 70/15/15: train 7,034 / val 1,507 / test 1,508.
- Class imbalance handled with class weights; evaluation uses accuracy and macro-F1.

## Methods
| Step | Details |
|---|---|
| Custom CNN | 5 conv blocks (Conv-BN-ReLU x2 + MaxPool), global average pooling, dropout |
| Pretrained models | ResNet50, EfficientNet-B0, DenseNet121 (head training, then full fine-tuning) |
| Hyperparameter tuning | Optuna (TPE, 12 trials) on the best pretrained model by validation accuracy (DenseNet121): learning rate, weight decay, batch size, dropout, optimizer, label smoothing |
| Training | Mixed precision, cosine LR schedule, early stopping on validation accuracy |
| Model selection | Highest validation accuracy (the test set is used only for final reporting) |

## Results (test set, 1,508 images)
| Model | Val Acc | Test Acc | Test Macro-F1 | Params (M) | Train time (min) | Inference (ms/img) |
|---|---|---|---|---|---|---|
| Custom CNN | 0.8812 | 0.9052 | 0.8971 | 2.42 | 28.3 | 3.49 |
| ResNet50 | 0.9914 | 0.9954 | 0.9931 | 23.52 | 14.5 | 3.70 |
| EfficientNet-B0 | 0.9934 | 0.9927 | 0.9932 | 4.02 | 14.0 | 3.26 |
| DenseNet121 | 0.9947 | 0.9940 | 0.9929 | 6.96 | 15.4 | 3.87 |
| DenseNet121 (tuned) | 0.9940 | 0.9954 | 0.9912 | 6.96 | 19.6 | 3.90 |

## Key findings
- Fine-tuned pretrained models reached about 99.3–99.5% test accuracy, roughly 9 points above the custom CNN (90.5%), and trained faster (14–15 min vs 28 min).
- The three pretrained models are practically indistinguishable on the 1,508-image test set (differences of 1–2 images). EfficientNet-B0 is the most efficient option (4.0M parameters, 3.3 ms/img).
- Optuna tuning gave no meaningful gain over default settings (val acc 0.9940 vs 0.9947), which suggests a performance ceiling on this dataset.
- The deployed model is the one with the highest validation accuracy: DenseNet121.

## Limitations
- The dataset contains augmented copies of a small number of original images. A random split can place near-identical images in train, validation and test, so the reported accuracy is likely optimistic and may not generalize to images from other sources.
- No external test set and no clinical validation.
- The app always returns one of the 6 classes, even for non-oral images, and its confidence scores can be overconfident.
- Future work: group-aware split or testing on original images only, Grad-CAM explanations, k-fold cross-validation, ensembling.

## Run locally
```bash
pip install -r requirements.txt
python app.py
```
Place the trained checkpoint next to `app.py` as `model.pt`.

## License
Code: MIT. Dataset: see the Kaggle dataset page for its terms.


## Project Screenshots

### Gradio Interface

![Gradio Interface](Screenshot%202026-10-27%20024425.png)

### Model Evaluation

![Model Evaluation](Screenshot%202026-10-27%20024447.png)
