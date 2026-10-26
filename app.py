import torch, torch.nn as nn
from torchvision import models, transforms
import gradio as gr

CKPT = "model.pt"   # the trained checkpoint, placed next to this file
ckpt = torch.load(CKPT, map_location="cpu")
arch, dropout, CLASSES = ckpt["arch"], ckpt["dropout"], ckpt["classes"]
n = len(CLASSES)

def build(arch):
    if arch == "resnet50":
        m = models.resnet50(weights=None); m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, n))
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None); m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, n))
    elif arch == "densenet121":
        m = models.densenet121(weights=None); m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier.in_features, n))
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return m

model = build(arch)
model.load_state_dict(ckpt["state_dict"])
model.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

@torch.no_grad()
def predict(img):
    if img is None:
        return None
    p = torch.softmax(model(tf(img.convert("RGB")).unsqueeze(0)), 1)[0].numpy()
    return {c: float(x) for c, x in zip(CLASSES, p)}

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an oral image"),
    outputs=gr.Label(num_top_classes=6, label="Prediction"),
    title="🦷 Oral Disease Classifier",
    description="For educational purposes only — not a medical diagnosis.",
    flagging_mode="never",
).launch()
