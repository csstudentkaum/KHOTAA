import io
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(title="DFU Classification API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Model setup ───────────────────────────────────────────────────────────
MODEL_PATH = "efficientnetv2s_final_model.pt"
CLASS_NAMES = ["Both", "Infection", "Ischaemia", "None"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    head_config = checkpoint["head_config"]
    num_classes = checkpoint["num_classes"]

    base = efficientnet_v2_s(weights=None)
    in_features = base.classifier[1].in_features

    layers = [nn.Dropout(head_config["dropout"])]
    prev = in_features
    for size in head_config["hidden_sizes"]:
        layers.append(nn.Linear(prev, size))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(head_config["dropout"]))
        prev = size
    layers.append(nn.Linear(prev, num_classes))

    base.classifier = nn.Sequential(*layers)
    base.load_state_dict(checkpoint["model_state_dict"])
    base.to(DEVICE)
    base.eval()
    return base


model = load_model()

# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "EfficientNetV2S", "classes": CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(400, "Upload a JPEG, PNG, or WebP image.")

    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = probs.argmax().item()

    return {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": round(probs[pred_idx].item(), 4),
        "probabilities": {
            name: round(prob, 4)
            for name, prob in zip(CLASS_NAMES, probs.tolist())
        },
    }
