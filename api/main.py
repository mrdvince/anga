from fastapi import FastAPI, UploadFile, File
import torch
import model.model as module_arch
from parse_config import ConfigParser
from PIL import Image
from io import BytesIO
from torchvision import transforms
import numpy as np
import os
import json
from pydantic import BaseModel


app = FastAPI(
    title="Plant leaf disease classifier",
    description="""
Given a plant leaf photo, 
return check the type of disease it has
(limited to the 38 classes trained on). 
Model used is a pretrained efficientnet b0, mixnet medium is also available
> Note: the dataset has __38 classes__ but some of the classes are healthy plant leafs
### Disease types
- Apple Frogeye Spot
- Apple Scab
- Cedar Apple Rust
- Cherry Powdery Mildew
- Maize Cercospora Leaf Spot
- Maize Common Rust
- Maize Northern Leaf Blight
- Grape Black Rot
- Grape Esca Black Measles
- Grape Isariopsis Leaf Spot
- Orange Haunglongbing
- Peach Bacterial Spot
- Pepper Bacterial Spot
- Potato Early Blight
- Potato Late Blight
- Squash Powdery Mildew
- Strawberry Leaf Scorch
- Tomato Bacterial Spot
- Tomato Early Blight
- Tomato Late Blight
- Tomato Leaf Mold
- Tomato Septoria Leaf Spot
- Tomato Two Spotted Spider Mite
- Tomato Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato Mosaic Virus
    """,
    docs_url="/playground",
    redoc_url="/docs",
)


def read_imagefile(data) -> Image.Image:
    image = Image.open(BytesIO(data)).convert("RGB")
    return image


def load_model():
    config = ConfigParser.from_config(config="config.json")
    model = config.init_obj("arch", module_arch)
    checkpoint = torch.load("saved/model_best.pth", map_location="cpu")
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model = model.to("cpu")
    return model.eval()


def predict(bytes):
    img = read_imagefile(bytes.file.read())
    trfms = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(224)])
    model = load_model()
    output = model(trfms(img).unsqueeze(0))
    _, pred_tensor = torch.max(output, 1)

    preds = np.squeeze(pred_tensor.numpy())
    return preds


def get_classes():
    with open(
        os.path.join(os.path.dirname(__file__), "../", "data", "cat_to_name.json"), "r"
    ) as f:
        cats = json.loads(f.read())
    return cats


@app.post("/", tags=["Post image"])
def get_predictions(file: UploadFile = File(...)):
    """
    Upload an image file  and get a prediction/classification (whatever you want to call it)

    - **upload file**: the intended image file
    """
    pred = predict(file)
    cats = get_classes()
    return {"index": int(pred), "label": cats[str(pred)]}
