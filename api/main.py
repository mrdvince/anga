from fastapi import FastAPI, UploadFile, File
import torch
import model.model as module_arch
from parse_config import ConfigParser
from PIL import Image
from io import BytesIO
from torchvision import transforms
import numpy as np

app = FastAPI(
    title="Plant leaf disease classifier",
    description="Given a plant leaf photo, return check the type of disease it has, (limited to the 38 classes trained on). Model used is a pretrained efficientnet b0, mixnet medium is also available",
    docs_url="/playground",
    redoc_url="/docs",
)


def read_imagefile(data) -> Image.Image:
    image = Image.open(BytesIO(data))
    return image


def load_model():
    config = ConfigParser.from_config(config="config.json")
    model = config.init_obj("arch", module_arch)
    checkpoint = torch.load("saved/model_best.pth", map_location="cpu")
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model = model.to("cpu")
    model.eval()
    return model


def predict(bytes):
    img = read_imagefile(bytes.file.read())
    trfms = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(224)])
    tensor_image = trfms(img)
    model = load_model()
    output = model(tensor_image.unsqueeze(0))
    y, pred_tensor = torch.max(output, 1)
    preds = np.squeeze(pred_tensor.numpy())
    return preds


@app.post("/")
def get_predictions(file: UploadFile = File(...)):
    x = predict(file)
    print(x)
    preds = "leaf"
    return {"predicted": preds}
