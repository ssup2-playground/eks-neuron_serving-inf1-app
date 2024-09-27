import logging
import time
import json

from PIL import Image
from io import BytesIO
import numpy

import torch
import torch_neuron
from torchvision import transforms

from fastapi import FastAPI, UploadFile

# Init server
app = FastAPI()
model_resnet50 = torch.jit.load("./models/resnet50_neuron.pt")
imagenet_index_name = eval(open("./indexes/imagenet_index_name.txt", "r").read())

# Invoke handlers
@app.post("/resnet50")
def post_invoke(file: UploadFile) -> dict[str, str]:
    # Get PIL image from request
    image = Image.open(BytesIO(file.file.read())).convert("RGB")

    # Get resized and normalized image
    image_transformer = transforms.Compose([
	    transforms.Resize([224, 224]),
	    transforms.ToTensor(),
	    transforms.Normalize(
	        mean=[0.485, 0.456, 0.406],
	        std=[0.229, 0.224, 0.225]
        )
	])
    processed_image = image_transformer(image)
    
    # Append new axis
    input_image = torch.tensor(processed_image.numpy()[numpy.newaxis, ...])

    # Invoke model
    result_model = model_resnet50(input_image)
    result_probability = torch.nn.functional.softmax(result_model, dim=1)
    result_top5_index = reversed(result_model[0].sort()[1][-5:])

    # Return
    result = {}
    for index in result_top5_index:
        result[imagenet_index_name[index]] = str(result_probability[0][index].item())
    return result

# Health handlers
@app.get("/healthz")
def get_heatlhz() -> dict[str, str]:
    logging.info("healthz")
    return {"status": "up"}
