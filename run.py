import logging
import time
import json

from PIL import Image
from io import BytesIO
from typing import List
import numpy

import torch
import torch_neuron
from torchvision import transforms

from fastapi import FastAPI, UploadFile

# Init server
app = FastAPI()

model_resnet50 = torch.jit.load("./models/resnet50_neuron.pt")
model_resnet50_parallel = torch.neuron.DataParallel(model_resnet50)

imagenet_index_name = eval(open("./indexes/imagenet_index_name.txt", "r").read())

# Functions
def preprocess_image(file: UploadFile) -> torch.Tensor:
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

    # Append new axis and return
    return torch.tensor(processed_image.numpy()[numpy.newaxis, ...])


# Invoke handlers
@app.post("/resnet50")
def post_invoke_resnet50(file: UploadFile) -> dict[str, str]:
    # Preprocess images to invoke model
    input_image = preprocess_image(file)

    # Invoke model
    result_model = model_resnet50(input_image)
    result_probability = torch.nn.functional.softmax(result_model[0], dim=0)
    result_top5_indexes = reversed(result_model[0].sort()[1][-5:])

    # Return
    result = {}
    for i in result_top5_indexes:
        result[imagenet_index_name[i]] = str(result_probability[i].item())
    return result

@app.post("/resnet50_batch")
def post_invoke_resnet50_batch(files: List[UploadFile]) -> List[dict[str, str]]:
    # Preprocess and batch images
    input_images = preprocess_image(files[0])
    for i in range(len(files) - 1):
        input_images = torch.cat([input_images, preprocess_image(files[i + 1])], 0)

    # Invoke model
    results_probability = []
    results_top5_indexes = []
    results_model = model_resnet50_parallel(input_images)
    for i in range(len(results_model)):
        results_probability.append(torch.nn.functional.softmax(results_model[i], dim=0))
        results_top5_indexes.append(reversed(results_model[i].sort()[1][-5:]))

    # Return
    results = []
    for i in range(len(results_model)):
        results_dict = {}
        for j in results_top5_indexes[i]:
            results_dict[imagenet_index_name[j]] = str(results_probability[i][j].item())
        results.append(results_dict)
    return results


# Health handlers
@app.get("/healthz")
def get_heatlhz() -> dict[str, str]:
    logging.info("healthz")
    return {"status": "up"}
