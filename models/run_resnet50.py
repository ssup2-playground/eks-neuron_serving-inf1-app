## Ref : https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/resnet50.html#Define-a-preprocessing-function
## Ref : https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/resnet50.html#Run-inference-using-the-Neuron-model
import os

from PIL import Image
import numpy 

import torch
from torchvision import models, transforms, datasets
import torch_neuron

def preprocess(batch_size=1, num_neuron_cores=1):
    # Define a normalization function using the ImageNet mean and standard deviation
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    # Get PIL image from request
    image = Image.open("../images/kitten_small.jpg").convert("RGB")

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
    return input_image

# Get a sample image
image = preprocess()

# Get Index
index_name = eval(open("../indexes/imagenet_index_name.txt", "r").read())

# Run inference using the CPU model
model = models.resnet50(pretrained=True)
model.eval()
output_cpu = model(image)

# Run inference using the Neuron model
model_neuron = torch.jit.load('resnet50_neuron.pt')
output_neuron = model_neuron(image)

# Verify that the CPU and Neuron predictions are the same by comparing
# the top-5 results
top5_cpu = output_cpu[0].sort()[1][-5:]
top5_neuron = output_neuron[0].sort()[1][-5:]

# Lookup and print the top-5 labels
top5_labels_cpu = [index_name[idx] for idx in top5_cpu]
top5_labels_neuron = [index_name[idx] for idx in top5_neuron]
print("CPU top-5 labels: {}".format(top5_labels_cpu))
print("Neuron top-5 labels: {}".format(top5_labels_neuron))
