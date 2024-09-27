## Ref : https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/resnet50.html#Compile-model-for-Neuron

import torch
from torchvision import models, transforms, datasets
import torch_neuron

# Create an example input for compilation
image = torch.zeros([1, 3, 224, 224], dtype=torch.float32)

# Load a pretrained ResNet50 model
model = models.resnet50(pretrained=True)

# Tell the model we are using it for evaluation (not training)
model.eval()

# Analyze the model - this will show operator support and operator count
torch.neuron.analyze_model(model, example_inputs=[image])

# Compile the model using torch.neuron.trace to create a Neuron model
# that that is optimized for the Inferentia hardware
model_neuron = torch.neuron.trace(model, example_inputs=[image])

# The output of the compilation step will report the percentage of operators that
# are compiled to Neuron, for example:
#
# INFO:Neuron:The neuron partitioner created 1 sub-graphs
# INFO:Neuron:Neuron successfully compiled 1 sub-graphs, Total fused subgraphs = 1, Percent of model sub-graphs successfully compiled = 100.0%
#
# We will also be warned if there are operators that are not placed on the Inferentia hardware

# Save the compiled model
model_neuron.save("resnet50_neuron.pt")
