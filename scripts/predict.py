import torch

from scripts.train import get_model_instance_segmentation

model = get_model_instance_segmentation(3)
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode
