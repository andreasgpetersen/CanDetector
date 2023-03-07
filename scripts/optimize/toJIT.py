# Use to convert regular model to Torch Script model via tracing
import torch
import torchvision


# An instance of your model.
model = torchvision.models.resnet18()
model = torch.load("/home/andreasgp/MEGAsync/DTU/9. Semester/Deep Learning/object-tracking-project/02456-project/models/resnet50_4epoch_entire_dataset.pth")
model = model.to("cpu")
# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)


#traced_script_module = torch.jit.script(model)


# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
#traced_script_module = torch.jit.trace(model, example)

# How to calculate output with new model
#output = traced_script_module(torch.ones(1, 3, 224, 224))



# Serialize cript module to file
#traced_script_module.save("traced_resnet_model.pt")
