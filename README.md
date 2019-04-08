# Resnet_Object_Detection
Use an 18 layer resnet model to detect an object of interest in an image.

## Approach
A resnet18 model takes an image as input and outputs 4 values representing the coordinates of the top left and bottom right corners of the bounding box, around the object of interest. 
Note: This is a simple model. It does not handle cases with multiple or no objects in an image.
