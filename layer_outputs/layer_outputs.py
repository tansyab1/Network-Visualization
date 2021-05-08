#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import cv2, torch
import numpy as np
from torchvision import models
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from matplotlib import pyplot as plt


#------------------------------------------------------------------------------
#	Main execution
#------------------------------------------------------------------------------
# Create model
model = models.vgg16_bn(pretrained=True)
model.eval()
model.cuda()
print(model)

# Register
registers = []
hook = lambda *x: registers.append(x[2])
model.features[12].register_forward_hook(hook)

# Read and Pre-process an image
img = cv2.imread("../images/dog.jpg")[...,::-1]
image = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
X = transforms.functional.to_tensor(image)
X = transforms.functional.normalize(X, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
X = torch.unsqueeze(X, dim=0)
X = X.cuda()

# Visualize
model(X)
output = registers[0][0].detach().cpu().numpy()
for i in range(output.shape[0]):
	plt.subplot(8, int(output.shape[0]/8), i+1)
	plt.subplots_adjust(wspace=.001)
	plt.axis('off')
	plt.imshow(output[i])
plt.show()