from dataIO import load_mnist
import os
import numpy as np
import matplotlib.pyplot as plt
from model import * 
from defaults import * 
from PIL import Image
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as functional
from tqdm import tqdm
from tqdm import trange
import imageio

def load_mnist(batch_size, rotate=0, scale=1):
  dataset_transform = transforms.Compose([
               transforms.RandomAffine([rotate, rotate+1], scale=[scale, scale]),
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
           ])
  
  train_dataset = datasets.MNIST('../data', 
                               train=True, 
                               download=True, 
                               transform=dataset_transform)
  test_dataset = datasets.MNIST('../data', 
                                 train=False, 
                                 download=True, 
                                 transform=dataset_transform)


  train_loader = torch.utils.data.DataLoader(train_dataset, 
                                             batch_size=batch_size,
                                             shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, 
                                             batch_size=batch_size,
                                             shuffle=False)
  return train_loader, test_loader
  
egwcaps = EGWCaps(reconstruction_type="Conv")
egwcaps.load_state_dict(torch.load("../saved_models/.."))
egwcaps.cuda()
""
iter(load_mnist(20)[1]).next()[1]

j = 1
for i in tqdm(range(1, 361, 4)):
    _, test_loader = load_mnist(j+1, rotate=0, scale=i/64)
    images, targets = iter(test_loader).next()

    target = targets[j].item()
    output, reconstruction, _ = egwcaps(images.cuda())
    output = torch.norm(output, dim=2).data.squeeze()
    pred = output.squeeze().max(dim=1)[1][j].item()
    im = images[j, 0].data.cpu().numpy()
    rec = reconstruction[j,0].data.cpu().numpy()

    plt.figure(figsize=(20,10))
    plt.subplot(1,3,1)
    plt.title("Confidence")
    plt.ylim([0,1])
    plt.bar(range(0,10), output[j])
    plt.bar(pred, output[j,pred])
    plt.xticks(range(10))
    plt.subplot(1,3,2)
    plt.title("Input Image")
    plt.axis('off')
    plt.imshow(im, cmap="gray")
    plt.subplot(1,3,3)
    plt.title("Reconstructed Image")
    plt.axis('off')    
    plt.imshow(rec, cmap="gray")
    plt.savefig("rotation/test{}.png".format(i))

images = []
for i in trange(1,361,4):
  images.append(imageio.imread("rotation/test{}.png".format(i)))
imageio.mimsave('./movie.gif', images)

j = 1
confidences_correct = []
confidences_correct_i = []
confidences_false = []
confidences_false_i = []
for i in tqdm(range(0, 360, 2)):
  _, test_loader = load_mnist(j+1, rotate=i)
  images, targets = iter(test_loader).next()

  target = targets[j].item()
  output, reconstruction, _ = egwcaps(images.cuda())
  output = torch.norm(output, dim=2)
  pred = output.squeeze().max(dim=1)[1][j].item()
  
  if pred == target:
    confidences_correct.append(output[j,target,0].item())
    confidences_correct_i.append(i)
  else:
    confidences_false.append(output[j,target,0].item())
    confidences_false_i.append(i)
    
# Show Image
_, test_loader = load_mnist(j+1, rotate=0)
images, targets = iter(test_loader).next()
im = images[j, 0].data.numpy()
plt.imshow(im, cmap="gray")

_, test_loader = load_mnist(1+1, rotate=0)
images, targets = iter(test_loader).next()
im = images[1, 0].data.numpy()
plt.axis('off')
plt.imshow(im, cmap="gray")

plt.figure(figsize=(20,10))
plt.plot(confidences_correct_i, confidences_correct, '.')
plt.plot(confidences_false_i, confidences_false, '.')
plt.xlabel("Rotation degrees")
plt.ylabel("Confidence")
plt.xlim([0,360])
plt.ylim([0,1])

