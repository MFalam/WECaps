import os.path as path
import time
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from defaults import *
from dataIO import *
from track import *
from base import *
from model import EGWCaps
from options import create_options
from tqdm import tqdm
from EGW import *

print(torch.__version__)
def get_alpha(epoch):
    # WARNING: Does not support alpha value saving when continuning training from a saved model
    if opts.anneal_alpha == "none":
        alpha = opts.alpha
    if opts.anneal_alpha == "1":
        alpha = opts.alpha * float(np.tanh(epoch/ANNEAL_TEMPERATURE - np.pi) + 1) / 2
    if opts.anneal_alpha == "2":
        alpha = opts.alpha * float(np.tanh(epoch/(2 * ANNEAL_TEMPERATURE)))
    return alpha

def onehot(tensor, num_classes=10):
    return torch.eye(num_classes).cuda().index_select(dim=0, index=tensor) # One-hot encode 

def transform_data(data,target,use_gpu, num_classes=10):
    data, target = Variable(data), Variable(target)
    if use_gpu:
        data, target = data.cuda(), target.cuda()
    target = onehot(target, num_classes=num_classes)
    return data, target

class GPUParallell(nn.DataParallel):
  
  def __init__(self, egwcaps, device_ids):
    super(Test, self).__init__(egwcaps, device_ids=device_ids)
    self.egwcaps = egwcaps
    self.num_classes = egwcaps.num_classes
    
  def loss(self, images,labels, capsule_output,  reconstruction): 
    return self.egwcaps.loss(images, labels, capsule_output, reconstruction)
  
  def forward(self, x, target=None):
    return self.egwcaps(x, target)

def get_network(opts):
    if opts.dataset == "mnist":
        egwcaps = EGWCaps(reconstruction_type=opts.decoder,
                          routing_iterations = opts.routing_iterations,
                          batchnorm=opts.batch_norm,
                          loss=opts.loss_type,
                          leaky_routing=opts.leaky_routing)
    if opts.dataset == "fashionmnist":
        egwcaps = EGWCaps(reconstruction_type=opts.decoder,
                          routing_iterations = opts.routing_iterations,
                          batchnorm=opts.batch_norm,
                          loss=opts.loss_type,
                          leaky_routing=opts.leaky_routing)            
    if opts.dataset == "small_norb":
        if opts.decoder == "Conv":
            opts.decoder = "Conv32"
        egwcaps = EGWCaps(reconstruction_type=opts.decoder,
                          imsize=32,
                          num_classes=5,
                          routing_iterations = opts.routing_iterations, 
                          primary_caps_gridsize=8,
                          num_primary_capsules=32,
                          batchnorm=opts.batch_norm,
                          loss = opts.loss_type,
                          leaky_routing=opts.leaky_routing)
    if opts.dataset == "cifar10":
        if opts.decoder == "Conv":
            opts.decoder = "Conv32"
        egwcaps = EGWCaps(reconstruction_type=opts.decoder,
                          imsize=32, 
                          routing_iterations = opts.routing_iterations,
                          primary_caps_gridsize=8,
                          img_channels=3, 
                          batchnorm=opts.batch_norm,
                          num_primary_capsules=32,
                          loss=opts.loss_type,
                          leaky_routing=opts.leaky_routing)
    if opts.dataset == "affnist":
        if opts.decoder == "Conv":
            opts.decoder = "Conv32"
        egwcaps = EGWCaps(reconstruction_type=opts.decoder,
                          imsize=32, 
                          routing_iterations = opts.routing_iterations,
                          primary_caps_gridsize=8,
                          img_channels=3, 
                          batchnorm=opts.batch_norm,
                          num_primary_capsules=32,
                          loss=opts.loss_type,
                          leaky_routing=opts.leaky_routing)    
    if opts.use_gpu:
        egwcaps.cuda()
    if opts.gpu_ids:
        egwcaps = GPUParallell(egwcaps, opts.gpu_ids)
        print("Training on GPU IDS:", opts.gpu_ids)
    return egwcaps

def load_model(opts, egwcaps): 
    model_path = path.join(SAVE_DIR, opts.filepath)
    if path.isfile(model_path):
        print("Saved model found")
        egwcaps.load_state_dict(torch.load(model_path))
    else:
        print("Saved model not found; Model initialized.")
        initialize_weights(egwcaps)
    

def get_dataset(opts):
    if opts.dataset == 'mnist':
        return load_mnist(opts.batch_size)
    if opts.dataset == 'fashionmnist':
        return load_fashionmnist(opts.batch_size)    
    if opts.dataset == 'small_norb':
        return load_small_norb(opts.batch_size)
    if opts.dataset == 'cifar10':
        return load_cifar10(opts.batch_size)
    if opts.dataset == 'affnist':
        return load_affnist(opts.batch_size)    
    raise ValueError("Dataset not supported:" + opts.dataset)
    

def main(opts):
    egwcaps = get_network(opts)

    optimizer = torch.optim.Adam(egwcaps.parameters(), lr=opts.learning_rate)

    """ Load saved model"""
    load_model(opts, egwcaps)

    train_loader, valid_loader, test_loader = get_dataset(opts)
    stats = Statistics(LOG_DIR, opts.model)
    
    for epoch in range(opts.epochs):
        egwcaps.train()
        
        # Annealing alpha
        alpha = get_alpha(epoch)

        for batch, (data, target) in tqdm(list(enumerate(train_loader)), ascii=True, desc="Epoch{:3d}".format(epoch)):
            optimizer.zero_grad()
            data, target = transform_data(data, target, opts.use_gpu, num_classes=egwcaps.num_classes)

            capsule_output, reconstructions, _ = egwcaps(data, target)
            predictions = torch.norm(capsule_output.squeeze(), dim=2)
            data = denormalize(data)
            loss, rec_loss, marg_loss = egwcaps.loss(data, target, capsule_output, reconstructions, alpha)
            loss.backward()
            optimizer.step()
            
            stats.track_train(loss.data.detach().item(), rec_loss.detach().item(), marg_loss.detach().item(), target.detach(), predictions.detach())
        
        """Evaluate on test set"""
        egwcaps.eval()
        for batch_id, (data, target) in tqdm(list(enumerate(test_loader)), ascii=True, desc="Test {:3d}".format(epoch)):
            data, target = transform_data(data, target, opts.use_gpu, num_classes=egwcaps.num_classes)

            capsule_output, reconstructions, predictions = egwcaps(data)
            data = denormalize(data)
            loss, rec_loss, marg_loss = egwcaps.loss(data, target, capsule_output, reconstructions, alpha)


            stats.track_test(loss.data.detach().item(),rec_loss.detach().item(), marg_loss.detach().item(), target.detach(), predictions.detach())

        stats.save_stats(epoch)

        # Save reconstruction image from testing set
        if opts.save_images:
            data, target = iter(test_loader).next()
            data, _ = transform_data(data, target, opts.use_gpu)
            _, reconstructions, _ = egwcaps(data)
            filename = "reconstruction_epoch_{}.png".format(epoch)
            if opts.dataset == 'cifar10':
                save_images_cifar10(IMAGES_SAVE_DIR, filename, data, reconstructions)
            else:
                save_images(IMAGES_SAVE_DIR, filename, data, reconstructions, imsize=egwcaps.imsize)

        # Save model
        model_path = get_path(SAVE_DIR, "model{}.pt".format(epoch))
        torch.save(egwcaps.state_dict(), model_path)
        egwcaps.train()


if __name__ == '__main__':
    opts = create_options()
    main(opts)
