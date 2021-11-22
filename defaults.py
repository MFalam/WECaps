import os

# Default values for command arguments
LEARNING_RATE = 0.001 # The learning rate reduces to 0.0005
DATASET = "mnist" # 'mnist', 'fashionmnist', 'small_norb', 'cifar10'
DECODER = "Conv" 
ANNEAL_TEMPERATURE = 8 
ALPHA = 0.0005 
BATCH_SIZE = 128
EPOCHS = 100
USE_GPU = True
ROUTING_ITERATIONS = 3
VALIDATION_SIZE = 1000
# models directory
SAVE_DIR = "models"
# plots directory
PLOT_DIR = "plots"
# logs directory
LOG_DIR = "logs"
# options directory
OPTIONS_DIR = "options"
# reconstracted images directory
IMAGES_SAVE_DIR = "reconstruct-images"
# Random seed for validation split
VALIDATION_SEED = 978371396
# smallNorb Dataset directort
SMALL_NORB_PATH = os.path.join("datasets", "smallNORB")
