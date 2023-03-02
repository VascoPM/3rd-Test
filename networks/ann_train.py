import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, Activation



def get_initialization (activation_fn):
    # Select right initialization for respective activation function between SELU and LeakyReLU
    if activation_fn.lower() == "selu":
        return 'lecun_normal'
    if activation_fn.lower() == "leakyrelu":
        return 'he_normal'
    
def get_activation_fn (activation_fn, alpha = 0.2):
    # Select right initialization for respective activation function between SELU and LeakyReLU
    if activation_fn.lower() == "selu":
        return Activation('selu')
    if activation_fn.lower() == "leakyrelu":
        return LeakyReLU(alpha=alpha)
    
def get_optimizer(lr, optimizer):
    # Select optmizer between adam and sgd
    if optimizer.lower() == "nadam":
        return tf.keras.optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    if optimizer.lower() == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)