# Importing the Libraries
import numpy as np
import argparse
import time
from PIL import Image
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import keras.backend as K
from keras.applications import VGG16


# Command Line Arguments
parser=argparse.ArgumentParser(description='Neural Style Transfer')
parser.add_argument('content_img', metavar='content', type=str, help='Content Image Path')
parser.add_argument('style_img', metavar='style', type=str, help='Style Image Path')
parser.add_argument('result_img_name', metavar='res_prefix', type=str, help='Generated Image Path')
parser.add_argument('--epoch', type=int, default=50, required=False, help='Number of Epochs')
parser.add_argument('--content_weight', type=float, default=0.025, required=False, help='Content Weight')
parser.add_argument('--style_weight', type=float, default=1.0, required=False, help='Style Weight')
parser.add_argument('--height', type=int, default=512, required=False, help='Image Height')
parser.add_argument('--width', type=int, default=512, required=False, help='Image Width')
args = parser.parse_args()



# Setting default parameters
imageHeight=args.height
imageWidth=args.width
imageSize=imageHeight*imageWidth
imageChannels=3
content_path=args.content_img
style_path=args.style_img
target_name=args.result_img_name
target_extension='.png'





def preprocessImage(imagePath):
    
    VGG_mean=[103.939, 116.779, 123.89] # VGG16 mean values
    image=Image.open(imagePath) # Opening the image 
    image=image.resize((imageWidth, imageHeight)) # Resizing the image
    imageArray=np.asarray(image, dtype='float32') # Converting the image to an array
    imageArray=np.expand_dims(imageArray, axis=0) # Adding an extra column
    '''
    first column
    0 content
    1 style
    2 generated
    '''
    imageArray=imageArray[:, :, :, :3] # Taking R,G,B
    imageArray=imageArray[:, :, :, ::-1] # Converting RGB to BGR
    imageArray[:, :, :, 0]-=103.939 # Mean shifting
    imageArray[:, :, :, 1]-=116.779
    imageArray[:, :, :, 2]-=123.68
    return imageArray


def get_layers(content_matrix, style_matrix, generated_matrix):
    
    input=K.concatenate([content_matrix, style_matrix, generated_matrix], axis=0) # Merging all three arrays into one
    model=VGG16(input_tensor=input, weights='imagenet', include_top=False) # VGG16 model excluding the FC layers
    layers=dict([(layer.name, layer.output) for layer in model.layers])  # Dictionary of all the layers

    c_layer_name=['block2_conv2']
    s_layer_name=['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
    c_layer_output=[layers[layer] for layer in c_layer_name]
    s_layer_output=[layers[layer] for layer in s_layer_name]

    return c_layer_output, s_layer_output


def content_loss(content_features, generated_features):
    
    return (1.0/2.0)*K.sum(K.square(generated_features-content_features)) # Formula for Content Loss


def gram_matrix(features):
    
    return K.dot(features, K.transpose(features)) # Gram matrix = multiplication of feature map and its transpose


def style_loss(style_matrix, generated_matrix):
    
    style_features=K.batch_flatten(K.permute_dimensions(style_matrix, (2, 0, 1))) 
    generated_features=K.batch_flatten(K.permute_dimensions(generated_matrix, (2, 0, 1)))

    style_gram_matrix=gram_matrix(style_features)
    generated_gram_matrix=gram_matrix(generated_features)

    return K.sum(K.square(style_gram_matrix-generated_gram_matrix))/(4.0*(imageChannels**2)*(imageSize**2))


def total_loss(c_layers, s_layers, generated):
    
    content_weight=args.content_weight # content weight
    style_weight=args.style_weight # style weight
    
    c_loss=0
    for layer in c_layers:
        content_features=layer[0, :, :, :] 
        generated_features=layer[2, :, :, :]
        c_loss+=content_loss(content_features, generated_features)

    s_loss=0
    for layer in s_layers:
        style_features=layer[1, :, :, :]
        generated_features=layer[2, :, :, :]
        s_loss+=style_loss(style_features, generated_features)*(1.0/len(s_layers))
    
    return content_weight*c_loss + style_weight*s_loss 


def loss_and_gradient(generatedImageArray):
    
    generatedImageArray=generatedImageArray.reshape((1, imageHeight, imageWidth, 3)) # Reshape
    output=finalOutput([generatedImageArray])
    loss_value=output[0]
    gradient_values=output[1].flatten().astype('float64')
    return loss_value, gradient_values


def save_image(filename, generatedImage):
    
    generatedImage=generatedImage.reshape((imageHeight, imageWidth, 3)) # Reshaping the matrix .i.e. removing the first column
    
    generatedImage[:, :, 0]+=103.939 # Re-applying mean shift
    generatedImage[:, :, 1]+=116.779
    generatedImage[:, :, 2]+=123.68

    generatedImage=generatedImage[:, :, ::-1] # Converting BGR to RGB
    generatedImage=np.clip(generatedImage, 0, 255).astype('uint8') # Restricting all values to 0-255

    imsave(filename, Image.fromarray(generatedImage)) # Converting array to image and saving it


class Evaluator(object):
    
    def __init__(self): # Constructor
        self.loss_value=0
        self.gradient_values=0

    def loss(self, x):
        loss_value, gradient_values=loss_and_gradient(x)
        self.loss_value=loss_value
        self.grad_values=gradient_values
        return self.loss_value

    def grads(self, x):
        gradient_values=np.copy(self.gradient_values)
        
        '''
        if we use gradient_values=self.gradient_values,
        then any changes made to self.gradient_values will also effect gradient_values.
        gradient_values=np.copy(self.gradient_values) allocates a separate memory location and names it gradient_values.
        '''
        
        self.loss_value=0
        self.gradient_values=0
        return gradient_values


if __name__ == '__main__':
 
    generated_image=np.random.uniform(0, 255, (1, imageHeight, imageWidth, 3)) # Random noise for first generated image
    name='{}-{}{}'.format(target_name, 0, target_extension)
    save_image(name, generated_image)

    contentArray=preprocessImage(content_path) # Storing processed content image array
    styleArray=preprocessImage(style_path) # Storing processed style image array

    contentImage=K.variable(contentArray) # Converting numpy array to Keras variable
    styleImage=K.variable(styleArray) # Converting numpy array to Keras variable
    generatedImage=K.placeholder((1, imageHeight, imageWidth, 3)) # Creating an array for generated image
    loss=K.variable(0) # Creating a Loss Keras variable with intial value as 0

    content_layers, style_layers=get_layers(contentImage, styleImage, generatedImage) # Getting the content and style layers
 
    loss=total_loss(content_layers, style_layers, generatedImage) # Computing total loss
    grads=K.gradients(loss, generatedImage) # Computing the gradient

    output=[loss] # Loss list
    output+=grads # Adding the gradient tensor to the loss
    finalOutput=K.function([generatedImage], output) # Function name is finalOutput with parameter as generatedImage 

    evaluator=Evaluator() 
    epochs=args.epoch # Storing number of epochs

    

    for i in range(epochs):
        print('Epoch:', i)
        generated_image, min_val, _ = fmin_l_bfgs_b(evaluator.loss, generated_image, fprime=evaluator.grads, maxfun=20)
        print('Loss:', min_val)
        name='{}-{}{}'.format(target_name, i+1, target_extension)
        save_image(name, generated_image)
