[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

# Deep Learning Project

In this project, we have trained a Fully Convolutional Network to learn from the images generated from the quadsim simulator. 

[image_0]: ./docs/misc/sim_screenshot.png
![alt text][image_0] 

## Instructions to run the project
* Clone this repository
```
$ git clone https://github.com/saurabdixit/RoboND-DeepLearning-Project.git
```
* Download following data and save the unziped version in the data folder.
[Training Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip) 

[Validation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip)

[Sample Evaluation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip)

* Make sure that you have RoboND conda environment and tensorflow installed. Navigate to the code directory, activate your conda environment, and open the jupyter notebook.
```
conda activate RoboND
jupyter notebook model_training_final.ipynb
```
* Run all cells. I have commented out the FCN training cell and model saving cell. So, it should pick up existing trained model and provide you the result.
* Thanks for running the project! You should see following result at the end.

## Network Architecture
In this project, we have used tensorflow's library to implement fully convolutional neural network. Our FCN has following components:
1. Encoder block
2. Decoder block
3. 1x1 convolution layer

## Image for encoder, 1x1 convolution, and decoder block

Following section will cover above components in more details.

### Encoder block
In simple words, Encoder converts an input image to high dimensional feature vector. For example: If you are processing an image of a golden retriver dog, it might contain features like Fur, eye-ball, tongue, teeth etc. 

Here is my implementation of encoder layer
```python
def encoder_block(input_layer, filters, strides=1):
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters=filters, strides=strides)
    return output_layer

```

Using above function you can add multiple convolutional layer. Later section will discuss how many convolutional layer I am using in my FCN model.
The depth of the convolutional layer can be controlled by the filters parameter and strides controls the size of filter. Following is the correlation:
1. filters: depth. Eg: If you have 32 filters, depth will be 32. Hence, in other words. There is 1 to 1 correlation
2. strides: controls the output layer size. Eg: If you are working with input of size 160x160 and stride is set to 2, your output layer size will be 80x80. That doesn't mean that output_layer_size = input_layer_size/2. It depends on kernel_size, and padding. In our case, we are using stride of 1 and 2 only. Hence, the output layer size is 1/2 of input layer size.

Note that More deep the layer is more will be the segmentation. Neural network will decide what part of the image will be monitored by which neuron.

### Decoder block
As the name indicates, Decoder's functionality is reverse than that of Encoder. It takes high dimensional feature vector and creates a segmentation mask on the image. 
For example: If we are looking at the image taken from the car and we want to identify what components are present where. The encoder block will just answer the question that there is a traffic light in the image, there is a person standing in the image, there is a road in the image etc. However, in order to make intelligent decisions, we have to understand where the person is in respect to road, is he/she in the middle of road etc.

The Decoder block resolve our issue by providing us answers to above question. Following is my implementation of Decoder block

```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    Upsampled_small_ip_layer = bilinear_upsample(small_ip_layer)

    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    concatenated_layers = layers.concatenate([Upsampled_small_ip_layer,large_ip_layer])

    # TODO Add some number of separable convolution layers
    output_layer = encoder_block(concatenated_layers,filters=filters, strides=1)
    return output_layer

```
Above function fullfills three purposes as mentioned in the model_training jupyter notebook:
1. Bilinear upsampling 
2. Layer concatenation of both the layers
3. Feature extraction from concatenation layer



### 1x1 convolution layer
1x1 convolution layer is used between encoder and decoder. As discussed in the lecture, if you use this in the middle of encoder and decoder, it will act as a mini-neural network running on the patch instead of linear classifier. It is a way to make the model more deeper and have more parameters without any additional cost.

In FCN model, I am using 1x1 convolution using conv2d_batchnorm function provided in the model_training notebook.


## Fully Convolutional Model
I tried many combinations of encoder and decoder block. There are many problems that I faced because of which I had to go with many convolutional layers. I am discussing those problem in next section.

Following implementation worked best for me:

```python
def fcn_model(inputs, num_classes):
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    #Layer 1
    layer_1 = encoder_block(inputs, filters=32, strides=2)

    #Layer 2
    layer_2 = encoder_block(layer_1, filters=64, strides=2)

    #Layer 3
    layer_3 = encoder_block(layer_2, filters=128, strides=2)

    #Layer 4
    layer_4 = encoder_block(layer_3, filters=256, strides=2)

    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    layer_1x1 = conv2d_batchnorm(layer_4, filters=1028, kernel_size=1, strides=1)    

    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks

    # Decoder 1
    decoder_1 = decoder_block(layer_1x1, layer_3, filters=128)

    # Decoder 2
    decoder_2 = decoder_block(decoder_1, layer_2, filters=64)

    # Decoder 3
    decoder_3 = decoder_block(decoder_2, layer_1, filters=32)

    # Last decoder block
    x = decoder_block(decoder_3, inputs, filters=num_classes)    

    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(x)
```
Following is the shape of the input, all layers, and output:
1.  Input:     (?, 160, 160, 3)
2.  Layer 1:   (?, 80, 80, 32)
3.  Layer 2:   (?, 40, 40, 64)
4.  Layer 3:   (?, 20, 20, 128)
5.  Layer 4:   (?, 10, 10, 256)
6.  1x1 conv:  (?, 10, 10, 1024)
7.  Decoder 1: (?, 20, 20, 128)
8.  Decoder 2: (?, 40, 40, 64)
9.  Decoder 3: (?, 80, 80, 32)
10. Decoder 4: (?, 160, 160, 3)
11. Output:    (?, 160, 160, 3)

In next section, I am discussing about my previous trials and what went wrong.

# Problem I ran into, previous trials and comparison with current model.
## Problems I ran into
I know I should be discussing this section at the end. However, I would like to discuss it here because there were some computation concerns I ran into. Hence, I chose the above model.
* I am submitting this project at the last moment because I had some interviews in past two months.
* I started this project 2 days back and didn't get much time to make it to AWS 48 hours setup time.
* I tried to use my local NVIDIA GPU to run tensorflow. However, tensorflow-gpu library was giving me ton of errors which I tried to fix but was not successful.
* The only way for me was to run it slowly on my computer.
* What I did was uploaded the model_training.ipynb to the segmentation lab folder and tested my parameters there. Initially, I thought of training the FCN on Udacity's GPU enabled workspace, downloading the model_weights file locally, and then verifying the accuracy on that. For some reason, I did not get any outputs from it. I am not sure why? If possible, can reviewer answer my question?
* Anyways, so I tried multiple parameters on GPU enabled workspace, used the parameters locally for training, and was able to train the model using parameters mentioned in following section.









