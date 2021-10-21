# Lecture 9

# CNN Architecture case study

In this lecture, we want to investigate successful network architectures and their evolution.

## LeNet-5

LeNet was one of the first CNNs that performed very well. The network is applied to the recognition of handwritten zip code digits.

The architecture of the network is $Conv \to Pool \to Conv \to Pool \to FC \to FC \to FC$

![https://miro.medium.com/max/2000/1*1TI1aGBZ4dybR6__DI9dzA.png](https://miro.medium.com/max/2000/1*1TI1aGBZ4dybR6__DI9dzA.png)

### AlexNet (2012)

AlexNet is the first large-scale CNN used for image classification. It enters the Image-Net competition and dominates the competitors with a high margin. The overall architecture of the network is similar to LeNet.

Relu activation function was first used in AlexNet.

![https://miro.medium.com/max/1375/1*arJJYgK-_7VcuKKX1TCKMA.png](https://miro.medium.com/max/1375/1*arJJYgK-_7VcuKKX1TCKMA.png)

### ZFNet (2013)

ZFNet won the prize of Image-Net 2013. the architecture is very similar to the AlexNet. Just changed some hyperparameters, like the number of filters in each layer and the stride length.

![https://pechyonkin.me/images/201901-zfnet/zfnet.png](https://pechyonkin.me/images/201901-zfnet/zfnet.png)

### VGGNet (2014)

VGGNet used a much smaller filter and a much deeper network compared to previous networks. Several Conv layers in the row with smaller filters have the same effective field as a single layer with a bigger filter, but the firsts have more non-linearities and fewer parameters.

VGGNet needs about 96 MB of memory in each forward pass per image that is too much and hardly scaled for larger input images. Also, the number of parameters is about 138 M, and most of the parameters reside in the last FC layers.

The second FC layer generalizes well for other tasks and could be used for transfer learning. 

![https://neurohive.io/wp-content/uploads/2018/11/vgg16.png](https://neurohive.io/wp-content/uploads/2018/11/vgg16.png)

### GoogLeNet (2014)

The main feature of GoogLeNet was network Inception modules that dramatically reduced the network parameter (12X less than AlexNet). Also, FC layers didn't connect to Conv layers directly. Although this network has 22 layers, it's efficient.

![https://www.researchgate.net/profile/Bo-Zhao-67/publication/312515254/figure/fig3/AS:489373281067012@1493687090916/nception-module-of-GoogLeNet-This-figure-is-from-the-original-paper-10_W640.jpg](https://www.researchgate.net/profile/Bo-Zhao-67/publication/312515254/figure/fig3/AS:489373281067012@1493687090916/nception-module-of-GoogLeNet-This-figure-is-from-the-original-paper-10_W640.jpg)

Inception module

To solve the vanishing gradient problem in GoogLeNet, They add some auxiliary classification output to inject additional gradient to lower layers. They aren't used for the Test phase.

![https://media.geeksforgeeks.org/wp-content/uploads/20200429201549/Inceptionv1_architecture.png](https://media.geeksforgeeks.org/wp-content/uploads/20200429201549/Inceptionv1_architecture.png)

### ResNet (2015)

ResNet introduced a new method that let to make a much deeper network.

When we stack *plain* layers on top of each other, the performance will decrease. But we know the maximum performance of a deeper network has to be the same or better than a shallower network because we could copy the first layers from the shallower network and the rest layers to be identity mapping layers. So the first guess is learning identity mapping is hard for the network.

![https://media.geeksforgeeks.org/wp-content/uploads/20200424200128/abc.jpg](https://media.geeksforgeeks.org/wp-content/uploads/20200424200128/abc.jpg)

 The solution is to use network layers to fit a residual mapping. With this new approach, the network learns something if there is something to learn. Also, this method lets us add some layers to the network after the network is trained partially to enhance the accuracy.

![https://media.geeksforgeeks.org/wp-content/uploads/20200424011510/Residual-Block.PNG](https://media.geeksforgeeks.org/wp-content/uploads/20200424011510/Residual-Block.PNG)

Below you can see the architecture of the ResNet network with 152 layers. Another advantage of ResNet is to preserve the gradient even for the first layers.

![https://tariq-hasan.github.io/assets/images/resnet.png](https://tariq-hasan.github.io/assets/images/resnet.png)