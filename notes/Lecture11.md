# Lecture 11

# Segmentation, Localization, Detection

Until now, the main problem that we tried to solve was image classification, assigning one label to input image from the fixed set of categories. In the rest of this lecture, we try to give an overall view of other computer vision problems.

### Semantic Segmentation

In semantic segmentation, we go one step forward and try to classify each pixel of an image instead of the input image as a whole.

❗Developing training data for semantic segmentation is too expensive.

![https://miro.medium.com/max/1354/1*ma9XpjFwPgkM078YSGY9iA.png](https://miro.medium.com/max/1354/1*ma9XpjFwPgkM078YSGY9iA.png)

![https://miro.medium.com/max/875/1*_pldf5yn6Ty3jy6cFGz5mQ.png](https://miro.medium.com/max/875/1*_pldf5yn6Ty3jy6cFGz5mQ.png)

![https://cdn-images-1.medium.com/max/600/1*kvh9u8W2sHlQoBPfwERggA.gif](https://cdn-images-1.medium.com/max/600/1*kvh9u8W2sHlQoBPfwERggA.gif)

The first idea to tackle this problem may be to use the *sliding-window* technic. Choose a small window, then move the window on the image and try to classify the center pixel of the window. But this method is not practical because it's very computationally expensive. Also, many computations are done that are redundant and could be shared.

The second idea is to design a big fully connected convolutional network that does predictions for each pixel at once. In this method, we preserve input image resolution in the whole network. preserving input image resolution in the whole network is expensive and could be problematic.

The last and commonly used idea is based on the previous one. But instead of preserving image resolution in the whole network, down-sampling and up-sampling are done. Down-sampling is done by usual convolutional layers and building the feature map. The new and unfamiliar part of the network is deconvolution layers that do up-sampling and produce classification scores for each pixel from the feature map. 

We briefly explain some commonly used layers for up-sampling.

### Nearest Neighbor

$$\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
\to 
\begin{bmatrix}
1 & 1 & 2 & 2\\
1 & 1 & 2 & 2\\
3 & 3 & 4 & 4\\
3 & 3 & 4 & 4\\
\end{bmatrix}$$

### Bed of Nails

$$\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
\to 
\begin{bmatrix}
1 & 0 & 2 & 0\\
0 & 0 & 0 & 0\\
3 & 0 & 4 & 0\\
0 & 0 & 0 & 0\\
\end{bmatrix}$$

### Max Unpooling

Max Unpooling is an improvement on the Bed of Nails layer. Most of the time, down-sampling and Up-sampling are done by symmetric networks that convolution and deconvolution parts are reflections of each other. So in max-pooling layers, the network remembers the location of the maximum element in each pool, and when arriving at the corresponding max-unpooling layers, input elements of the layer get laid in the remembered position of the bigger matrix.

$$\begin{bmatrix}
1 & 3 & 2 & *7\\
4 & *6 & 5 & 3\\
*4 & 2 & 3 & *5\\
1 & 2 & 0 & 1\\
\end{bmatrix}
\to
\begin{bmatrix}
6 & 7\\
4 & 5
\end{bmatrix}
\to
...
\to
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
\to 
\begin{bmatrix}
0 & 0 & 0 & *2\\
0 & *1 & 0 & 0\\
*3 & 0 & 0 & *4\\
0 & 0 & 0 & 0\\
\end{bmatrix}$$

### Transpose Convolution

The previous layers for up-sampling don't have any learnable parameters. But transpose convolution tried to be like the inverse of the convolution operation and has a kernel as a learnable parameter.

A general explanation of transpose convolution:

1. Initiate output with zero.
2. Move kernel on output like moving it on input in typical convolution.
3. Multiply each element of input with all elements of the kernel.
4. Sum up corresponding elements of output with the results of the previous part (part 3).

![https://miro.medium.com/max/875/1*faRskFzI7GtvNCLNeCN8cg.png](https://miro.medium.com/max/875/1*faRskFzI7GtvNCLNeCN8cg.png)

### Classification + Localization

In Classification + Localization problems, we have to first classify the image, and next localize the object in the image (for example, putting a rectangle around the object). Its difference with object detection is there is just one instance of the object in the picture.

In these kinds of problems, the base is the same as classification problems. It just needs some additional parts for object localization. We could reuse the network for image classification and augment the last layer for localization data to solve the problem. It's like there are two parallel layers in the last. The final loss is the waited sum of these two losses.

![https://gblobscdn.gitbook.com/assets%2F-LvMRntv-nKvtl7WOpCz%2F-LvMRp9FltcwEeVxPYFs%2F-LvMRqhQOGHwE0Sjd29J%2FLocalizationRegression2.png](https://gblobscdn.gitbook.com/assets%2F-LvMRntv-nKvtl7WOpCz%2F-LvMRp9FltcwEeVxPYFs%2F-LvMRqhQOGHwE0Sjd29J%2FLocalizationRegression2.png)

### Object Detection

In object detection problems, we want to find the class of the objects and their location in the image. Its difference with classification + localization is that there may be any number of objects of any kind in the image (zero or more). This makes the problem challenging because there is a varying number of the output.

The first idea that comes to mind is to use the sliding window and transfer the problem to the classification of these many windows. But the problem of brute force sliding windows for object detection is computationally expensive and isn't practical.

![https://www.researchgate.net/profile/Peter-M-Roth/publication/266215670/figure/fig1/AS:669537766744083@1536641651568/Object-detection-by-sliding-window-approach_W640.jpg](https://www.researchgate.net/profile/Peter-M-Roth/publication/266215670/figure/fig1/AS:669537766744083@1536641651568/Object-detection-by-sliding-window-approach_W640.jpg)

Instead of the brute force approach for finding windows, people use region proposals. Region proposal uses traditional computer vision technics like image processing to find windows that more likely an object could be present. With these techniques, the number of images to enter the network to be classified would be reduced. Precision may be low, but recall is good. The RCNN network could also output localization data to correct object location in the input region. RCNN is better than the previous approach but is slow yet. 

![https://miro.medium.com/max/753/1*yWJB5OkMK4UJxLHoeWWWWg.png](https://miro.medium.com/max/753/1*yWJB5OkMK4UJxLHoeWWWWg.png)

An improvement on RCNN is fast-RCNN that instead of applying region proposals in the raw input image, region proposal is done on the image feature map produced by CNN. Fast-RCNN is better than the previous method, but the bottleneck is the region proposal network computation.

  

![https://www.researchgate.net/profile/Qi-Liao-7/publication/327551089/figure/fig3/AS:669218995441678@1536565650740/Faster-R-CNN-architecture_W640.jpg](https://www.researchgate.net/profile/Qi-Liao-7/publication/327551089/figure/fig3/AS:669218995441678@1536565650740/Faster-R-CNN-architecture_W640.jpg)

In the faster-RCNN, the region proposal becomes network part and learnable. This method removes the bottleneck of fast-RCNN.

![https://miro.medium.com/max/875/1*pSnVmJCyQIRKHDPt3cfnXA.png](https://miro.medium.com/max/875/1*pSnVmJCyQIRKHDPt3cfnXA.png)

There are other approaches to object detection like YOLO or SSD that don't use proposal regions. They are faster but less accurate.

### Instance Segmentation

Instance segmentation is hybrid of semantic segmentation and object detection. We could add some layers in the final stages to do segmentation at the pixel level to solve this problem.

![https://paperswithcode.com/media/methods/Screen_Shot_2020-05-23_at_7.44.34_PM.png](https://paperswithcode.com/media/methods/Screen_Shot_2020-05-23_at_7.44.34_PM.png)