# Lecture 7

# Stochastic Gradient Descent (SGD) problems

Below we mention the most important problem of the SGD method for optimization. 

## High condition number for the loss function

if the ratio of largest to smallest in the hessian matrix is high, the learning rate could become much lower. Because of oscillation along the axis with a high gradient.

![https://miro.medium.com/max/4308/1*ImvekfhM6sXo2IyAdslKLg.png](https://miro.medium.com/max/4308/1*ImvekfhM6sXo2IyAdslKLg.png)

## Local minima or saddle points

The gradient in local minima or saddle point is zero, so SGD gets stuck and can't find the optimal point.

![https://www.mathsisfun.com/calculus/images/function-min-max.svg](https://www.mathsisfun.com/calculus/images/function-min-max.svg)

## Noisy gradient

In the SGD, we calculate the gradient in mini-batches, which approximates the gradient on training data, not the exact one. It could raise the number of iterations for convergence.

[https://pythonmachinelearning.pro/wp-content/uploads/2017/09/GD-v-SGD-825x321.png.webp](https://pythonmachinelearning.pro/wp-content/uploads/2017/09/GD-v-SGD-825x321.png.webp)

# SGD + Momentum

In SGD + Momentum, we suppose the gradient as acceleration, not velocity.

$$v_0 = 0 \\v_{t+1} = \rho v_t + \nabla f(x_t)\\ x_{t+1} = x_t - \alpha v_{t+1}$$

This new approach could help to solve all the problems we talk about it above.

# Nesterov Momentum

$$v_0 = 0 \\v_{t+1} = \rho v_t - \alpha \nabla f(x_t + \rho v_t)\\ x_{t+1} = x_t + v_{t+1}$$

# RMSProp

$$sq_0 = 0 \\ sq_{t+1}= decay\times sq_t + (1-decay) \times (\nabla f(x_t))^2 \\ x_{t+1} = x_t - lr \times \frac{\nabla f(x_t)}{\sqrt{sq_{t+1}+\epsilon}}$$

# Adam

$$fm_0 = sm_0 = 0 \\ fm_{t+1}= \frac{ \beta_1\times fm_t + (1-\beta_1) \times (\nabla f(x_t))}{1-(\beta_1)^t}\\ sm_{t+1}= \frac{\beta_2\times sm_t + (1-\beta_2) \times (\nabla f(x_t))^2 }{1-(\beta_2)^t}\\ x_{t+1} = x_t - lr \times \frac{fm_{t+1}}{\sqrt{sm_{t+1}+\epsilon}}$$

# Annealing learning rate

A high learning rate acts better in the first epochs but may not find the optimal point. On the other hand, a low learning rate takes too much time to converge. A good approach may be to start with a high learning rate and slowly decrease it over time.

![https://www.researchgate.net/profile/Hajar-Feizi/publication/341609757/figure/fig2/AS:894745802977280@1590335431623/Changes-in-the-loss-function-vs-the-epoch-by-the-learning-rate-40_W640.jpg](https://www.researchgate.net/profile/Hajar-Feizi/publication/341609757/figure/fig2/AS:894745802977280@1590335431623/Changes-in-the-loss-function-vs-the-epoch-by-the-learning-rate-40_W640.jpg)

# Model Ensembles

In practice, for improving the accuracy of NN, multiple independent NN is trained and, in the end, average the results. One way could train models with different initialization. Another approach is to store different snapshots from the model in the training phase.

# Drop out

In Drop out method, at each iteration for training the model, some neurons would be deactivated. This process adds some randomness to the model. The model is less likely to overfit the training data. At test time, we average the results by multiplying the probability of deactivation to each layer output.

![https://www.researchgate.net/profile/Amine-Ben-Khalifa/publication/309206911/figure/fig3/AS:418379505651712@1476760855735/Dropout-neural-network-model-a-is-a-standard-neural-network-b-is-the-same-network_W640.jpg](https://www.researchgate.net/profile/Amine-Ben-Khalifa/publication/309206911/figure/fig3/AS:418379505651712@1476760855735/Dropout-neural-network-model-a-is-a-standard-neural-network-b-is-the-same-network_W640.jpg)

# Data Augmentation

Another approach to prevent overfitting is to use data augmentation. In data-augmentation, images are transformed with some method to make new samples from existing samples. Commonly used data augmentations are horizontal flip, random crops, and more.

![https://snowdog-archive.now.sh/images/2018/03/dogo_left_right.png](https://snowdog-archive.now.sh/images/2018/03/dogo_left_right.png)

# Transfer learning

The main idea of transfer learning is to use different models trained for other problems and just tune the last layers for your dataset. The philosophy behind it is that the last layers are problems specific, and the first layers are common things that we encounter in almost any problem. This technic can improve accuracy or decrease training time.

![https://pennylane.ai/qml/_images/transfer_learning_general.png](https://pennylane.ai/qml/_images/transfer_learning_general.png)