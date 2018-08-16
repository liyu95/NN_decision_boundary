# On the decision boundary of deep neural networks

## Preprint holder

## Basic idea
While deep learning models and techniques have achieved great empirical success, our understanding of the source of success in many aspects remains very limited. In an attempt to bridge the gap, we investigate the decision boundary of a production deep learning architecture with weak assumptions on both the training data and the model. We demonstrate, both theoretically and empirically, that the last weight layer of a neural network converges to a linear SVM trained on the output of the last hidden layer, for both the binary case and the multi-class case with the commonly used cross-entropy loss. Furthermore, we show empirically that training a neural network as a whole, instead of only fine-tuning the last weight layer, may result in better bias constant for the last weight layer, which is important for generalization. In addition to facilitating the understanding of deep learning, our result can be helpful for solving a broad range of practical problems of deep learning, such as catastrophic forgetting and adversarial attacking.


## Folders

### src
This folder contains the source code for the experiments. The three notebook files show the training process and the result on MNIST, CIFAR-10 and multiclass classification. *main.py* is the source code for the experiments for the simulated data. *utils.py* and *tf_utils.py* are some help functions.

### model
This folder contains some trained model. The name of those models are self-explained.

### result
This folder contains some of the results, including some very old exploratory results. 

## Interesting results
### MNIST binary classification using ResNet, trained with SGD, Adam can also give similar result
![alt text](https://github.com/lykaust15/NN_decision_boundary/blob/master/result/exploration/SVM_decision_boundary_transform.png)![alt text](https://github.com/lykaust15/NN_decision_boundary/blob/master/result/exploration/NN_decision_boundary_transform.png)![alt text](https://github.com/lykaust15/NN_decision_boundary/blob/master/result/exploration/NN_transform_reinitailize.png)

### CIFAR-10 binary classification using DenseNet, trained with SGD, Adam can also give similar result



### MNIST 3 class classification using DenseNet, trained with Momentum
![alt text](https://github.com/lykaust15/NN_decision_boundary/blob/master/result/mnist_3_class_densenet/svm_transformed.png)![alt text](https://github.com/lykaust15/NN_decision_boundary/blob/master/result/mnist_3_class_densenet/nn_transformed.png)