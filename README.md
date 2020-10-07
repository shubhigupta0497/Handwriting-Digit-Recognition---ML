HANDWRITING RECOGNITION USING ARTIFICIAL NEURAL NETWORKS

Abstract— This project focuses on the implementation and development of handwritten digit recognition algorithm (backpropagation) using Artificial Neural Networks. It is a classification problem solved using aforementioned algorithm. A total of 400 input layers are used, corresponding to each pixel of the image and 10 output layers, corresponding to each digit from 0-9. The number of nodes in each hidden layer have been set to 25. The implementation of the same is done on Matlab, using Octave-GUI 4.2.1. ANN model was trained with 5000 images of various digits. The algorithm predicted English numerical digits with a maximum accuracy of 97.8%. 

Index Terms—Hand Written; Artificial Neural Network; Back Propagation; English Numerical Digits; Sigmoid Function


I.	INTRODUCTION 

The Main objective of this project is to implement an algorithm using Artificial Neural Networks that can correctly identify a handwritten digit ranging from 0 to 9. The Algorithm is given an image of a handwritten digit in the form of pixels and its job is to correctly predict the digit on a particular image. It uses a dataset of training examples where in each image is represented as vector of 400 pixels. This dataset can be trained using Artificial Neural Networks and Back-Propagation algorithm and the results obtained can be used to predict the handwritten digits on an image. The program code has been implemented and written in ‘Octave’. An Artificial Neural Network (ANN) is a model based on the structure and the learning ability of biological nervous systems, such as our brain. ANN has thousands of artificial neurons called nodes/units which are interconnected to each other. These nodes are arranged in a series of layers. The different layers in a typical ANN are 

A. Input Layer: It contains the nodes which receive input from user.

B. Output Layer: It contains the nodes that give output corresponding to a particular input. 

C. Hidden Layer: These units lie in between the input and output layers. The main function of hidden layer is to transform the input given by the user. 


ANN’s receive input from the external world in the form of an image/data in vector form. Each input is multiplied by its corresponding weights. These weights are derived using a set of learning rules called Back-propagation. During Back-propagation the network works backwards going from the output units to the input units until the error between the actual and the desired outcomes is minimized. The weighted inputs are all added and the sum is calculated. The sum can take any value ranging from 0 to infinity and hence in order to limit the value of the sum it’s passed through an activation function.


II.	LITERATURE REVIEW 

There are a few pre-existing approaches that can be used for the purpose of character recognition, a few of which have been discussed below a) Hidden Markov Model b) Naïve Bayes classifiers c) Support Vector Machine 

A. Hidden Markov Model Hidden Markov process is a partially observable process (hence the name ‘hidden’). It comprises of various set of states (which can be used for handwritten digits 0-9). Let qt be the state of the system at any random time ‘t’, and transition probability from current state Si to another state(digit) Sj would be aij= P(qt = Sj | qt-1 = Si), 1 ≤ i,j ≤ n. 

B. Naïve Bayes classifiers These are probabilistic classifiers which are based upon the principle of Bayes’ Theorem: statistical principle to calculate probability based on prior conditions of the system. The key assumption of NB classifiers is that the features have to be absolutely independent of each other. Hence, it is basically a conditional probability model. 

C. Support Vector Machine Support Vector Machines are another classification algorithms used extensively for machine learning problems. Consider a data set which contains 2 features and 2 classifications for the sake of simplicity. There are 2 extreme boundaries that can be used for separation of the circle and square labels, but we wish to find the one in the middle, the optimal 2 hyper-plane. For that, we try to maximize the length between the upper and lower boundaries. Let the equation of an hyper-plane be Y=wT+b. Y= {1,0,- 1}, depending on whether it is in the upper segment, the decision boundary or the lower segment respectively. Let there be 2 points ‘A’ and ‘B’ between the 2 extreme boundaries. The distance between them would be 2/|w|, where |w| is the length of vector w and hence, the normalization factor. We try to minimize the value of |w|, so as to maximize the margin, or in other words, try to minimize (1/2)*|w|^2. There are several Matlab quadratic programming algorithms used to solve this and hence, we square the |w| term. Most of the values of data set do not contribute to determining wT, the ones far from decision boundary.

III.	PROPOSED APPROACH 

The proposed Neural Network Architecture consists of 3 layers i.e. input layer, hidden layer and the output layer. The input and the hidden layer is connected by weights theta1 and the hidden and the output layer is connected by weights theta2. The weighted sum from the hidden as well as the output layer can take any value ranging from 0 to infinity and hence in order to limit the value of the sum it’s passed through an activation function. In this scenario we use sigmoid function as the activation function where the value of sigmoid function always lies between 0 and 1. 


A.	Input Layer 

The input from the outside world/user is given to the input layer. Input is given in the form of a matrix X where the number of training examples is same as the number of rows in the matrix and the 400 pixels extracted from the image is arranged as one single row in the input matrix X. Hence the dimensions of matrix is given as X (Number of Training Examples x 400).

B.	Hidden Layer 

There is no valid formula to calculate the number of units/nodes in the hidden layer. To minimize computational costs here the number of hidden nodes have been taken as 25. 

C.	Output Layer 

The output layer in this algorithm consists of 10 units with each unit representing various digits from 0-9. 
Randomly initialize the weights 
The weights connecting the input and the hidden layer as well as the hidden and the output layer are randomly initialized. The range of the weights for Theta_layer_one -0.17 to 0.17 and Theta_layer_two -0.17 to 0.17 


Forward Propagation 

Step-1: The inputs given to the input layer is multiplied with the weights connecting the input and the hidden layer and then passed through sigmoid function. i.e. - Output one = SIGMOID(X*Theta_layer_one)

Step-2: The Output_one is then multiplied with the weights connecting the hidden and the output layer and then passed through the sigmoid function. i.e.Output_two = SIGMOID (Output_one*Theta_layer_two). Hence this way we clearly obtain the final output of our network. 


Cost Function

Initial value of the cost function is calculated using the randomly initialized values of weights connecting the input and the hidden layer and weights connecting the hidden and the output layer. Error regularization is taken into account while calculating the value of cost function and adjustments are made for the same.


Back Propagation 

Back-propagation gives us a way to determine the error in the output of a previous layer given the output of a current layer. The process starts at the last layer and calculates the change in the weights for the last layer. Then we can calculate the error in the output of the previous layer. Using the error in each layer partial derivatives can be calculated for weights connecting the input and the hidden layer as well as weights connecting the hidden and the output layer. 


Optimization Function fmincg

Built in optimization function fmincg is used to obtain the minimum value of cost function. It’s fed with the initial value of the cost function, partial derivatives and the random values of weights calculated earlier. The value obtained using fmincg also depends upon the number of iterations. 


Algorithm: 

Step 0: Start 

Step 1: Choose a suitable architecture for the neural network

Step 2: Randomly initialize the weights 

Step 3: Implement forward propagation 

Step 4: Implement the cost function 

Step 5: Implement back propagation to compute partial derivative 3 

Step 6: Use inbuilt optimization function fmincg to minimize cost function. 

Step 7: Stop

