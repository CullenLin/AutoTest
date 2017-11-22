# AutoTest with Tensorflow
This prototype is trying to analyze test steps in test case with NLP and logistic regression algorithm. Ideally, the goal is that we only have to 
write description detail about a test case for each step, and this test frameworks will run it for you.

# How the algorithm works
This algorithm contains two phases:
1. Collect features and labels. 
2. Train the model using training feature and labels

## Collect features and labels
This step analyzes the plain text description of the test step in the training data, and convert it to binary data structure. The output is
a dict which contains features and labels (of type ndarray in numpy):

1. Analyze test steps in the training example which is in plain text file 'trainning_testcase.txt'
2. Analyze test methods in 'testcase_method.json', and parse it as the classifiers that the algorithm try to predict
3. Extract training features from test steps, and extract training labels which is the classifier in step2
4. Save features and labels which is a numpy array to a '.npy' file for next processing

## Train the model
The algorithm we used here is 'one-versus-all classification'. actually, Tensorflow already provides such algorithm implementation with 
'softmax', the code is in 'AITestcase.py' which does the following computation:

1. Define our hypothesis h(x) = WX + b
2. Define cost (loss) function with tf.nn.softmax_cross_entropy_with_logits
3. Minimize the cost using gradientdescent algorithm: tf.train.GradientDescentOptimizer

Finally, this algorithm will ouput our parameters W and b, and we will use them to predict our new test steps in test case.
