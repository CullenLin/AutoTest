#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Model training for Iris data set using Validation Monitor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Define training samples
TESTCASE_TRAINING = os.path.join(os.path.dirname(__file__), "training_features.npy")

def readData():  

  # Load training examples
  data = np.load(TESTCASE_TRAINING)
  features = data.item().get('features')
  labels = data.item().get('labels')

  num_of_labels = labels.shape[1]
  num_of_training_sample = labels.shape[0]
  num_of_features = features.shape[1]
  print('number of test methods: ' + str(num_of_labels))
  print('number of training samples: ' + str(num_of_training_sample))
  print('number of features: ' + str(num_of_features))


  # Build training dataset
  labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
  features_placeholder = tf.placeholder(features.dtype, features.shape)
  
  dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
  batched_dataset = dataset.batch(2).repeat()    # repeat to fetch element in dataset
  iterator = batched_dataset.make_initializable_iterator()

  X = tf.placeholder(tf.float32, [None, num_of_features])
  W = tf.get_variable('weigth', dtype=tf.float32, initializer=tf.random_uniform([num_of_features, num_of_labels], maxval=2))
  b = tf.get_variable('bais', dtype=tf.float32, initializer=tf.random_uniform([num_of_labels], maxval=2)) #tf.Variable([5], dtype=tf.float32)
  y = tf.matmul(X, W) + b
  y_ = tf.placeholder(tf.float32, [None, num_of_labels])

  # Define loss function
  cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  
  with tf.Session() as sess:   
    tf.global_variables_initializer().run()
    # Retrieve a single traning sample:
    sess.run(iterator.initializer, feed_dict={features_placeholder: features, labels_placeholder: labels})
      
    # Train
    for i in range(1000):
      if i%20==0:
        print('training ' + str(i) + ' times...')
      next_element = iterator.get_next()
      batch_xs, batch_ys = sess.run(next_element)   
      sess.run(train_step, feed_dict={X: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={X: [[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]],
                                      y_: [[0, 1, 0, 0, 0]]}))

    print('Trained parameters: ')
    print(W.eval())
    print('Trained bais: ')
    print(b.eval())

    # Predict new samples
    print(sess.run(y, feed_dict={X: [[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]]}))

def main(unused_argv):
  readData()
  
if __name__ == "__main__":
  tf.app.run()
