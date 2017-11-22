import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator


def _parseData(filename):
    
    print('parsedata...')
    data = tf.read_file(filename)
 
    return filename

def main(unused_argv):
    tr_file = tf.constant(['training_features.csv'])
    tr_data = tf.data.Dataset.from_tensor_slices(tr_file)
    #tr_data.map(_parseData)
    iterator = tr_data.make_initializable_iterator()
    next_element = iterator.get_next()  

    mymat = tf.get_variable('mymat', dtype=tf.int32, initializer=tf.zeros([5, 3], dtype=tf.int32))
    assign_ops = mymat[0].assign(tf.one_hot(0, 3, dtype=tf.int32))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(assign_ops.eval())
        
        #sess.run(training_init_op)
        sess.run(iterator.initializer)
        print(sess.run(next_element))

if __name__ == "__main__":
  tf.app.run()
