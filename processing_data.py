import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator


def readTextFile(filename):
    _CSV_COLUMN_DEFAULTS = [[1], [0], [''], [''], [''], [''],['']]
    _CSV_COLUMNS = [
    'age', 'workclass', 'education', 'education_num',
    'marital_status', 'occupation', 'income_bracket'
]
    
    dataset = tf.data.TextLineDataset(filename)
    iterator = dataset.make_one_shot_iterator()
    textline = iterator.get_next()

    with tf.Session() as sess:
        print(textline.eval())

    # convert text to list of tensors for each column
    def parseCSVLine(value):
        columns = tf.decode_csv(value, _CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        return features
    
    dataset2 = dataset.map(parseCSVLine)
    iterator2 = dataset2.make_one_shot_iterator()
    textline2 = iterator2.get_next()  

    with tf.Session() as sess:
        print(textline2)
            
        

def readBinaryFile(filename):
    pass

def main(unused_argv):
    readTextFile('training_features.csv')

if __name__ == "__main__":
  tf.app.run()
