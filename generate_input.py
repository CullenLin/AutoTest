import os
import json
import numpy as np
import tensorflow as tf

TRAINING_TESTCASE_FILE = 'trainning_testcase.txt'
TRAINING_FEATURE_FILE = 'training_features'
TESTCASE_VOCABULARY_FILE = 'testcase_vocabulary.dat'
TEST_METHOD_DEF_FILE = 'testcase_method.json'

def readTcVocabulary():
    vocabularyFile = open(TESTCASE_VOCABULARY_FILE, "rt")
    feature_words = []
    while True:
        word=vocabularyFile.readline()
        if len(word)<1:
            break;
        feature_words.append(word.strip())
        
    return (feature_words, len(feature_words))

def readTestMethod():
    data = json.load(open(TEST_METHOD_DEF_FILE))
    if 'testMethods' not in data:
        print('testMethods is not defined in json')
        exit(1)
    testMethods = data['testMethods']
    num_of_test_methods = len(testMethods)
    return (testMethods, num_of_test_methods)

def main():
    print('generating training samples...')
    feature_words, featureSize = readTcVocabulary()        
    testMethods, methodSize = readTestMethod()

    # Define feature and labels  
    features =  []
    labels = []

    # open traning samples
    tcFile = open(TRAINING_TESTCASE_FILE, "rt")    
    example = tcFile.readline()

    # Parse training samples
    while example:
        tcMethod = example.split(':')[0].strip()
        tcStep = example.split(':')[1].lower().strip()

        # Get Label
        if tcMethod not in testMethods:
            continue
        labelIndex = testMethods.index(tcMethod)
        label = np.zeros([methodSize], dtype=np.int32)
        label[labelIndex] = 1
        labels.append(label)           

        # Get feature
        feature = []
        for word in feature_words:
            feature.append(1) if word in tcStep.split() else feature.append(0)
        features.append(feature)
        
        example = tcFile.readline().strip()            

    print(np.array(features))
    print(np.array(labels))

    # Save converted training samples
    print('Saving training data to \"' + TRAINING_FEATURE_FILE + '.npy\"')
    np.save(TRAINING_FEATURE_FILE, {'features':np.array(features), 'labels':np.array(labels)})
    #data = np.load('/tmp/training_data.npy')
    #print(data)
    #print(data.item().get('features'))
    #print(data.item().get('labels'))
        

if __name__ == "__main__":
    main()
