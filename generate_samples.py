import os
import json
import numpy as np
import tensorflow as tf
from test_utils import readTestMethod

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

def generated_training_samples():
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
        
def generate_test_samples():
    feature_words, featureSize = readTcVocabulary() 

    # open test samples
    tcFile = open('testing_tc_samples.txt', "rt")    
    example = tcFile.readline().lower().strip()
    
    features = []
    while example:
        feature = []
        for word in feature_words:
            feature.append(1) if word in example.split() else feature.append(0)
        features.append(feature)
        example = tcFile.readline().strip().lower()

    print(features)
    # Save converted test samples
    print('Saving testing data to \"' + 'testing_features' + '.npy\"')
    np.save('testing_features.npy', {'features':np.array(features)})
            
if __name__ == "__main__":
    #generated_training_samples()
    
    generate_test_samples()
