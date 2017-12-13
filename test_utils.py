import os
import json

TEST_METHOD_DEF_FILE = 'testcase_method.json'

def readTestMethod():
    data = json.load(open(TEST_METHOD_DEF_FILE))
    if 'testMethods' not in data:
        print('testMethods is not defined in json')
        exit(1)
    testMethods = data['testMethods']
    num_of_test_methods = len(testMethods)
    return (testMethods, num_of_test_methods)

def getTestMethod(index):
    testMethods, num_of_test_methods = readTestMethod()
    
    return testMethods[index]