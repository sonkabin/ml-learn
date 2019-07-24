import numpy as np

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['mayba', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to' ,'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1] # 1表示侮辱性的文字，0表示正常言论
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWord2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:   print('the word', word, 'is not in my vocabulary')
    return returnVec

'''
import mycode.Ch04.bayes as bayes
postingList, listClass = bayes.loadDataSet()
myVocabList = bayes.createVocabList(postingList)
bayes.setOfWord2Vec(myVocabList, postingList[0])

trainList = bayes.createTrainList(myVocabList, postingList)
p0V, p1V, pAb = bayes.trainNB0(trainList, listClass)
'''
def createTrainList(vocabList, postingList):
    trainList = []
    for postInDoc in postingList:
        trainList.append(setOfWord2Vec(vocabList, postInDoc))
    return trainList

def trainNB0(trainList, trainCategory):
    numTrainDocs = len(trainList)
    numWords = len(trainList[0]) # 所有单词的长度
    pAbusive = np.sum(trainCategory)/float(numTrainDocs) # 由于侮辱性文字是1，非侮辱性文字是0.故计算的是侮辱性文字的概率
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords) 
    p0Denom = 2.0;  p1Denom = 2.0 # 拉普拉斯修正：每个特征出现时取1，不出现取0，故为2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainList[i]
            p1Denom += np.sum(trainList[i])
        else:
            p0Num += trainList[i]
            p0Denom += np.sum(trainList[i])
    p1Vect = np.log(p1Num / p1Denom) # 侮辱性文字的概率
    p0Vect = np.log(p0Num / p0Denom) # 取自然对数，防止下溢出
    return p0Vect, p1Vect, pAbusive

# vec2Classify：待分类向量对应于词向量中的表示，pClass1：分为侮辱性文字的概率
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = np.sum(vec2Classify * p0Vec) + np.log(pClass1)
    print(p1, p0)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    postingList, listClass = loadDataSet()
    myVocabList = createVocabList(postingList)
    trainList = createTrainList(myVocabList, postingList)
    p0V, p1V, pAb = trainNB0(trainList, listClass)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWord2Vec(myVocabList, testEntry))
    print(testEntry,'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWord2Vec(myVocabList, testEntry))
    print(testEntry,'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))

# 词袋模型
def bagOfWord2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = [];   classList = []; fullText = []
    for i in range(1, 26):
        wordList = textParse(open('mycode/Ch04/email/spam/%d.txt' % i, encoding='Windows-1252').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('mycode/Ch04/email/ham/%d.txt' % i, encoding='Windows-1252').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50));    testSet = []
    for i in range(10):
        randomIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randomIndex])
        del(trainingSet[randomIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), trainClasses)
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWord2Vec(vocabList, docList[docIndex])
        if classifyNB(wordVector, p0V, p1V, pSpam) != classList[docIndex]:
            print('prediction is error')
            errorCount += 1
    print('the error rate is:', float(errorCount) / len(testSet))
        