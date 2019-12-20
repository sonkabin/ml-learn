import numpy as np
import cv2 as cv

def ocr_hand_written_digit(base_path):
    img = cv.imread(base_path + 'digits.png')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Now we split the image to 5000 cells, each 20x20 size
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

    # Make it into a Numpy array. It size will be (50,100,20,20)
    x = np.array(cells)
    # print(x.shape)
    train = x[:,:50].reshape(-1, 400).astype(np.float32) # Size = (2500,400)
    test = x[:, 50:100].reshape(-1, 400).astype(np.float32)

    k = np.arange(10)
    train_labels = np.repeat(k, 250)[:, np.newaxis]
    test_labels = train_labels.copy()

    knn = cv.ml.KNearest_create()
    knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
    ret, result, neighbours, dist = knn.findNearest(test, 5)

    # Now we check the accuracy of classification
    # For that, compare the result with test_labels and check which are wrong
    matches = result==test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    print( accuracy )

def ocr_english_alphabets(base_path):
    # Load the data, converters convert the letter to a number
    data= np.loadtxt(base_path + 'letter-recognition.data', dtype= 'float32', delimiter = ',',
                        converters= {0: lambda ch: ord(ch)-ord('A')})
    # split the data to two, 10000 each for train and test
    train, test = np.vsplit(data, 2)
    
    # split trainData and testData to features and responses
    responses, train = np.hsplit(train, [1])
    labels, test = np.hsplit(test, [1])

    knn = cv.ml.KNearest_create()
    knn.train(train, cv.ml.ROW_SAMPLE, responses)
    ret, result, neighbours, dist = knn.findNearest(test, 5)

    correct = np.count_nonzero(result == labels)
    accuracy = correct*100.0/10000
    print(accuracy)

if __name__ == '__main__':
    base_path = 'opencv/data/'
    # ocr_hand_written_digit(base_path)
    ocr_english_alphabets(base_path)