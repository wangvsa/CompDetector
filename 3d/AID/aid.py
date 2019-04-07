import numpy as np


class AdaptiveDetector:
    def __init__(self):
        self.it = 0
        self.order = 0
        self.fp = 0
        self.filterBound = 0.00078125
        #self.filterBound = 0.0001
        self.selectOrderFreq = 20

    def simple(self, d1):
        return d1
    def lcf(self, d1, d2):
        return 2 * d1 - d2
    def qcf(self, d1, d2, d3):
        return 3 * d1 - 3 * d2 + d3
    def ccf(self, d1, d2, d3, d4):
        return 4 * d1 - 6 * d2 + 4 * d3 - d4


    '''
    Perform detection for the given data point
    args:
        v: observed value
        pred: prediction value
        lastPredError: prediction error of last frame's data point
        valueRange: value range of this frame
    return:
        True if its an error, Flase otherwise
    '''
    def detect_one_point(self, v, pred, lastPredErr, valueRange):

        radius = valueRange * self.filterBound + lastPredErr
        upperBound = pred + radius + radius * self.fp
        lowerBound = pred - radius - radius * self.fp

        result = False
        if v > upperBound or v < lowerBound:
            #print(v, pred, upperBound, lowerBound)
            result = True

        return result

    '''
    Perform detection on a data frame
    args:
        d, d1, d2, d3, d4, d5: 5 consecutive data frames, d is the current one
    '''
    def detect(self, d, d1, d2, d3, d4, d5):
        if self.it % self.selectOrderFreq == 0:
            self.order = self.select_best_order(d, d1, d2, d3, d4, d5)
            print("order: ", self.order)

        if self.order == 0:      # simple
            pred = self.simple(d1)
            lastPredErr = np.abs(self.simple(d2) - d1)
        elif self.order == 1:    # LCF
            pred = self.lcf(d1, d2)
            lastPredErr = np.abs(self.lcf(d2, d3) - d1)
        elif self.order == 2:    # QCF
            pred = self.qcf(d1, d2, d3)
            lastPredErr = np.abs(self.qcf(d2, d3, d4) - d1)
        elif self.order == 3:   # CCF
            pred = self.ccf(d1, d2, d3, d4)
            lastPredErr = np.abs(self.ccf(d2, d3, d4, d5) - d1)

        valueRange = np.max(d) - np.min(d)
        for x in range(d.shape[0]):
            for y in range(d.shape[1]):
                for z in range(d.shape[2]):
                    hasError = self.detect_one_point(d[x,y,z], pred[x,y,z], lastPredErr[x,y,z], valueRange)
                    if hasError:
                        return True
        return False


    '''
    Choose the best fiting curve
    0: lcf, 1: qcf, 2: ccf
    '''
    def select_best_order(self, d, d1, d2, d3, d4, d5):
        simple_error = np.abs(self.simple(d2) - d1)
        lcf_error = np.sum(np.abs(self.lcf(d2, d3) - d1))
        qcf_error = np.sum(np.abs(self.qcf(d2, d3, d4) - d1))
        ccf_error = np.sum(np.abs(self.ccf(d2, d3, d4, d5) - d1))
        order = np.argmin([lcf_error, qcf_error, ccf_error])
        return order

