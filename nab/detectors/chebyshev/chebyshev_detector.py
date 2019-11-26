from nab.detectors.base import AnomalyDetector

import math

class ChebyshevDetector(AnomalyDetector):
    """ An streaming version of the algorithm found in the paper: 
    "Data Outlier Detection using the Chebyshev Theorem"
    using Welford's online algorithm to calculate mean and standard deviation
    """

    def __init__(self, *args, **kwargs):
        super(ChebyshevDetector, self).__init__(*args, **kwargs)
        self.p1 = 0.1 # Stage 1 probability 
        self.p2 = 0.001 # Stage 2 probability 
        self.k1 = 1/math.sqrt(self.p1)
        self.k2 = 1/math.sqrt(self.p2)
        self.n1 = 0
        self.m1 = 0
        self.m1_2 = 0
        self.std1 = 1
        self.n2 = 0
        self.m2 = 0
        self.m2_2 = 0
        self.std2 = 1

    def handleRecord(self, inputData):
        """Returns a tuple (anomalyScore).
        The input value is considered an outlier if it resides outside the Outlier Detection Values (upper or lower).
        The anomalyScore is calculated based on the normalized distance the input value has from the upper or lower 
        ODVs, if the input value is considered and outlier, otherwise it is 0.0.
        The probabilities p1 and p2 have been tuned a bit to give good performance on NAB.
        """
        anomalyScore = 0.0
        inputValue = inputData["value"]
        # stage 1 statistics
        self.n1 += 1
        delta = inputValue - self.m1
        self.m1 += delta/self.n1
        self.m1_2 += delta * (inputValue - self.m1)
        self.std1 = math.sqrt(self.m1_2/(self.n1-1)) if self.n1-1 > 0 else 0.000001
        odv1_high = self.m1 + self.k1 * self.std1
        odv1_low = self.m1 - self.k1 * self.std1
        if inputValue <= odv1_high and inputValue >= odv1_low:
            # Passed the first test, let's calculate the second stage statistics
            self.n2 += 1
            delta = inputValue - self.m2
            self.m2 += delta/self.n2
            self.m2_2 +=  delta * (inputValue - self.m2)
            self.std2 = math.sqrt(self.m2_2/(self.n2-1)) if self.n2-1 > 0 else 0.000001
        odv2_high = self.m2 + self.k2 * self.std2
        odv2_low = self.m2 - self.k2 * self.std2
        if inputValue > odv2_high:
            ratio = (inputValue - odv2_high)/inputValue
            anomalyScore = ratio
        elif inputValue < odv2_low:
            ratio = abs((odv2_low - inputValue)/odv2_low)
            anomalyScore = ratio
        return (anomalyScore, )
