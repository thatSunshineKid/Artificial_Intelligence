# naiveBayes.py
# -------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        # probability of a label y
        p_y = util.Counter()
        for y in trainingLabels:
            p_y[y] += 1 
        
        p_y.normalize()

        self.p_y = p_y

        c_table = {}
        types = {}

        for l in self.legalLabels:
            c_table[l] = {}
            for f in self.features:
                c_table[l][f] = util.Counter()

        
        # get counts of when the ith feature is 0 and 1 for a particular label
        for i,y in enumerate(trainingLabels):

            for f,v in trainingData[i].items():
                c_table[y][f][v] += 1
                types[v] = True
        

        # score each smoothing variable
        scores = []

        for k in kgrid:

            # probability distributions of p(F_i | Y) for all i
            self.pfs = {}

            for f in self.features:

                # initialize the distribution p(F_i | Y)
                d = {}

                # estimate the distribution (with smoothing)
                for y in self.legalLabels:
                    d[y] = util.Counter()

                    for v in types:
                        d[y][v] = float(c_table[y][f][v] + k) / (c_table[y][f].totalCount() + len(types)*k)

                self.pfs[f] = d.copy()

            self.setSmoothing(k)
            guesses = self.classify(validationData)

            score = 0

            for i, g in enumerate(guesses):
                if g == validationLabels[i]:
                    score += 1
            scores.append(score)

        # find argmax for k
        # recompute the joint for this smoothing value

        k = kgrid[scores.index(max(scores))]

        self.pfs = {}

        for f in self.features:
            d = {}

            # estimate the distribution (with smoothing)
            for y in self.legalLabels:
                d[y] = util.Counter()

                for v in c_table[y][f]:
                    d[y][v] = float(c_table[y][f][v] + k) / (c_table[y][f].totalCount() + len(types)*k)

            self.pfs[f] = d.copy()


    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        logJoint = util.Counter()

        for l in self.legalLabels:
            for f in self.features:
                p = self.pfs[f][l][datum[f]]

                logJoint[l] += math.log(p)

            logJoint[l] += math.log(self.p_y[l])

        return logJoint

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2)

        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        featuresOdds = []

        candidates = []

        for f in self.features:
            candidates.append(self.pfs[f][label1][1] / self.pfs[f][label2][1])
        
        print candidates

        for i in range(100):
            m = max(candidates)
            c = candidates.index(m)

            featuresOdds.append(self.features[c])

            del candidates[c]


        return featuresOdds
