# mira.py
# -------
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


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """

        def score(f,y,w):
            return w[y] * f

        def norm_squared(v):
            n = 0

            for k,val in v.items():
                n += val**2

            return n

        def scalar_mult(v,s):
            w = util.Counter()

            for k,val in v.items():
                w[k] = val * s

            return w

        weights = {}
        for c in Cgrid:
            weights[c] = self.weights.copy()


        for iteration in range(self.max_iterations):
            print "Starting iteration ", iteration, "..."
            for i in range(len(trainingData)):
                y = trainingLabels[i]
                f = trainingData[i]
                
                # update all classifiers
                for c in Cgrid:
                    weight = weights[c]
                    scores = [score(f,l,weight) for l in self.legalLabels]

                    m = max(scores)
                    idx = scores.index(m)
                    guess = self.legalLabels[idx]

                    tau = min(c, float((weight[guess] - weight[y]) * f + 1) / (2 * norm_squared(f)))
                    
                    if guess != y:
                        ftau = scalar_mult(f,tau)
                        weight[y] += ftau                   
                        weight[guess] -= ftau

        scores = util.Counter()

        for c in Cgrid:
            self.weights = weights[c]
            guesses = self.classify(validationData)

            for i,g in enumerate(guesses):
                if g == validationLabels[i]:
                    scores[c] += 1
        
        idx = max(scores, key=scores.get)
        self.weights = weights[idx]

    def classify(self, data ):

        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


    def findHighOddsFeatures(self, label1, label2):
        """
        Returns a list of the 100 features with the greatest difference in feature values
                         w_label1 - w_label2

        """
        featuresOdds = []

        "*** YOUR CODE HERE ***"

        return featuresOdds
