from collections import Counter
from math import log10
import pandas as pd

def entropyOfProbabilities(ps):
    return -sum([p*log10(p) for p in ps])

def convertCountsToProbabilities(counter):
    total = sum(counter.values())
    return [val/total for val in counter.values()]

def getProbabilityDict(counter):
    total = sum(counter.values())
    return {k:v/total for (k,v) in counter.items()}

def calculateEntropy(series):
    assert type(series) is pd.Series, "Not a series"
    classCounter = Counter(series)
    probabilities = convertCountsToProbabilities(classCounter)
    entropy = entropyOfProbabilities(probabilities)
    return entropy


def calculateInformationGain(featureSeries, targetSeries):
    overallEntropy = calculateEntropy(targetSeries)
    probDict = getProbabilityDict(Counter(featureSeries))
    filteredDataFrames = [(probability, featureSeries == k) for k, probability in probDict.items()]
    categorizedEntropy = [weight*calculateEntropy(targetSeries[filterPred]) for weight, filterPred in filteredDataFrames]
    return overallEntropy - sum(categorizedEntropy)


def isPure(classes):
    return len(classes) == 1


def getClasses(y):
    return set(y)


def buildNode(X, y, bestColumn):
    node = dict()
    node["category"] = bestColumn
    nodeClasses = getClasses(y)
    node["leaf"] = isLeaf  = isPure(nodeClasses)
    if isLeaf:
        (nodeClass,) = nodeClasses
        node["class"] = nodeClass
    else:
        if len(X.columns) > 1:
            node["children"] = children = dict()
            categories = getClasses(X[bestColumn])
            Xcol = X[bestColumn]
            for category in categories:
                filterCrit =  Xcol == category
                Xfiltered = X[filterCrit]
                yfiltered = y[filterCrit]
                children[category] = buildDecisionTree(Xfiltered.drop([bestColumn], axis=1), yfiltered)
    return node
    
    
def buildDecisionTree(X, y):
    candidateColumns = list(X.columns)
    bestColumn = candidateColumns[0]
    bestInformationGain = calculateInformationGain(X[bestColumn], y)
    if len(candidateColumns) > 1:
        for col in candidateColumns[1:]:
            informationGain = calculateInformationGain(X[col], y)
            if informationGain > bestInformationGain:
                bestColumn = col
                bestInformationGain = informationGain
    print("Splitting on", bestColumn)
    root = buildNode(X, y, bestColumn)
    
    return root


def predict(dTree, X_inp):
    if dTree["leaf"]:
        return dTree["class"]
    else:
        cat = dTree["category"]
        Xcat = X_inp[cat]
        return predict(dTree["children"][Xcat], X_inp)
    
def printTree(dTree, depth=0):
    if dTree["leaf"]:
        print(dTree["class"])
    if not dTree["leaf"]:
        print(dTree["category"],end=' ')
        for (cat, node) in dTree["children"].items():
            print("    ", end=' ')
            print(cat,end=' ')
            printTree(node, depth + 1)