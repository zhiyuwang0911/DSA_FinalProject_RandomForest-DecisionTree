import numpy as np
import pandas as pd

class Node():
    # Constructor for tree's nodes
    def __init__(self, feature_index=None, threshold=None, leftChild=None, rightChild=None, info_gain=None, value=None):
        # Attributes for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.info_gain = info_gain
        self.leftChild = leftChild 
        self.rightChild = rightChild
        
        # Attribute for leaf node
        self.value = value

class DecisionTreeClassifier():
    # Constructor for Decision Tree
    def __init__(self, min_samples_split=2, max_depth=2):
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
    
    # To train the tree
    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.buildTree(dataset)

    # Builds tree recursively
    def buildTree(self, dataset, curr_depth=0):
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.getBestSplit(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                # recur left
                left_subtree = self.buildTree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.buildTree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # compute leaf node
        leaf_value = self.getLeafValue(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    # Printing the entire tree
    def printTree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.printTree(tree.leftChild, indent + indent)
            print("%sright:" % (indent), end="")
            self.printTree(tree.rightChild, indent + indent)

    # Finding the most effective way to split data
    def getBestSplit(self, dataset, num_samples, num_features):
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.splitData(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.getInfoGain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        return best_split
    
    # Splitting the data
    def splitData(self, dataset, feature_index, threshold):
        leftDataset = np.array([row for row in dataset if row[feature_index]<=threshold])
        rightDataset = np.array([row for row in dataset if row[feature_index]>threshold])
        return leftDataset, rightDataset
    
    # Predicting new dataset
    def predict(self, X):
        predictions = [self.singlePrediction(x, self.root) for x in X]
        return predictions
    
    # Predicting single data point
    def singlePrediction(self, x, tree):
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.singlePrediction(x, tree.leftChild)
        else:
            return self.singlePrediction(x, tree.rightChild)
        
    # Computing information gain 
    def getInfoGain(self, parent, leftNode, rightNode, mode="entropy"):
        leftWeight = len(leftNode) / len(parent)
        rightWeight = len(rightNode) / len(parent)
        if mode=="gini":
            gain = self.getGiniIndex(parent) - (leftWeight*self.getGiniIndex(leftNode) + rightWeight*self.getGiniIndex(rightNode))
        else:
            gain = self.getEntropy(parent) - (leftWeight*self.getEntropy(leftNode) + rightWeight*self.getEntropy(rightNode))
        return gain
    
    # Computing entropy
    def getEntropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels: # cls refers to the classes in class_labels
            classProbability = len(y[y == cls]) / len(y)
            entropy += -classProbability * np.log2(classProbability)
        return entropy
    
    # Computing gini index
    def getGiniIndex(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            classProbability = len(y[y == cls]) / len(y)
            gini += classProbability**2
        return 1 - gini

    # Computing leaf node's value    
    def getLeafValue(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)
    