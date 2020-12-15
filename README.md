# Conformal Prediction
One of the disadvantages of machine learning as a discipline is the lack of reasonable confidence measure for any given prediction.
Conformal prediction uses past experience to determine precise levels of confidence in new predictions. Given an error probability ε,
together with a method that makes a prediction ŷ of a label y, it produces a set of labels, typically containing ŷ, that also contains y with probability 1 − ε.
Conformal prediction can be applied to any method for producing ŷ: a nearest-neighbor method, a support-vector machine, ridge regression, etc.

