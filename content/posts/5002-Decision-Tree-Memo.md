+++
date = '2025-05-16T21:17:29+08:00'
draft = false 
title = '5002 Decision Tree Memo'
+++
# Type of Decision Tree
- ID3
	Information gain
- C4.5
	Information gain normalized by entropy(split info).
- CART

# Entropy
To measure how informative a distribution is.
The formula is as follows:
$\text{Entropy} = - \sum p log p$
Here, the logarithm is based on a base of two.
The greater the entropy is, the less informative the distribution is.
For example:
```
P(tail) = 0.5  P(tail) = 0.5  entropy = 1
P(tail) = 1    P(tail) = 0    entropy = 
```

*We assume 0log 0 = 0 in the calculation*


# Information gain
Suppose we get a table having columns: race, income, label

We calculate the base information(entropy):
Info(label) = entropy(p_yes, p_no)

We calculate the information associated with an attribute.
Info(race) = p_black * entropy(p_yes, p_no | black) + p_white * entropy(p_yes, p_no | white)
*Here, p_black means the proportion of the black attribute*

And we can calculate the information gain by:
Gain(race) = Info(label) - Info(race)

**We should choose the one with the greatest gain.**


# C4.5
We use a new measurement:
Gain(race) = Info(label) - Info(race) / entropy(race)

### Comparison with ID3
ID3 tends to choose the attributes with more values.
C4.5 tries to penalize attributes with more values.

## CART
It uses gini index to calculate the Info.
gini(p_i, ...) = 1 - \sum p_i^2
Info(label) = gini(p_yes, p_no)
Info(white) = gini(p_yes, p_no | white)
Info(black) = gini(p_yes, p_no | black)
Info(race) = p_black * Info(black) + p_white * Info(white)
Gain(race) = Info(label) - Info(race)


## Intuition of Gini
Suppose we get two kinds of balls: x and y.
The probability of picking x is p_x, and correspondingly p_y.
The probability of taking two balls of different kinds is:
1 - P_x^2 - P_y^2.
