from node import Node
import math

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  t = Node()

  # find the classes in the examples
  key = 'Class'
  classes = list(set(sub[key] for sub in examples))

  # counts how many of each class there is
  for class_ in classes:
    count = 0
    for example in examples:
      if example['Class'] == class_:
        count += 1
    t.classes[class_] = count
  
  # assigns t with most common value
  t.label = max(t.classes, key = t.classes.get)

  #check if the node just has one class
  if t.classes[t.label] == len(examples):
    return t
    
  #check if the node has no attributes
  for key in examples[0]:
    if key != 'Class':
      t.attributes.append(key)

  if len(t.attributes) == 0:
    return t
  
  # finds entropy of parent node
  total_entropy = 0
  for class_ in classes:
    frac = t.classes[class_]/len(examples)
    class_entropy = - (frac) * math.log2(frac)
    total_entropy += class_entropy

  # finds information gain for each attribute
  for attribute in t.attributes:
    information_gain = total_entropy
    values= list(set(sub[attribute] for sub in examples))
    for value in values:
      value_data = [row for row in examples if row[attribute] == value]
      value_count = len(value_data)
      value_entropy = 0
      for class_ in classes:
        class_count = len([row for row in value_data if row["Class"] == class_])
        class_entropy = 0
        if class_count != 0:
          frac = class_count/value_count
          class_entropy = -frac * math.log2(frac)
        value_entropy += class_entropy

      information_gain -= (value_count/len(examples)) * value_entropy
    t.attribute_gain[attribute] = information_gain
    print('information gain', information_gain)
  print(t.attribute_gain)

  # finds maximum information gain
  t.decision_attribute = max(zip(t.attribute_gain.values(), t.attribute_gain.keys()))[1]
  
 

def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''

def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''


def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''
data =  [dict(a=1, b=0, Class=2), dict(a=1, b=1, Class=1),
          dict(a=2, b=0, Class=2), dict(a=2, b=1, Class=3),
          dict(a=3, b=0, Class=1), dict(a=3, b=1, Class=3)]
tree = ID3(data, 0)
