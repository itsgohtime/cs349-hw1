from node import Node
import math

def missing_attributes(examples):
  for ex in examples:
    missing_data = False
    class_ = ex['Class']
    for key in ex:
      if ex[key] == '?':
        missing_data = True
        missing_key = key

    if missing_data:
      class_data = [row for row in examples if row['Class'] == class_]
      values= list(set(sub[missing_key] for sub in class_data))
      keys = ex.keys()

      for datapoint in class_data:
        matching = False
        if datapoint[missing_key] != '?':
          matching = True
          for key in keys:
            if key != missing_key:
              if datapoint[key] != ex[key]:
                matching = False
        
        if matching:
          ex[missing_key] = datapoint[missing_key]

      if ex[missing_key] == '?':
        key_dict = {}
        for val in values: 
          if val != '?':
            key_dict[val] = len([row for row in class_data if row[missing_key] == val])

        ex[missing_key] = max(key_dict)

  return examples

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  t = Node()

  examples = missing_attributes(examples)

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

  #check if most common class is the entire set
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

  # finds maximum information gain
  t.decision_attribute = max(zip(t.attribute_gain.values(), t.attribute_gain.keys()))[1]
  information_gain =  max(zip(t.attribute_gain.values(), t.attribute_gain.keys()))[0]

  a_val = list(set(row[t.decision_attribute] for row in examples))

  for a in a_val:
    # get subset of examples 
    D_a = []
    for dict in examples:
      if dict[t.decision_attribute] == a:
        D_a.append(dict.copy())
    
    for row in D_a:
      row.pop(t.decision_attribute)

    if len(D_a) == 0:
      t_prime = Node()
      t_prime.label = max(t.classes, key = t.classes.get)
      t.children[a] = t_prime
    else:
      t.children[a] = ID3(D_a, default)
  
  return t


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
  numerator = 0
  denominator = 0
  for test in examples:
    result = evaluate(node, test)
    if result == test['Class']:
      numerator += 1
      denominator += 1
    else:
      denominator += 1

  return numerator/denominator



def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''

  if not node.children:
    return node.label
  attribute = node.decision_attribute
  attribute_val = example[attribute]
  if node.children:
    child_node = node.children[attribute_val]
    return evaluate(child_node, example)
