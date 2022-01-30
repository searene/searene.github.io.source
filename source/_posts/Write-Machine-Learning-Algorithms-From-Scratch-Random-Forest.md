title: 'Write Machine Learning Algorithms From Scratch: Random Forest'
date: 2017-12-23 10:14:33
tags: 'machine learning'
categories: Coding
thumbnail: /images/forest.jpg
---

# Introduction

Random Forest is a supervised classification algorithm, it can classify data according to various given features.

Assuming that we want to determine whether a person is male or female according to his/her weight, height and 100m-race time. Training data is as follows.

| Person | Weight(kg) | Height(meter) | 100m-race time(second) | Gender |
| ------ | ---------- | ------------- | ---------------------- | ------ |
| A      | 50         | 1.62          | 18                     | Female |
| B      | 70         | 1.81          | 16                     | Male   |
| C      | 60         | 1.72          | 15                     | Female |
| D      | 70         | 1.71          | 19                     | Male   |
| E      | 52         | 1.69          | 17                     | Female |

We can load these data and train them with the random forest classification algorithm. The model obtained from training could be used for prediction. E.g., We will be able to predict this person's gender using the trained model.

| Weight(kg) | Height(meter) | 100m-race time(second) |
| ---------- | ------------- | ---------------------- |
| 60         | 1.62          | 16                     |

Notice that we will mainly focus on how to use random forest and how to write the algorithm from scratch. We won't dive into the esoteric mathematical principles behind it. After finishing this post, you will be able to understand various parameters seen in third-party random forest implementations.

All the code mentioned in the post is available for download. So please refer to the code if there's anything unclear in the post.

# Execution

Let's first run the code that we will write, so we could know what it's like.

1. Install Python3

2. Download code

   ```bash
   git clone git@github.com:searene/demos.git && cd demos/RandomForest
   ```

3. Download Dependencies

   ```bash
   pip install numpy pandas
   ```

4. Execution

   ```bash
   python evaluate_random_forest.py
   Average cross validation accuracy for 1 trees: 0.6887700534759359
   Test accuracy for 1 trees: 0.6190476190476191
   Average cross validation accuracy for 3 trees: 0.6898395721925135
   Test accuracy for 3 trees: 0.8571428571428571
   Average cross validation accuracy for 10 trees: 0.6983957219251338
   Test accuracy for 10 trees: 0.7619047619047619
   ```

So you can see that, we get the highest accuracy with 3 trees, which is about 85%.

# How It Works

Random Forest is rather complex, so let's use an example.

| Person | Weight(kg) | Height(meter) | 100m-race time(second) | Gender |
| ------ | ---------- | ------------- | ---------------------- | ------ |
| A      | 50         | 1.62          | 18                     | Female |
| B      | 70         | 1.81          | 16                     | Male   |
| C      | 60         | 1.72          | 15                     | Female |
| D      | 70         | 1.71          | 19                     | Male   |
| E      | 52         | 1.69          | 17                     | Female |

We mentioned before that we could use these data to train our random forest model, in order to predict new items. So how to train? In fact, training is equivalent to building a tree here. Steps are as follows.

1. Based on D's height, anyone whose height is less or equal to 1.71m belong to one group, and anyone whose height is greater than 1.71m belong to another group, then we get two groups(Don't think too much about why to split in this way, this is just an example, we will talk about the reason in detail later).

   ```
        A, B, C, D, E
       /             \
      /               \
   A, D, E           B, C
   ```

2. For group `A, D, E`, based on A's 100m-race time, anyone whose time is less or equal to 18s belong to one group, and anyone whose time is greater than 18s belong to another group. The same goes to group `B, C`. Based on C's height, anyone whose height is less than or equal to 1.72m belong to one group, and anyone whose height is greater than 1.72m belong to another group. After splitting, we get a tree like this.

   ```
          A, B, C, D, E
         /             \
        /               \
      B, C            A, D, E
     /    \          /      \
    /      \        /        \
   C        B     A, E        D
   ```

3. Now only group `A, E` could be further split. So let's base on A's weight, anyone whose weight is less than or equal to 50kg belong to one group, and anyone whose weight is greater than 50kg belong to another group. After that, we mark each leaf node with their genders.

   ```
             A, B, C, D, E
            /             \
           /               \
         B, C            A, D, E
        /    \          /      \
       /      \        /        \
   C(F)       B(M)   A, E       D(M)
                     /   \
                    /     \
                  A(F)   E(M)
   ```

   That's it, a tree in the random forest! Now we can use this tree to predict new data. Assuming we want to predict this person's gender:

| Weight(kg) | Height(meter) | 100m-race time(second) |
| ---------- | ------------- | ---------------------- |
| 60         | 1.62          | 16                     |

   Just like training, this person's height is 1.62m, which is less than or equal to 1.71, so he/she belongs to group `B, C` in the second layer. Again, compare based on his/her height, which is less than or equal to 1.72m, so he/she belongs to leaf node C, which means the prediction result is `Female`. This is the whole process of prediction.

   # The Principle To Split A Tree Into Two Groups

   In the above example, we first split the whole data into two groups according to D's height, then continue to split them according to D's height, A's weight, etc. What's going on here? It seemed that we were casually splitting the data with no principle. OK, I concede that it's true. I just want to show you guys how to build a random forest tree. In fact, the genuine tree-building-process would split the data according to gini index. E.g., assuming we split the data according to A's weight, we will get two groups of data: `A` and `B, C, D, E`. Let's call them group1 and group2 respectively, then we can calculate gini index according to the following equation.

   <div style="background-color: #fff5cc; border-color: #ffe273; padding: 10px; border-left: 10px solid #ffe273">$gini$ = [1 - (the number of males in group1 / the number of people in group1)$^2$ - (the number of females in group1 / the number of people in group1)$^2$] $\times$ (the number of people in group1 / the total number of people in both groups) + [1 - (the number of males in group2 / the number of people in group2)$^2$ - (the number of females in group2 / the number of people in group2)$^2$] $\times$ (the number of people in group2 / the total number of people in both groups)</div>

   So the gini index should be calculated as follows if we split the data based on A's weight.
$$
   gini = 0 + (1 - 0.25 - 0.25) \times 0.8 = 0.4
$$

   We can also split the data based on A's height, to get another gini index.
$$
   gini = 0 + (1 - 0.25 - 0.25) \times 0.8 = 0.4
$$
We can also split based on A's 100m-race time, B's weight, B's height, ..., E's 100m-race time, 3 x 5 = 15 ways in total. We calculate the gini index for each of the 15 ways, and choose the one with the smallest gini index. So we should split based on D's weight if we got the smallest gini index based on D's weight. Why choose the smallest one? Because the smaller gini index is, the purer each group will be. We are not going to dive into the reason in detail here because it's more about the math rather than the implementation.

The code to calculate gini index is as follows.

   ```python
   def get_gini_index(left, right, categories):
       gini_index = 0
       for group in left, right:
           if len(group) == 0:
               continue
           score = 0
           for category in categories:
               p = [row[-1] for row in group].count(category) / len(group)
               score += p * p
           gini_index += (1 - score) * (len(group) / len(left + right))
       return gini_index
   ```
   We use the above piece of code in this way:

   ```python
   A = [50, 1.62, 18, 'Female']
   B = [70, 1.81, 16, 'Male']
   C = [60, 1.72, 15, 'Female']
   D = [70, 1.71, 19, 'Male']
   E = [52, 1.69, 17, 'Female']
   left = [A]
   right = [B, C, D, E]
   gini_index = get_gini_index(left, right, ['Male', 'Female'])
   print(gini_index) # 0.4
   ```

# Use multiple trees to boost the accuracy

You may wonder why it's called the random forest when we only used one tree? Good question! In fact, we shouldn't only use one tree. The correct process is as follows.

1. Choose 90% of the data randomly for training.
2. Train those data, i.e. the process of building a tree shown above.
3. Use this tree to predict, get the prediction `x`.
4. Repeat the above three steps, build another tree, get another prediction `y`.
5. Repeat the first three steps again, get another prediction `z`.
6. Choose the one that appears the most in `x, y, z`, which should be our final prediction, return it.

So you should know why it's call random forest, right? We built 3 trees in total, and got the final result based on 3 predictions obtained from 3 trees. The number 3 can be changed, too. You can also build 5 trees, 10 trees, etc., whatever works out for you. Moreover, the sampling ratio 90% can be changed, too. 80%, 70%, whatever you like.

The purpose of building multiple trees is to avoid overfitting. From Wikipedia:

> In statistics, **overfitting** is "the production of an analysis that corresponds too closely or exactly to a particular set of data, and may therefore fail to fit additional data or predict future observations reliably".

# Code

Now that we know how it works, it's time for us to dive into the code. Notice that some parameters in the code are not mentioned before, so let's review them together.

1. min_size: when the number of data in some node is less than `min_size`, further splitting is not allowed. I.e., the current group is taken as a leaf node, the value of the leaf node is determined by the category that appears the most in the group.
2. max_depth: The maximum depth of a tree, further splitting is not allowed when `max_depth` is exceeded, the value of the node is determined by the category that appears the most in the group.
3. n_features: The number of features chosen to build the current tree. In case if you don't know what a feature is, weight, height, 100m-race time are both called features in the previous example. We choose `n_features` features for training each time we build a tree. In this way, features used in each tree is different, which means the final trees we build will be different, so overfitting could be avoid.

Code to implement random forest is as follows.

```python
import random


class Node:
    def __init__(self, data):

        # all the data that is held by this node
        self.data = data

        # left child node
        self.left = None

        # right child node
        self.right = None

        # category if the current node is a leaf node
        self.category = None

        # a tuple: (row, column), representing the point where we split the data
        # into the left/right node
        self.split_point = None


def build_model(train_data, n_trees, max_depth, min_size, n_features, n_sample_rate):
    trees = []
    for i in range(n_trees):
        random.shuffle(train_data)
        n_samples = int(len(train_data) * n_sample_rate)
        tree = build_tree(train_data[: n_samples], 1, max_depth, min_size, n_features)
        trees.append(tree)
    return trees


def predict_with_single_tree(tree, row):
    if tree.category is not None:
        return tree.category
    x, y = tree.split_point
    split_value = tree.data[x][y]
    if row[y] <= split_value:
        return predict_with_single_tree(tree.left, row)
    else:
        return predict_with_single_tree(tree.right, row)


def predict(trees, row):
    prediction = []
    for tree in trees:
        prediction.append(predict_with_single_tree(tree, row))
    return max(set(prediction), key=prediction.count)


def get_most_common_category(data):
    categories = [row[-1] for row in data]
    return max(set(categories), key=categories.count)


def build_tree(train_data, depth, max_depth, min_size, n_features):
    root = Node(train_data)
    x, y = get_split_point(train_data, n_features)
    left_group, right_group = split(train_data, x, y)
    if len(left_group) == 0 or len(right_group) == 0 or depth >= max_depth:
        root.category = get_most_common_category(left_group + right_group)
    else:
        root.split_point = (x, y)
        if len(left_group) < min_size:
            root.left = Node(left_group)
            root.left.category = get_most_common_category(left_group)
        else:
            root.left = build_tree(left_group, depth + 1, max_depth, min_size, n_features)

        if len(right_group) < min_size:
            root.right = Node(right_group)
            root.right.category = get_most_common_category(right_group)
        else:
            root.right = build_tree(right_group, depth + 1, max_depth, min_size, n_features)
    return root


def get_features(n_selected_features, n_total_features):
    features = [i for i in range(n_total_features)]
    random.shuffle(features)
    return features[:n_selected_features]


def get_categories(data):
    return set([row[-1] for row in data])


def get_split_point(data, n_features):
    n_total_features = len(data[0]) - 1
    features = get_features(n_features, n_total_features)
    categories = get_categories(data)
    x, y, gini_index = None, None, None
    for index in range(len(data)):
        for feature in features:
            left, right = split(data, index, feature)
            current_gini_index = get_gini_index(left, right, categories)
            if gini_index is None or current_gini_index < gini_index:
                x, y, gini_index = index, feature, current_gini_index
    return x, y


def get_gini_index(left, right, categories):
    gini_index = 0
    for group in left, right:
        if len(group) == 0:
            continue
        score = 0
        for category in categories:
            p = [row[-1] for row in group].count(category) / len(group)
            score += p * p
        gini_index += (1 - score) * (len(group) / len(left + right))
    return gini_index


def split(data, x, y):
    split_value = data[x][y]
    left, right = [], []
    for row in data:
        if row[y] <= split_value:
            left.append(row)
        else:
            right.append(row)
    return left, right

```

So how to use this piece of code? Let's take [Sonar](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+%28sonar,+mines+vs.+rocks%29), which is real-life data as an example(You can have a glimpse of its contents in [here](https://github.com/searene/demos/blob/master/RandomForest/resources/sonar.all-data.csv)). The last column in Sonar represents category, which are two of them in total, R and M. R means rock and M means metal. The first 60 columns represents data obtained by bouncing sonar signals off a surface(R or M) at various angles and under various conditions. Let's load these data and split them into two groups, one for training and one for testing. Training data is used to build models, and test data is used to check the accuracy of the model.

The code is as follows.

```python
import random
import numpy as np
import pandas as pd
from math import sqrt

from random_forest import build_model, predict


class CrossValidationSplitter:
    def __init__(self, data, k_fold):
        self.data = data
        self.k_fold = k_fold
        self.n_iteration = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.n_iteration >= self.k_fold:
            raise StopIteration
        self.n_iteration += 1
        return self.__load_data()

    def __load_data(self):
        n_train_data = (1 / self.k_fold) * len(self.data)
        data_copy = self.data[:]
        train_data = []
        while len(train_data) < n_train_data:
            train_data.append(self.__pop_random_row(data_copy))
        test_data = data_copy
        return train_data, test_data

    def __pop_random_row(self, data):
        random.shuffle(data)
        return data[0]


def split_data(data, rate):
    random.shuffle(data)
    n_train_data = int(len(data) * rate)
    return data[: n_train_data], data[n_train_data:]


def calculate_accuracy(model, validate_data):
    n_total = 0
    n_correct = 0
    predicted_categories = [predict(model, row[:-1]) for row in validate_data]
    correct_categories = [row[-1] for row in validate_data]
    for predicted_category, correct_category in zip(predicted_categories, correct_categories):
        n_total += 1
        if predicted_category == correct_category:
            n_correct += 1
    return n_correct / n_total


df = pd.read_csv('resources/sonar.all-data.csv', header=None)
data = df.values.tolist()
train_data_all, test_data = split_data(data, 0.9)

for n_tree in [1, 3, 10]:
    accuracies = []
    cross_validation_splitter = CrossValidationSplitter(train_data_all, 5)
    model = None
    for train_data, validate_data in cross_validation_splitter:
        n_features = int(sqrt(len(train_data[0]) - 1))
        model = build_model(
            train_data=train_data,
            n_trees=n_tree,
            max_depth=5,
            min_size=1,
            n_features=n_features,
            n_sample_rate=0.9
        )
        accuracies.append(calculate_accuracy(model, validate_data))
    print("Average cross validation accuracy for {} trees: {}".format(n_tree, np.mean(accuracies)))
    print("Test accuracy for {} trees: {}".format(n_tree, calculate_accuracy(model, test_data)))
```
The result is as follows.

```
Average cross validation accuracy for 1 trees: 0.6887700534759359
Test accuracy for 1 trees: 0.6190476190476191
Average cross validation accuracy for 3 trees: 0.6898395721925135
Test accuracy for 3 trees: 0.8571428571428571
Average cross validation accuracy for 10 trees: 0.6983957219251338
Test accuracy for 10 trees: 0.7619047619047619
```
As you can see, we get the highest accuracy with 3 trees(around 85%), we have reason to believe that we could get a better result if further tunning is conducted.
