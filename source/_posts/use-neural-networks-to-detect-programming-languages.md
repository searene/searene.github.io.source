title: Use Deep Learning to Detect Programming Languages
date: 2017-11-26 15:56:45
tags: ['Neural Network']
thumbnail: /images/deep-learning.jpg
categories: Coding
---

# Introduction

This post introduces a way to use deep learning to detect programming languages. Take the following code as an example.

```python
def test():
    print("something")
```

We will get an answer `python` if we use the program to be introduced in the post to detect the language of the above code, which is also the correct answer. In fact, through a preliminary test, the accuracy of the program is around 90%. We have reason to believe that we are able to get a better result if the training dataset is larger or further tuning is conducted.

# Execution

First let's try running the program, so we can have an intuitive perspective on what the program is about.

1. Install third-party libraries

   - [Anaconda(Python 3.6+)](https://www.anaconda.com/download/)

   - Gensim

     ```bash
     conda install -c anaconda gensim
     ```

   - Keras

     ```bash
     conda install -c conda-forge keras
     ```

   -  Tensorflow

     ```bash
     pip install tensorflow==1.3.0
     ```

2. Download the program

  ```bash
  git clone git@github.com:searene/demos.git && cd demos/PLDetector-demo
  ```

3. Train the model

   ```bash
   python -m src.neural_network_trainer                                                  
   Using TensorFlow backend.                                                                                             
   ...
   _________________________________________________________________
   Layer (type)                 Output Shape              Param #
   =================================================================
   embedding_1 (Embedding)      (None, 500, 100)          773100
   _________________________________________________________________
   conv1d_1 (Conv1D)            (None, 496, 128)          64128                                                          
   _________________________________________________________________                                                     
   max_pooling1d_1 (MaxPooling1 (None, 248, 128)          0                                                              
   _________________________________________________________________                                                     
   flatten_1 (Flatten)          (None, 31744)             0                                                              
   _________________________________________________________________                                                     
   dense_1 (Dense)              (None, 8)                 253960                                                         
   =================================================================                                                     
   Total params: 1,091,188                                                                                               
   Trainable params: 318,088
   Non-trainable params: 773,100
   _________________________________________________________________
   INFO:root:None
   Epoch 1/10
    - 1s - loss: 0.4304 - acc: 0.8823
   Epoch 2/10
    - 1s - loss: 0.1357 - acc: 0.9657
   Epoch 3/10
    - 1s - loss: 0.0706 - acc: 0.9788
   Epoch 4/10
    - 1s - loss: 0.0392 - acc: 0.9887
   Epoch 5/10
    - 1s - loss: 0.0266 - acc: 0.9927
   Epoch 6/10
    - 1s - loss: 0.0203 - acc: 0.9945
   Epoch 7/10
    - 1s - loss: 0.0169 - acc: 0.9948
   Epoch 8/10
    - 1s - loss: 0.0145 - acc: 0.9956
   Epoch 9/10
    - 1s - loss: 0.0131 - acc: 0.9959                                                                                    
   Epoch 10/10                                                                                                           
    - 1s - loss: 0.0120 - acc: 0.9959                                                                                    
   INFO:root:Test Accuracy: 94.642857
   ```
   We will have three important files as soon as the above step is completed.

   * resources/models/model.h5
   * resources/models/model.json
   * resources/vocab_tokenizer

   We will introduce the three files in detail later on.

4. Detection

   ```bash
   python -m src.detector

   Using TensorFlow backend.
   Python
   ```

   The following python code is detected by default by `detector.py`

   ```python
   def test():
       print("something")
   ```

   Of course you can modify `detector.py` to detect other code.

# Project Structure

Let's first have a rough idea of the project structure. Don't worry, it will only take 1 ~ 2 minutes.

- resources/code/train: training data. The name of each subfolder representes a programming language. There are around 10 code files in each subfolder, i.e. 10 files per programming language for training.

  ![train文件夹结构](/images/2017-11-26-163206_448x610_scrot.png)

- resources/code/test: the same as `resources/code/train` except that it's used for testing accuracy instead of training.
- `models` directory & `vocab_tokenizer`: stored training result
- src/config.py: some constants used in the program
- src/neural_network_trainer.py: code used to train the model
- src/detector.py: code used to load the model and detect programming languages

# How It Works

## Construct Vocabulary

let's first get our heads around the training process, aka the contents in `neural_network_trainer.py`. the first step to train the neural network is to build a vocabulary. Vocabulary is actually a list of words, which consists of some common words in the training data. When we are done with building a vocabulary and start detecting the programming language, we will try splitting the code into a list of words, and remove those which are not in the vocabulary, then we put the remaining words into the neural network for detection.

OK, you might want to ask, why removing words that are not in the vocabulary? Wouldn't it work if we just put all the words into the neural network? Actually, this is impossible. Because each word in the vocabulary is mapped to a word vector, which is constructed during training. So words that are not in the vocabulary don't have word vectors to map, which means the neural network is unable to process this word.

So how do we build the vocabulary? It's fairly easy, we just need to scan all the code in `resources/code/train` and extract common words in it. Those common words will make up our vocabulary. Key code is as follows.


```python
def build_vocab(train_data_dir):
    vocabulary = Counter()
    files = get_files(train_data_dir)
    for f in files:
        words = load_words_from_file(f)
        vocabulary.update(words)

    # remove rare words
    min_count = 5
    vocabulary = [word for word, count in vocabulary.items() if count >= min_count]
    return vocabulary
```

Run `build_vocab` to get the vocabulary.

```python
vocab = build_vocab(config.train_data_dir)
print(vocab) # [..., 'script', 'text', 'head', 'appendChild', 'parentNode', 'removeChild', ...]
```

So, as you can see, the vocabulary is just a list of words, that's it.

## Build vocab_tokenizer

The next step is to build `vocab_tokenizer`. So what is `vocab_tokenzier`? It's a simple variable, you can imagine it as a dictionary, which maps each word in the vocabulary to a number. Why would we map those words to numbers? Because our neural network is only able to run with numbers, rather than strings.

We use `Tokenizer` provided by `Keras` to build `vocab_tokenizer`.

```python
def build_vocab_tokenizer_from_set(vocab):
    vocab_tokenizer = Tokenizer(lower=False, filters="")
    vocab_tokenizer.fit_on_texts(vocab)
    return vocab_tokenizer
```

Then we save this `vocab_tokenizer` as a file, to be used later.

```python
def save_vocab_tokenizer(vocab_tokenzier_location, vocab_tokenizer):
    with open(vocab_tokenzier_location, 'wb') as f:
        pickle.dump(vocab_tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
```

## Build Word Vectors

Before diving into word vectors, we first need to know what they are.

To put it simply, word vectors are just vectors, and each word in the vocabulary is mapped to a word vector. You may still not get it. This may seem too simple, let's take the following Java code as an example.

```java
public static void main(String[] args) {
    System.out.println("something")
}
```

The `word2vec` variable we are building here is actually a dictionary, which is like this(word -> word_vector).

```python
word2vec = {
    'public': [2, 1, 10],
    'static': [2, 1, 9],
    'main': [1, 10, 3],
    'String': [1, 20, 3],
    'args': [1, 40, 3],
    'System': [20, 10, 3],
    'out': [3, 10, 3],
    'println': [1, 39, 3],
    'something': [1, 20, 3]
}
```

Here comes the question. Why would we build word vectors, instead of just using the number given by `vocab_tokenizer`? This is because word vectors have a very special and useful characteristic: **The more close two words are, the smaller their word vectors are**(Note that the calculation of the distance between vectors are of the field of math, which can be dealt with using multiple methods. It doesn't matter if you don't know how to calculate it, you only need to know the distance between vectors can be calculated). This characteristic will boost the accuracy of our neural network dramatically.

For example, `public` and `staic` are only seen together in Java, so the distance between their word vectors should be small. However, `public` and `System` is not that close, i.e. we may only see one of them at a time, so the distance between their word vectors are larger.

Now that we know why it is necessary to build word vectors, the next problem is how we build them. There are multiple ways to do it. Here we use the `Word2Vec` algorithm provided by `gensim` to achieve it. Steps are as follows.

1. Load all the training data, extract those words which are in the vocabulary.
2. Map each word into its respective number by using `vocab_tokenizer`.
3. Put those numbers into `Word2Vec` library and obtain word vectors.

The code is as follows.

```python
def build_word2vec(train_data_dir, vocab_tokenizer):
    all_words = []
    files = get_files(train_data_dir)
    for f in files:
        words = load_words_from_file(f)
        all_words.append([word for word in words if is_in_vocab(word, vocab_tokenizer)])
    model = Word2Vec(all_words, size=100, window=5, workers=8, min_count=1)
    return {word: model[word] for word in model.wv.index2word}
```

## Build the Neural Network

Everything is ready, now it's the time to train the neural network! First we need to know the input and output of the neural network, take the following code as an example.

```python
def test():
    print("something")
```

Map `def`, `test`, `print` and `something` into their respective numbers, we get the input

```python
input = [0, 1, 2, 3]
```

The output of the neural network is the probability of each language.

```python
output = [0.5, 0.1, 0.04, 0.06, 0.1, 0.1, 0.05, 0.05]
```

The code is as follows.

```python
all_languages = ["Python", "C", "Java", "Scala", "Javascript", "CSS", "C#", "HTML"]
```

So we know the above code is most likely to be written by Python, because Python has the most probability(0.5)

Now that we know the input and output, let me introduce how the neural network is constructed. There are three parts in total.

1. Embedding Layer: it's used to map each word into its respective word vector
2. Conv1D, MaxPooling1D: this part is a classic deep learning layer. To put it simply, what it does is extraction and transformation. Refer to corresponding tutorials of deep learning for details.
3. Flatten, Dense: convert the multi-dimensional array into one-dimensional, and output the prediction.

Key code is as follows.

```python
def build_model(train_data_dir, vocab_tokenizer, word2vec):
    weight_matrix = build_weight_matrix(vocab_tokenizer, word2vec)

    # build the embedding layer
    input_dim = len(vocab_tokenizer.word_index) + 1
    output_dim = get_word2vec_dimension(word2vec)
    x_train, y_train = load_data(train_data_dir, vocab_tokenizer)

    embedding_layer = Embedding(input_dim, output_dim, weights=[weight_matrix], input_length=input_length,
                                trainable=False)
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(filters=128, kernel_size=5, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(len(all_languages), activation="sigmoid"))
    logging.info(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, verbose=2)
    return model
```

All right, we built our neural network, not a trivial achievement! Then let's write a function, which uses the neural network to detect test code, check out its accuracy.

```python
def evaluate_model(test_data_dir, vocab_tokenizer, model):
    x_test, y_test = load_data(test_data_dir, vocab_tokenizer)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    logging.info('Test Accuracy: %f' % (acc * 100))
```

As what we have got before, the test accuracy is around 94%~95%, which is good enough. Let's save the neural network as files, so we can load it when detecting.

```python
def save_model(model, model_file_location, weights_file_location):
    os.makedirs(os.path.dirname(model_file_location), exist_ok=True)
    with open(model_file_location, "w") as f:
        f.write(model.to_json())
    model.save_weights(weights_file_location)
```

## Load the Neural Network For Detection

This part is simple, we only need to load `vocab_tokenizer` and the neural network for detection. The code is as follows.

```python
vocab_tokenizer = load_vocab_tokenizer(config.vocab_tokenizer_location)
model = load_model(config.model_file_location, config.weights_file_location)

def to_language(binary_list):
    i = np.argmax(binary_list)
    return all_languages[i]

def get_neural_network_input(code):
    encoded_sentence = load_encoded_sentence_from_string(code, vocab_tokenizer)
    return pad_sequences([encoded_sentence], maxlen=input_length)

def detect(code):
    y_proba = model.predict(get_neural_network_input(code))
    return to_language(y_proba)
```

Use it like this.

```python
code = """
def test():
    print("something")
"""
print(detect(code)) # Python
```

# Summary

All in all, here are the steps to build the neural network.

1. Build vocabulary.
2. Build `vocab_tokenizer` using vocabulary, which is used to convert words into numbers.
3. Load words into `Word2Vec` to build word vectors.
4. Load word vectors into the neural network as part of the input layer.
5. Load all the training data, extract words that are in the vocabulary, convert them into numbers using `vocab_tokenizer`, load them into the neural network for training.

Three steps for detection:

1. Extract words in the code and remove those that are not in the vocabulary.
2. Convert those words into number through `vocab_tokenizer`, and load them into the neural network.
3. Choose the language which has the most probability, which the answer we want.

# Exercise

You may have already found out that, we only saved `vocab_tokenizer` and the neural network(which lies in the model directory), why didn't we save `word2vec` and `vocab`?

# Question

If you have any question, please leave it in the comment below, I'll try to answer it.
