# Gradient Frequency Attention
This is the code for '**Gradient Frequency Attention: Tell Neural Networks where speaker information is.**'

### 1.Datasets

We select Voxceleb1 and Voxceleb2 as our datasets.
In Voxceleb1, there are 1,211 speakers with 148,642 utterances
for training and 40 speakers with 4,870 utterances for
testing. Only the Voxceleb2 development set is used as a
training set. There are 5,994 speakers with 1,092,009 utterances.

### 2.Deep Speaker Verification

#### 2.1.Neural Networks

##### ResCNN

ResCNN is a CNN with Channel Block Attention Blocks for verification as Figure.1. In our testing, this model proved to outperform many SV systems on Voxceleb1
when we input spectrograms into it. Dropout is applied before  the average pooling layer.

![Figure.1](misc/rescnn_drawio.png "Gradient Frequency Attention Framework")

##### TDNN

TDNN is the neural network meantioned in *x-vectors*.

#### 2.2.Gradient Frequency Attention

![Figure.2](misc/attention_drawio.png "Gradient Frequency Attention Framework")


#### 2.3.Loss Function

#### 2.4.Score


### 4.Results





