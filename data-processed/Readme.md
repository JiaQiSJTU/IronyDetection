# Data

**Attention!!!**

train sample 3834--》 please divede the trainning data into training set(3450, i.e. dataList[:3450]) and validation set(384, i.e. dataList[3450:])

Since the data for **taskB** is imbalanced, please add a weight for the loss of each sample according to its label during the training time:
{label:weight}={0.0: 1.9907674552798615, 1.0: 2.7555910543130993, 2.0:12.23404255319149, 3.0: 18.852459016393443}

During the training test, use accuracy as the metric to evaluate the performance on validation set.

for test: （.txt）(label: 0,1,2,3)
please wirte file in this way (one sample a line):
label+\t+orinignal word list\n

Futher analysis of performace on test set is not required.


## **test / trainA / trainB.json**  

* test: test data  
* trainA: training data for task A  
* trainB：training data for task B (label is different from trainA)

```python
with open("*.json", 'r') as file:
    data = json.load(file)
    dataList=data['data']

    for item in dataList:
        wordList=item['word']
        pos=item['pos']
        label=item['label']
```
## **train\_feature / test\_feature**  
149 dimensional feature vector for each sample, just concatenate it with the hidden state before the Fully-Connected Layer.

```python
with open('task3_test_feature.txt','r')as f:
    test_feature=f.read()
test_feature=JSONDecoder().decode(test_feature)
```

## **word2idx.json / pos2idx.json**  
* key:str([word] | [pos])  
* value:idx  
* "PADDING": 0, "UNK": 1

For each input word, get its idx by this dictionay. And then, use this idx to find the corresponding word\_embedding in word\_embedding.npy

```python
with open("*.json", 'r') as file:
    dictionary = json.load(file)
```


## **word\_embedding.npy / pos\_embedding.npy**
```python
 *_embedding = np.load( "*.npy" )
```

