from json import *
import json
import collections
import numpy as np

with open("trainA.json", 'r') as file:
    data = json.load(file)
    trainList=data['data']
print("load trainning data")

with open("test.json", 'r') as file:
    data = json.load(file)
    testList=data['data']
print("load test data")
# word_dict={'PADDING':[0,9999],'UNK':[1,9999]}
# pos_dict={'PADDING':0,'UNK':1}
word_list=['PADDING','UNK']
pos_list=['PADDING','UNK']


for sent in trainList:
    for word in sent['word']:
        if not str(word) in word_list:
            word_list.append(str(word))
    for pos in sent['pos']:
        if not pos in pos_list:
            pos_list.append(pos)

for sent in testList:
    for word in sent['word']:
        if not str(word) in word_list:
            word_list.append(str(word))
    for pos in sent['pos']:
        if not pos in pos_list:
            pos_list.append(pos)

word2id={word:index for index, word in enumerate(word_list)}
# id2word={index:word for index, word in enumerate(word_list)}
pos2id={pos:index for index, pos in enumerate(pos_list)}
# id2pos={index:pos for index, pos in enumerate(pos_list)}

print("Write word2idx")

json_str=json.dumps(word2id)
with open("word2idx.json",'w') as json_file:
    json_file.write(json_str)

print("Write pos2idx")
json_pos=json.dumps(pos2id)
with open("pos2idx.json",'w') as json_file2:
    json_file2.write(json_pos)

posnum=len(pos_list)
posvec=np.arange(posnum)
pos_embedding=(np.arange(posnum)==posvec[:,None]).astype(np.int)

print("Write pos_embedding")
np.save('pos_embedding.npy',pos_embedding)

num=len(word_list)


#-----------------------------------------------------
#读取第一个预训练embedding
print("Load first pretrained embedding")
embdict=dict()
size1=0
with open('word2vec_twitter_model.bin','rb')as f:#3039345
    header = f.readline()
    vocab_size1, layer1_size = map(int, header.split())
    binary_len = np.dtype('float32').itemsize * layer1_size
    for line in range(vocab_size1):
        word = []
        while True:
            ch = f.read(1).decode(errors='ignore')
            if ch ==' ':
                word = ''.join(word)
                break
            if ch != '\n':
                word.append(ch)
        if len(word) != 0:
            tp= np.fromstring(f.read(binary_len), dtype='float32')
            if word in word_list:
                embdict[str(word)]=tp.tolist()
                size1=len(tp.tolist())
        else:
            f.read(binary_len)

#-----------------------------------------------------
#读取第二个预训练embedding
print("Load second pretrained embedding")
embdict2=dict()

#model_swm_300-6-10-low.w2v
size2=0
with open('raw_model_swm-300-6-3.w2v','r',encoding='utf-8',errors='ignore')as f: #600140
    header = f.readline()
    vocab_size2, layer1_size = map(int, header.split())
    binary_len = np.dtype('float32').itemsize * layer1_size
    for line in range(vocab_size2):
        k = f.readline().split()
        word=k[0]
        if len(word) != 0:
            tp=[float(x) for x in k[1:]]
            #tp= np.fromstring(' '.join(k[1:]), dtype='float32')
            if word in word_list:
                embdict2[str(word)]=tp
                size2=len(tp)
        else:
            continue

#------------------------------------------------
#输出word embedding list
print("Get word embedding list")
zero1=np.zeros(size1)
zero2=np.zeros(size2)
print(size1)
print(size2)


wordEmbedding_list=[]
for item in word_list:
    wordembedding1=[]
    wordembedding2=[]

    if item in embdict.keys():
        wordembedding1=embdict[item]
    else:
        wordembedding1=zero1

    if item in embdict2.keys():
        wordembedding2=embdict2[item]
    else:
        wordembedding2=zero2

    wordEmbedding_list.append(np.concatenate((np.array(wordembedding1),np.array(wordembedding2)),axis=0))

wordEmbedding_list=np.asarray(wordEmbedding_list)

print("write word embedding")
np.save('word_embedding.npy',wordEmbedding_list)

