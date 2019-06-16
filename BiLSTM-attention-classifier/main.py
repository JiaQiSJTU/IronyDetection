import os
import json
from json import JSONDecoder
import numpy as np
import tensorflow as tf
import torch
from keras_preprocessing import sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

##meta
task='B'
if task=='A':
    n_classes=2
else:
    n_classes=4
learning_rate = 1e-5
max_epoch=50
log_path = './log/task{}/'.format(task)
exp_path = './exp/task{}/'.format(task)
result_path = './results/task{}/'.format(task)
restore_path = './exp/task{}/epoch_{}'.format(task, 24)#A:48 B:34
max_len =50
emb_size=728
inference= True
if inference:
    batch_size=784
else:
    batch_size = 16
##rnn
rnn_size=100
layer_size=2
dropout_keep_prob=0.5

##attention
attn_size = 200
use_final_state = False
print('task',task)



def load_dict(fname):
    with open(fname, 'r') as file:
        dictionary = json.load(file)
    return dictionary

def load_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        dataList=data['data']
    return dataList

def load_feature(filename):
    with open(filename,'r')as f:
        feature=f.read()
    feature=JSONDecoder().decode(feature)
    return feature

##global
data_folder = "data"
trainA_file = os.path.join(data_folder, 'trainA.json')
trainB_file = os.path.join(data_folder, 'trainB.json')
test_file = os.path.join(data_folder, 'test.json')

train_feature_f = os.path.join(data_folder, 'task3_train_feature.txt')
test_feature_f = os.path.join(data_folder, 'task3_test_feature.txt')

train_feature = load_feature(train_feature_f)
test_feature = load_feature(test_feature_f)

word2idx_f = os.path.join(data_folder, 'word2idx.json')
pos2idx_f = os.path.join(data_folder, 'pos2idx.json')
word_embeds_f = os.path.join(data_folder, 'word_embedding.npy')
pos_embeds_f = os.path.join(data_folder, 'pos_embedding.npy')


##prepare data
train_totalA = load_data(trainA_file)
train_totalB = load_data(trainB_file)

trainA = train_totalA[:3450]
validA = train_totalA[3450:]
trainB = train_totalB[:3450]
validB = train_totalB[3450:]
train_fea = train_feature[:3450]
valid_fea = train_feature[3450:]
test = load_data(test_file)

word2idx = load_dict(word2idx_f)
pos2idx = load_dict(pos2idx_f)
vocab_size = len(word2idx)

word_embedding = np.load(word_embeds_f)
pos_embedding = np.load(pos_embeds_f)


class IronyDataset(Dataset):
    def __init__(self, raw_data, feature_data):
        self.data = raw_data
        self.feature = feature_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raw_samples = self.data[index]
        length= len(raw_samples['word'])
        sentence = []
        for i in range(length):
            word_emb = word_embedding[word2idx[raw_samples['word'][i]]]
            pos_emb = pos_embedding[pos2idx[raw_samples['pos'][i]]]
            sample = np.concatenate([word_emb, pos_emb])
            sentence.append(sample)
        label = self.data[index]["label"]
        feature = self.feature[index]
        return sentence, label, feature

def collate_fn(batch):
    # sorted_batch = sorted(batch, key=lambda x: x[0].size(0), reverse=True)
    sequences = [x[0] for x in batch]
    sequences_padded = []
    for sequence in sequences:
        # sequence_padded =np.pad(np.array(sequence),
        #                             [(0, max_len - len(sequence))], 'constant', constant_values=(0, 0))
        sequence=np.array(sequence)
        sequence_padded = np.concatenate([sequence,np.zeros(shape=[max_len-sequence.shape[0],sequence.shape[1]])], axis=0)
        sequences_padded.append(sequence_padded)
    sequences_padded = np.array(sequences_padded)
    lengths = np.array([len(x) for x in sequences])
    labels = np.array(list(map(lambda x: x[1], batch)))
    features = np.array(list(map(lambda x: x[2], batch)))
    return sequences_padded, labels, lengths, features
if task=='A':
    train_data = IronyDataset(trainA, train_fea)
    valid_data = IronyDataset(validA, valid_fea)
else:
    train_data = IronyDataset(trainB, train_fea)
    valid_data = IronyDataset(validB, valid_fea)
test_data = IronyDataset(test, test_feature)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, collate_fn=collate_fn)


#########build graph######################################
# placeholder
output_keep_prob = tf.placeholder(tf.float32, name='output_keep_prob')
input_data = tf.placeholder(tf.float32, shape=[batch_size, max_len, emb_size], name='input_data')
targets = tf.placeholder(tf.int64, shape=[batch_size], name='targets')
input_data_len = tf.placeholder(tf.int32, shape=[batch_size], name='input_data_len')
mask = tf.sequence_mask(input_data_len, maxlen=max_len)
fc_feature = tf.placeholder(tf.float32, shape=[batch_size, 149], name='fc_feature')
# imbalance_weight = tf.convert_to_tensor([1.9907674552798615,2.7555910543130993,12.23404255319149, 18.852459016393443], dtype=tf.float32)
imbalance_weight = tf.convert_to_tensor([1,1,2,3], dtype=tf.float32)
# forward
with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
    lstm_fw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
    lstm_fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list), output_keep_prob=output_keep_prob)

#backward
with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
    lstm_bw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
    lstm_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_bw_cell_list), output_keep_prob=output_keep_prob)


with tf.name_scope('bi_rnn'), tf.variable_scope('bi_rnn'):
    outputs, final_state  = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_m, lstm_bw_cell_m, input_data, dtype=tf.float32)
    outputs = tf.concat(outputs, axis= 2)
    final_state = tf.concat(final_state, axis = 2)


with tf.name_scope('attention'),tf.variable_scope('attention'):
    attention_w = tf.Variable(tf.truncated_normal([2*rnn_size, attn_size], stddev=0.1), name='attention_w')
    attention_b = tf.Variable(tf.constant(0.1, shape=[attn_size]), name='attention_b')
    if use_final_state:
        u_w = final_state
    else:
        u_w = tf.Variable(tf.truncated_normal([attn_size, 1], stddev=0.1), name='attention_uw')
    attn_z = tf.tanh(tf.matmul(outputs, tf.tile(tf.expand_dims(attention_w,axis=0), multiples=[batch_size ,1,1]))+attention_b)
    attn_z = tf.reduce_mean(tf.matmul(attn_z, tf.tile(tf.expand_dims(u_w,axis=0), multiples=[batch_size ,1,1])), axis=-1)
    print(attn_z)
    attn_z_mask = attn_z - 1e20*tf.cast(mask, tf.float32)
    print(mask, attn_z_mask)
    alpha = tf.nn.softmax(attn_z_mask)
    print(alpha)
    final_outputs = tf.reduce_sum(outputs*tf.expand_dims(alpha,axis=-1),axis=1)
    print(outputs)

with tf.name_scope('fc'),tf.variable_scope('fc'):
    fc_w = tf.Variable(tf.truncated_normal([2 * rnn_size+149, n_classes], stddev=0.1), name='fc_w')
    fc_b = tf.Variable(tf.zeros([n_classes]), name='fc_b')


logits  =tf.matmul(tf.concat([final_outputs, fc_feature], axis=1), fc_w)+fc_b
prob = tf.nn.softmax(logits)
if task=='B':
    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)*tf.nn.embedding_lookup(imbalance_weight, targets))
else:
    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels=targets))
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(total_loss)
accuracy = tf.reduce_mean(tf.cast( tf.equal(targets, tf.argmax(prob, axis=1)),tf.float32))
predictions = tf.argmax(prob, axis=1)
# tf.summary.scalar('train_loss', total_loss)
# tf.summary.scalar('accuracy', accuracy)
# merged = tf.summary.merge_all()

saver = tf.train.Saver(max_to_keep=100)

####################################################

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
train_writer = tf.summary.FileWriter(log_path+'train')
valid_writer = tf.summary.FileWriter(log_path+'valid')



with tf.Session(config=tf_config) as sess:
    if os.path.exists(os.path.join(exp_path, 'checkpoint')):
        saver.restore(sess, restore_path)
    else:
        sess.run(tf.global_variables_initializer())
    graph_writer = tf.summary.FileWriter(log_path + 'graph')
    graph_writer.add_graph(sess.graph)


    if inference:
        for sequences, labels, lengths, features in tqdm(test_loader, desc='test'):
            # print('input data',sequences,labels, lengths)
            results = sess.run([predictions], feed_dict={
                input_data: sequences,
                input_data_len: lengths.reshape(-1),
                output_keep_prob: 1.0,
                fc_feature: features
            })
            np.savetxt(X=results, fname=result_path+'result')


    else:
        for epoch in range(max_epoch):
            step = epoch * len(train_data)
            test_loss, test_acc = [] , []
            for sequences, labels, lengths, features in tqdm(valid_loader, desc='test'):
                # print('input data',sequences,labels, lengths)
                loss, acc = sess.run([total_loss, accuracy], feed_dict={
                    input_data:sequences,
                    input_data_len:lengths.reshape(-1),
                    targets:labels,
                    output_keep_prob: 1.0,
                    fc_feature:features
                })
                test_loss.append(loss)
                test_acc.append(acc)
            print('epoch_{}'.format(epoch+1),'test loss:', np.mean(test_loss), 'test acc', np.mean(test_acc))
            valid_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=np.mean(test_loss))]),
                                     epoch)
            valid_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='acc', simple_value=np.mean(test_acc))]), epoch)


            train_loss, train_acc = [], []
            for sequences, labels, lengths, features in tqdm(train_loader, desc='train'):
                # print('input data', sequences.shape, labels.shape, lengths.shape, features.shape)
                _, loss, acc = sess.run([train_op, total_loss, accuracy], feed_dict={
                    input_data: sequences,
                    input_data_len: lengths.reshape(-1),
                    targets: labels,
                    output_keep_prob: dropout_keep_prob,
                    fc_feature:features
                })
                train_loss.append(loss)
                train_acc.append(acc)

            print('epoch_{}'.format(epoch+1),'train loss:', np.mean(train_loss), 'train acc', np.mean(train_acc))
            train_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=np.mean(train_loss))]),epoch)
            train_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='acc', simple_value=np.mean(train_acc))]), epoch)


            if (epoch+1)%2==0:
                print(epoch+1, 'saved')
                saver.save(sess, exp_path+'epoch_{}'.format(epoch+1))











