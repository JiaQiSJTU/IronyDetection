

import os
import torch

DATA_DIR = "data/"
MODEL_DIR = "model/"

class TASK3(object):
    TASK_A = os.path.join(DATA_DIR, "trainA.json")
    TASK_B = os.path.join(DATA_DIR, "trainB.json")

    TEST = os.path.join(DATA_DIR, "test.json")

    TRAIN_FEATURE = os.path.join(DATA_DIR, "task3_train_feature.txt")
    TEST_FEATURE = os.path.join(DATA_DIR, "task3_test_feature.txt")

    WORD2IDX = os.path.join(DATA_DIR, "word2idx.json")
    POS2IDX = os.path.join(DATA_DIR, "pos2idx.json")

    WORDEMBEDDING = os.path.join(DATA_DIR, 'word_embedding.npy')
    POSEMBEDDING = os.path.join(DATA_DIR, 'pos_embedding.npy')

class MODELCONFIG(object):
    n_encoder_layer = 2
    dim_model = 728     # 200+28
    n_heads = 8     # dim_model % n_heads = 0
    d_ff = 877
    # d_ff = 728
    dropout = 0.5
    epoch_A = 3
    epoch_B = 3
    n_hidden = 1024
    label_weights = torch.tensor([1.9907674552798615, 2.7555910543130993, 12.23404255319149, 18.852459016393443])

class TRAINCONFIG(object):
    MODELPATH_A = os.path.join(MODEL_DIR, "taskA_transformer_params_1.pkl")
    MODELPATH_B = os.path.join(MODEL_DIR, "taskB_transformer_params_1.pkl")
    MODELPATH_A_MLP = os.path.join(MODEL_DIR, "taskA_MLP_params.pkl")
    MODELPATH_B_MLP = os.path.join(MODEL_DIR, "taskB_MLP_params.pkl")
    retrain = True




