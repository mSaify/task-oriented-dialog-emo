from tqdm import tqdm
from os import mkdir
from os.path import join, exists

import numpy as np
import pandas as pd
import tensorflow as tf

w = 2 ** np.arange(50)

def create_datasets_from_csv(tokenizer, data_path, buffer_size, batch_size, max_length, extra_paths = None):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))

    SOS_ID = tokenizer.encode('<s>')[0]
    EOS_ID = tokenizer.encode('</s>')[0]

    print('Reading the csv file...')
    df = pd.read_csv(data_path)
    if extra_paths is not None:
        dfs_extra = []
        for extra_path in extra_paths:
            df_extra = pd.read_csv(extra_path).drop_duplicates(subset = ['os_dialog'])
            dfs_extra.append(df_extra)

    def read_split_ids(read_path):
        all_dialog_ids = []
        with open(read_path, 'r') as f:
            for line in f:
                items = line.strip().split(',')
                dialog_ids = [int(i) for i in items[1:]]
                all_dialog_ids += dialog_ids
        return all_dialog_ids

    train_ids = read_split_ids('../amt/train_ids.txt')
    val_ids = read_split_ids('../amt/val_ids.txt')
    test_ids = read_split_ids('../amt/test_ids.txt')

    emot2id = {}
    with open('ebp_labels.txt', 'r') as f:
        for line in f:
            emot, index = line.strip().split(',')
            emot2id[emot] = int(index)

    def create_dataset(split_ids, extra = False):
        df_split = df[df['DialogID'].isin(split_ids)]
        df_split = df_split[df_split['Agreed emotion (2/3 vote)'] != '-']

        dialogs = df_split['Dialog'].tolist()
        labels = df_split['Agreed emotion (2/3 vote)'].tolist()

        if extra and extra_paths is not None:
            for df_extra in dfs_extra:
                df_extra_f = df_extra[~df_extra['os_dialog_id'].isin(test_ids)]
                dialogs += df_extra_f['os_dialog'].tolist()
                labels += df_extra_f['emotion'].tolist()

        labels = [emot2id[label.lower().replace(' (other)', '')] for label in labels]
        labels = np.array(labels, dtype = np.int32)

        inputs = np.ones((len(dialogs), max_length), dtype = np.int32)
        weights = np.ones((len(dialogs), max_length), dtype = np.float32)
        for i, dialog in tqdm(enumerate(dialogs), total = len(dialogs)):
            uttrs = dialog.split('\n')
            for j in range(len(uttrs)):
                if uttrs[j].startswith('- '):
                    uttrs[j] = uttrs[j][2:]
            uttr_ids = []
            weight = []
            total_weight = np.sum(w[:len(uttrs)])
            for j in range(len(uttrs)-1, -1, -1):
                encoded = tokenizer.encode(uttrs[j])
                weight += [w[j] / total_weight] * (len(encoded) + 2)
                uttr_ids += [EOS_ID] + encoded + [EOS_ID]
            uttr_ids = uttr_ids[:max_length]
            weight = weight[:max_length]
            uttr_ids[0] = SOS_ID
            uttr_ids[-1] = EOS_ID
            inputs[i,:len(uttr_ids)] = uttr_ids
            weights[i,:len(uttr_ids)] = weight

        assert inputs.shape[0] == labels.shape[0]
        print('Created dataset with {} examples.'.format(inputs.shape[0]))

        return inputs, weights, labels

    train_inputs, train_weights, train_labels = create_dataset(train_ids, extra = True)
    val_inputs, val_weights, val_labels = create_dataset(val_ids)

    train_dataset = (tf.data.Dataset.from_tensor_slices(train_inputs),
        tf.data.Dataset.from_tensor_slices(train_weights),
        tf.data.Dataset.from_tensor_slices(train_labels))
    val_dataset = (tf.data.Dataset.from_tensor_slices(val_inputs),
        tf.data.Dataset.from_tensor_slices(val_weights),
        tf.data.Dataset.from_tensor_slices(val_labels))

    train_dataset = tf.data.Dataset.zip(train_dataset).shuffle(buffer_size).batch(batch_size)
    val_dataset = tf.data.Dataset.zip(val_dataset).batch(batch_size)

    return train_dataset, val_dataset

def create_test_dataset_from_csv(tokenizer, batch_size, max_length):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))

    SOS_ID = tokenizer.encode('<s>')[0]
    EOS_ID = tokenizer.encode('</s>')[0]

    print('Reading the csv file...')
    df = pd.read_csv('/Users/msaify/repos/IITHAssignments/capstone/final_problem/EDOS-main/Data/Training data for the dialogue emotion classifier/1. MTurk_groundtruth_labels_9K.csv')

    test_ids = []
    with open('../amt/test_ids.txt', 'r') as f:
        for line in f:
            items = line.strip().split(',')
            dialog_ids = [int(i) for i in items[1:]]
            test_ids += dialog_ids

    emot2id = {}
    with open('ebp_labels.txt', 'r') as f:
        for line in f:
            emot, index = line.strip().split(',')
            emot2id[emot] = int(index)

    def create_dataset(split_ids):
        df_split = df[df['DialogID'].isin(split_ids)]
        df_split = df_split[df_split['Agreed emotion (2/3 vote)'] != '-']

        labels = df_split['Agreed emotion (2/3 vote)'].tolist()
        labels = [emot2id[label.lower().replace(' (other)', '')] for label in labels]
        labels = np.array(labels, dtype = np.int32)

        dialogs = df_split['Dialog'].tolist()
        inputs = np.ones((len(dialogs), max_length), dtype = np.int32)
        weights = np.ones((len(dialogs), max_length), dtype = np.float32)
        for i, dialog in tqdm(enumerate(dialogs), total = len(dialogs)):
            uttrs = dialog.split('\n')
            for j in range(len(uttrs)):
                if uttrs[j].startswith('- '):
                    uttrs[j] = uttrs[j][2:]
            uttr_ids = []
            weight = []
            total_weight = np.sum(w[:len(uttrs)])
            for j in range(len(uttrs)-1, -1, -1):
                encoded = tokenizer.encode(uttrs[j])
                weight += [w[j] / total_weight] * (len(encoded) + 2)
                uttr_ids += [EOS_ID] + encoded + [EOS_ID]
            uttr_ids = uttr_ids[:max_length]
            weight = weight[:max_length]
            uttr_ids[0] = SOS_ID
            uttr_ids[-1] = EOS_ID
            inputs[i,:len(uttr_ids)] = uttr_ids
            weights[i,:len(uttr_ids)] = weight

        assert inputs.shape[0] == labels.shape[0]
        print('Created dataset with {} examples.'.format(inputs.shape[0]))

        return inputs, weights, labels

    test_inputs, test_weights, test_labels = create_dataset(test_ids)
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_inputs),
        tf.data.Dataset.from_tensor_slices(test_weights),
        tf.data.Dataset.from_tensor_slices(test_labels))
    test_dataset = tf.data.Dataset.zip(test_dataset).batch(batch_size)

    return test_dataset

def create_os_dataset(tokenizer, batch_size, max_length):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))

    SOS_ID = tokenizer.encode('<s>')[0]
    EOS_ID = tokenizer.encode('</s>')[0]

    if not exists('../high_conf_data'):
        mkdir('../high_conf_data')

    if not exists('../high_conf_data/inputs.npy'):
        print('Reading the csv file...')
        df = pd.read_csv('../os_2018_dialogs_emobert_reduce_freq.csv')

        df_amt = pd.read_csv('../MTurkHitsAll_results.csv')
        df_sim = pd.read_csv('../similar_dialogs_3k.csv')
        amt_ids = df_amt['DialogID'].tolist()
        sim_ids = df_sim['os_dialog_id'].tolist()

        print('Excluding the existing ids...')
        df_ex = df[~df['dialogue_id'].isin(amt_ids + sim_ids)]
        N_rows = df_ex.shape[0]
        N_dialogs = len(set(df_ex['dialogue_id'].tolist()))

        print('Getting the dialogs and their ids...')
        dialog_ids = []
        current_dialog_id = -1
        inputs = np.ones((N_dialogs, max_length), dtype = np.int32)
        weights = np.ones((N_dialogs, max_length), dtype = np.float32)
        s = 0
        n = 0
        for i in tqdm(range(N_rows)):
            dialog_id = df_ex.iloc[i]['dialogue_id']
            if dialog_id != current_dialog_id:
                if current_dialog_id != -1:
                    dialog_ids.append(current_dialog_id)
                    dialog = df_ex.iloc[s:i]['text'].tolist()
                    dialog = dialog[-50:]  # cut the dialog if it's too long
                    uttr_ids = []
                    weight = []
                    total_weight = np.sum(w[:len(dialog)])
                    for j in range(len(dialog)-1, -1, -1):
                        encoded = tokenizer.encode(dialog[j])
                        weight += [w[j] / total_weight] * (len(encoded) + 2)
                        uttr_ids += [EOS_ID] + encoded + [EOS_ID]
                        if len(uttr_ids) >= max_length:
                            break
                    uttr_ids = uttr_ids[:max_length]
                    weight = weight[:max_length]
                    uttr_ids[0] = SOS_ID
                    uttr_ids[-1] = EOS_ID
                    inputs[n,:len(uttr_ids)] = uttr_ids
                    weights[n,:len(uttr_ids)] = weight
                    n += 1
                s = i
                current_dialog_id = dialog_id

        dialog_ids.append(current_dialog_id)
        dialog = df_ex.iloc[s:N_rows]['text'].tolist()
        dialog = dialog[-50:]  # cut the dialog if it's too long
        uttr_ids = []
        weight = []
        total_weight = np.sum(w[:len(dialog)])
        for j in range(len(dialog)-1, -1, -1):
            encoded = tokenizer.encode(dialog[j])
            weight += [w[j] / total_weight] * (len(encoded) + 2)
            uttr_ids += [EOS_ID] + encoded + [EOS_ID]
            if len(uttr_ids) >= max_length:
                break
        uttr_ids = uttr_ids[:max_length]
        weight = weight[:max_length]
        uttr_ids[0] = SOS_ID
        uttr_ids[-1] = EOS_ID
        inputs[n,:len(uttr_ids)] = uttr_ids
        weights[n,:len(uttr_ids)] = weight

        dialog_ids = np.array(dialog_ids)

        print('n = {}'.format(n))
        print('inputs.shape = {}'.format(inputs.shape))
        print('weights.shape = {}'.format(weights.shape))
        print('dialog_ids.shape = {}'.format(dialog_ids.shape))

        print('Saving the encoded data to files...')
        np.save('../high_conf_data/inputs.npy', inputs)
        np.save('../high_conf_data/weights.npy', weights)
        np.save('../high_conf_data/dialog_ids.npy', dialog_ids)

    print('Loading inputs and weights...')
    inputs = np.load('../high_conf_data/inputs.npy')
    weights = np.load('../high_conf_data/weights.npy')

    os_dataset = (tf.data.Dataset.from_tensor_slices(inputs),
                  tf.data.Dataset.from_tensor_slices(weights))
    os_dataset = tf.data.Dataset.zip(os_dataset).batch(batch_size)

    return os_dataset, inputs.shape[0]

def create_osed_dataset(tokenizer, batch_size, max_length):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))

    SOS_ID = tokenizer.encode('<s>')[0]
    EOS_ID = tokenizer.encode('</s>')[0]

    if not exists('../osed_data'):
        mkdir('../osed_data')

    if not exists('../osed_data/inputs.npy'):
        print('Reading the csv file...')
        df = pd.read_csv('../osed_dialogs.csv')
        N_rows = df.shape[0]
        N_dialogs = len(set(df['dialogue_id'].tolist()))

        print('Getting the dialogs and their ids...')
        dialog_ids = []
        current_dialog_id = -1
        inputs = np.ones((N_dialogs, max_length), dtype = np.int32)
        weights = np.ones((N_dialogs, max_length), dtype = np.float32)
        s = 0
        n = 0
        for i in tqdm(range(N_rows)):
            dialog_id = df.iloc[i]['dialogue_id']
            if dialog_id != current_dialog_id:
                if current_dialog_id != -1:
                    dialog_ids.append(current_dialog_id)
                    dialog = df.iloc[s:i]['text'].tolist()
                    dialog = dialog[-50:]  # cut the dialog if it's too long
                    uttr_ids = []
                    weight = []
                    total_weight = np.sum(w[:len(dialog)])
                    for j in range(len(dialog)-1, -1, -1):
                        encoded = tokenizer.encode(dialog[j])
                        weight += [w[j] / total_weight] * (len(encoded) + 2)
                        uttr_ids += [EOS_ID] + encoded + [EOS_ID]
                        if len(uttr_ids) >= max_length:
                            break
                    uttr_ids = uttr_ids[:max_length]
                    weight = weight[:max_length]
                    uttr_ids[0] = SOS_ID
                    uttr_ids[-1] = EOS_ID
                    inputs[n,:len(uttr_ids)] = uttr_ids
                    weights[n,:len(uttr_ids)] = weight
                    n += 1
                s = i
                current_dialog_id = dialog_id

        dialog_ids.append(current_dialog_id)
        dialog = df.iloc[s:N_rows]['text'].tolist()
        dialog = dialog[-50:]  # cut the dialog if it's too long
        uttr_ids = []
        weight = []
        total_weight = np.sum(w[:len(dialog)])
        for j in range(len(dialog)-1, -1, -1):
            encoded = tokenizer.encode(dialog[j])
            weight += [w[j] / total_weight] * (len(encoded) + 2)
            uttr_ids += [EOS_ID] + encoded + [EOS_ID]
            if len(uttr_ids) >= max_length:
                break
        uttr_ids = uttr_ids[:max_length]
        weight = weight[:max_length]
        uttr_ids[0] = SOS_ID
        uttr_ids[-1] = EOS_ID
        inputs[n,:len(uttr_ids)] = uttr_ids
        weights[n,:len(uttr_ids)] = weight

        dialog_ids = np.array(dialog_ids)

        print('n = {}'.format(n))
        print('inputs.shape = {}'.format(inputs.shape))
        print('weights.shape = {}'.format(weights.shape))
        print('dialog_ids.shape = {}'.format(dialog_ids.shape))

        print('Saving the encoded data to files...')
        np.save('../osed_data/inputs.npy', inputs)
        np.save('../osed_data/weights.npy', weights)
        np.save('../osed_data/dialog_ids.npy', dialog_ids)

    print('Loading inputs and weights...')
    inputs = np.load('../osed_data/inputs.npy')
    weights = np.load('../osed_data/weights.npy')

    os_dataset = (tf.data.Dataset.from_tensor_slices(inputs),
                  tf.data.Dataset.from_tensor_slices(weights))
    os_dataset = tf.data.Dataset.zip(os_dataset).batch(batch_size)

    return os_dataset, inputs.shape[0]


def create_airline_dataset_each_sentence(tokenizer, batch_size, max_length):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))

    SOS_ID = tokenizer.encode('<s>')[0]
    EOS_ID = tokenizer.encode('</s>')[0]

    if  exists('../airline_data'):
        mkdir('../airline_data')

    if not exists('../airline_data/inputs.npy'):
        print('Reading the csv file...')
        df = pd.read_csv('airline_dialogs_2.csv')
        N_rows = df.shape[0]

        N_dialogs = len(set(df['conversationId'].tolist()))

        print('Getting the dialogs and their ids...')
        dialog_ids = []
        current_dialog_id = -1
        inputs = np.ones((N_rows, max_length), dtype = np.int32)
        weights = np.ones((N_rows, max_length), dtype = np.float32)
        s = 0
        n = 0
        for i in tqdm(range(N_rows)):
            dialog_id = df.iloc[i]['conversationId']
            if dialog_id != current_dialog_id:
                if current_dialog_id != -1:
                    dialog_ids.append(current_dialog_id)
                    dialog = df.iloc[s:i]['utterance'].tolist()
                    dialog = dialog[-50:]  # cut the dialog if it's too long
                    uttr_ids = []
                    weight = []
                    total_weight = np.sum(w[:len(dialog)])
                    for j in range(len(dialog)-1, -1, -1):
                        encoded = tokenizer.encode(dialog[j])
                        weight += [w[j] / total_weight] * (len(encoded) + 2)
                        uttr_ids += [EOS_ID] + encoded + [EOS_ID]
                        inputs[i, ] = uttr_ids
                        if len(uttr_ids) >= max_length:
                            break
                    uttr_ids = uttr_ids[:max_length]
                    weight = weight[:max_length]
                    uttr_ids[0] = SOS_ID
                    uttr_ids[-1] = EOS_ID

                    inputs[i,:len(uttr_ids)] = uttr_ids
                    weights[n,:len(uttr_ids)] = weight
                    n += 1
                s = i
                current_dialog_id = dialog_id

        dialog_ids.append(current_dialog_id)
        dialog = df.iloc[s:N_rows]['utterance'].tolist()
        dialog = dialog[-50:]  # cut the dialog if it's too long
        uttr_ids = []
        weight = []
        total_weight = np.sum(w[:len(dialog)])
        for j in range(len(dialog)-1, -1, -1):
            encoded = tokenizer.encode(dialog[j])
            weight += [w[j] / total_weight] * (len(encoded) + 2)
            uttr_ids += [EOS_ID] + encoded + [EOS_ID]
            if len(uttr_ids) >= max_length:
                break
        uttr_ids = uttr_ids[:max_length]
        weight = weight[:max_length]
        uttr_ids[0] = SOS_ID
        uttr_ids[-1] = EOS_ID
        inputs[n,:len(uttr_ids)] = uttr_ids
        weights[n,:len(uttr_ids)] = weight

        dialog_ids = np.array(dialog_ids)

        print('n = {}'.format(n))
        print('inputs.shape = {}'.format(inputs.shape))
        print('weights.shape = {}'.format(weights.shape))
        print('dialog_ids.shape = {}'.format(dialog_ids.shape))

        print('Saving the encoded data to files...')
        np.save('../airline_data/inputs.npy', inputs)
        np.save('../airline_data/weights.npy', weights)
        np.save('../airline_data/dialog_ids.npy', dialog_ids)

    print('Loading inputs and weights...')
    inputs = np.load('../airline_data/inputs.npy')
    weights = np.load('../airline_data/weights.npy')

    os_dataset = (tf.data.Dataset.from_tensor_slices(inputs),
                  tf.data.Dataset.from_tensor_slices(weights))
    os_dataset = tf.data.Dataset.zip(os_dataset).batch(batch_size)

    return os_dataset, inputs.shape[0]


def create_airline_dataset(tokenizer, batch_size, max_length):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))

    SOS_ID = tokenizer.encode('<s>')[0]
    EOS_ID = tokenizer.encode('</s>')[0]

    if not exists('../airline_data'):
        mkdir('../airline_data')

    if  exists('../airline_data/inputs.npy'):
        print('Reading the csv file...')
        df = pd.read_csv('airline_dialogs.csv')
        N_rows = df.shape[0]

        N_dialogs = len(set(df['conversationId'].tolist()))

        print('Getting the dialogs and their ids...')
        dialog_ids = []
        current_dialog_id = -1
        inputs = np.ones((N_dialogs, max_length), dtype = np.int32)
        weights = np.ones((N_dialogs, max_length), dtype = np.float32)
        s = 0
        n = 0
        for i in tqdm(range(N_rows)):
            dialog_id = df.iloc[i]['conversationId']
            if dialog_id != current_dialog_id:
                if current_dialog_id != -1:
                    dialog_ids.append(current_dialog_id)
                    dialog = df.iloc[s:i]['utterance'].tolist()
                    dialog = dialog[-50:]  # cut the dialog if it's too long
                    uttr_ids = []
                    weight = []
                    total_weight = np.sum(w[:len(dialog)])
                    for j in range(len(dialog)-1, -1, -1):
                        encoded = tokenizer.encode(dialog[j])
                        weight += [w[j] / total_weight] * (len(encoded) + 2)
                        uttr_ids += [EOS_ID] + encoded + [EOS_ID]
                        if len(uttr_ids) >= max_length:
                            break
                    uttr_ids = uttr_ids[:max_length]
                    weight = weight[:max_length]
                    uttr_ids[0] = SOS_ID
                    uttr_ids[-1] = EOS_ID
                    inputs[n,:len(uttr_ids)] = uttr_ids
                    weights[n,:len(uttr_ids)] = weight
                    n += 1
                s = i
                current_dialog_id = dialog_id

        dialog_ids.append(current_dialog_id)
        dialog = df.iloc[s:N_rows]['utterance'].tolist()
        dialog = dialog[-50:]  # cut the dialog if it's too long
        uttr_ids = []
        weight = []
        total_weight = np.sum(w[:len(dialog)])
        for j in range(len(dialog)-1, -1, -1):
            encoded = tokenizer.encode(dialog[j])
            weight += [w[j] / total_weight] * (len(encoded) + 2)
            uttr_ids += [EOS_ID] + encoded + [EOS_ID]
            if len(uttr_ids) >= max_length:
                break
        uttr_ids = uttr_ids[:max_length]
        weight = weight[:max_length]
        uttr_ids[0] = SOS_ID
        uttr_ids[-1] = EOS_ID
        inputs[n,:len(uttr_ids)] = uttr_ids
        weights[n,:len(uttr_ids)] = weight

        dialog_ids = np.array(dialog_ids)

        print('n = {}'.format(n))
        print('inputs.shape = {}'.format(inputs.shape))
        print('weights.shape = {}'.format(weights.shape))
        print('dialog_ids.shape = {}'.format(dialog_ids.shape))

        print('Saving the encoded data to files...')
        np.save('../airline_data/inputs.npy', inputs)
        np.save('../airline_data/weights.npy', weights)
        np.save('../airline_data/dialog_ids.npy', dialog_ids)

    print('Loading inputs and weights...')
    inputs = np.load('../airline_data/inputs.npy')
    weights = np.load('../airline_data/weights.npy')

    os_dataset = (tf.data.Dataset.from_tensor_slices(inputs),
                  tf.data.Dataset.from_tensor_slices(weights))
    os_dataset = tf.data.Dataset.zip(os_dataset).batch(batch_size)

    return os_dataset, inputs.shape[0]

def create_linux_dataset(tokenizer, batch_size, max_length):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))

    SOS_ID = tokenizer.encode('<s>')[0]
    EOS_ID = tokenizer.encode('</s>')[0]

    if not exists('../linux_data'):
        mkdir('../linux_data')

    if not exists('../linux_data/inputs.npy'):
        print('Reading the csv file...')
        df = pd.read_csv('linux_dialogs_200000.csv')
        N_rows = df.shape[0]

        N_dialogs = len(set(df['conversationId'].tolist()))

        print('Getting the dialogs and their ids...')
        dialog_ids = []
        current_dialog_id = -1
        inputs = np.ones((N_dialogs, max_length), dtype = np.int32)
        weights = np.ones((N_dialogs, max_length), dtype = np.float32)
        s = 0
        n = 0
        for i in tqdm(range(N_rows)):
            dialog_id = df.iloc[i]['conversationId']
            if dialog_id != current_dialog_id:
                if current_dialog_id != -1:
                    dialog_ids.append(current_dialog_id)
                    dialog = df.iloc[s:i]['utterance'].tolist()
                    dialog = dialog[-50:]  # cut the dialog if it's too long
                    uttr_ids = []
                    weight = []
                    total_weight = np.sum(w[:len(dialog)])
                    for j in range(len(dialog)-1, -1, -1):
                        try:
                            encoded = tokenizer.encode(dialog[j])
                        except :
                            print("error")
                        weight += [w[j] / total_weight] * (len(encoded) + 2)
                        uttr_ids += [EOS_ID] + encoded + [EOS_ID]
                        if len(uttr_ids) >= max_length:
                            break
                    uttr_ids = uttr_ids[:max_length]
                    weight = weight[:max_length]
                    uttr_ids[0] = SOS_ID
                    uttr_ids[-1] = EOS_ID
                    inputs[n,:len(uttr_ids)] = uttr_ids
                    weights[n,:len(uttr_ids)] = weight
                    n += 1
                s = i
                current_dialog_id = dialog_id

        dialog_ids.append(current_dialog_id)
        dialog = df.iloc[s:N_rows]['utterance'].tolist()
        dialog = dialog[-50:]  # cut the dialog if it's too long
        uttr_ids = []
        weight = []
        total_weight = np.sum(w[:len(dialog)])
        for j in range(len(dialog)-1, -1, -1):
            encoded = tokenizer.encode(dialog[j])
            weight += [w[j] / total_weight] * (len(encoded) + 2)
            uttr_ids += [EOS_ID] + encoded + [EOS_ID]
            if len(uttr_ids) >= max_length:
                break
        uttr_ids = uttr_ids[:max_length]
        weight = weight[:max_length]
        uttr_ids[0] = SOS_ID
        uttr_ids[-1] = EOS_ID
        inputs[n,:len(uttr_ids)] = uttr_ids
        weights[n,:len(uttr_ids)] = weight

        dialog_ids = np.array(dialog_ids)

        print('n = {}'.format(n))
        print('inputs.shape = {}'.format(inputs.shape))
        print('weights.shape = {}'.format(weights.shape))
        print('dialog_ids.shape = {}'.format(dialog_ids.shape))

        print('Saving the encoded data to files...')
        np.save('../linux_data/inputs.npy', inputs)
        np.save('../linux_data/weights.npy', weights)
        np.save('../linux_data/dialog_ids.npy', dialog_ids)

    print('Loading inputs and weights...')
    inputs = np.load('../linux_data/inputs.npy')
    weights = np.load('../linux_data/weights.npy')

    os_dataset = (tf.data.Dataset.from_tensor_slices(inputs),
                  tf.data.Dataset.from_tensor_slices(weights))
    os_dataset = tf.data.Dataset.zip(os_dataset).batch(batch_size)

    return os_dataset, inputs.shape[0]


def create_media_dataset(tokenizer, batch_size, max_length):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))

    SOS_ID = tokenizer.encode('<s>')[0]
    EOS_ID = tokenizer.encode('</s>')[0]

    if not exists('../media_data'):
        mkdir('../media_data')

    if  exists('../media_data/inputs.npy'):
        print('Reading the csv file...')
        df = pd.read_csv('media_dialogs.csv')
        N_rows = df.shape[0]

        N_dialogs = len(set(df['conversationId'].tolist()))

        print('Getting the dialogs and their ids...')
        dialog_ids = []
        current_dialog_id = -1
        inputs = np.ones((N_dialogs, max_length), dtype = np.int32)
        weights = np.ones((N_dialogs, max_length), dtype = np.float32)
        s = 0
        n = 0
        for i in tqdm(range(N_rows)):
            dialog_id = df.iloc[i]['conversationId']
            if dialog_id != current_dialog_id:
                if current_dialog_id != -1:
                    dialog_ids.append(current_dialog_id)
                    dialog = df.iloc[s:i]['utterance'].tolist()
                    dialog = dialog[-50:]  # cut the dialog if it's too long
                    uttr_ids = []
                    weight = []
                    total_weight = np.sum(w[:len(dialog)])
                    for j in range(len(dialog)-1, -1, -1):
                        encoded = tokenizer.encode(dialog[j])
                        weight += [w[j] / total_weight] * (len(encoded) + 2)
                        uttr_ids += [EOS_ID] + encoded + [EOS_ID]
                        if len(uttr_ids) >= max_length:
                            break
                    uttr_ids = uttr_ids[:max_length]
                    weight = weight[:max_length]
                    uttr_ids[0] = SOS_ID
                    uttr_ids[-1] = EOS_ID
                    inputs[n,:len(uttr_ids)] = uttr_ids
                    weights[n,:len(uttr_ids)] = weight
                    n += 1
                s = i
                current_dialog_id = dialog_id

        dialog_ids.append(current_dialog_id)
        dialog = df.iloc[s:N_rows]['utterance'].tolist()
        dialog = dialog[-50:]  # cut the dialog if it's too long
        uttr_ids = []
        weight = []
        total_weight = np.sum(w[:len(dialog)])
        for j in range(len(dialog)-1, -1, -1):
            encoded = tokenizer.encode(dialog[j])
            weight += [w[j] / total_weight] * (len(encoded) + 2)
            uttr_ids += [EOS_ID] + encoded + [EOS_ID]
            if len(uttr_ids) >= max_length:
                break
        uttr_ids = uttr_ids[:max_length]
        weight = weight[:max_length]
        uttr_ids[0] = SOS_ID
        uttr_ids[-1] = EOS_ID
        inputs[n,:len(uttr_ids)] = uttr_ids
        weights[n,:len(uttr_ids)] = weight

        dialog_ids = np.array(dialog_ids)

        print('n = {}'.format(n))
        print('inputs.shape = {}'.format(inputs.shape))
        print('weights.shape = {}'.format(weights.shape))
        print('dialog_ids.shape = {}'.format(dialog_ids.shape))

        print('Saving the encoded data to files...')
        np.save('../media_data/inputs.npy', inputs)
        np.save('../media_data/weights.npy', weights)
        np.save('../media_data/dialog_ids.npy', dialog_ids)

    print('Loading inputs and weights...')
    inputs = np.load('../media_data/inputs.npy')
    weights = np.load('../media_data/weights.npy')

    os_dataset = (tf.data.Dataset.from_tensor_slices(inputs),
                  tf.data.Dataset.from_tensor_slices(weights))
    os_dataset = tf.data.Dataset.zip(os_dataset).batch(batch_size)

    return os_dataset, inputs.shape[0]
