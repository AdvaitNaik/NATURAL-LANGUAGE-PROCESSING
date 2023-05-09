"""
@author: advait naik
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

TRAIN_PATH = '../data/train'
DEV_PATH = '../data/dev'
TEST_PATH = '../data/test'
WORD_TO_INDEX = dict()
TAG_TO_INDEX = dict()
INDEX_TO_TAG = dict() 
INDEX_TO_WORD = dict()

GLOVE_VECTOR = dict()
WORD_TO_INDEX_GLOVE = dict()
INDEX_TO_WORD_GLOVE = dict()

# ------------------------Read Data---------------------------

def data_preparation(input_data: list[str]) -> list:
    """
    read data line and convert line in format [[word1, word2 ..., wordn], [tag1, tag2 ..., tagn]] 
    :param input_data:
    :return:
    """
    sentence = []
    label = []
    data = []
    for line in input_data:
        if not line.isspace():
            part = line.split()
            sentence.append(part[1])
            label.append(part[-1])
        if line.isspace():
            data.append([sentence, label])
            sentence = []
            label = []
    # print(line)
    sentence = []
    label = []
    part = line.split()
    sentence.append(part[1])
    label.append(part[-1])
    data.append([sentence, label])
    return data

def read_data(path: str) -> str:
    """
    read raw data
    :param path: raw data file path
    :return:
    """
    with open(path, 'r', encoding='utf-8') as file:
        input_data = file.readlines()
    # print(data)
    data = data_preparation(input_data) 
    return data

def data_preparation_test(input_data: list[str]) -> list:
    """
    read data line and convert line in format [[word1, word2 ..., wordn], [tag1, tag2 ..., tagn]] 
    :param input_data:
    :return:
    """
    sentence = []
    label = []
    data = []
    for line in input_data:
        if not line.isspace():
            part = line.split()
            sentence.append(part[1])
            label.append('O')
        if line.isspace():
            data.append([sentence, label])
            sentence = []
            label = []
    # print(line)
    sentence = []
    label = []
    part = line.split()
    sentence.append(part[1])
    label.append('O')
    data.append([sentence, label])
    return data

def read_data_test(path: str) -> str:
    """
    read raw data
    :param path: raw data file path
    :return:
    """
    with open(path, 'r', encoding='utf-8') as file:
        input_data = file.readlines()
    # print(data)
    data = data_preparation_test(input_data) 
    return data


# ------------------------Vocab Create---------------------------

def vocab_create1(train_data: list) -> None:
    """
    create WORD_TO_INDEX, TAG_TO_INDEX file 
    convert the data into numerical form
    :param train_data: train read data file
    :return:
    """
    global WORD_TO_INDEX, TAG_TO_INDEX
    TAG_TO_INDEX = {'O': 0, 'B-MISC': 1, 'I-MISC': 2, 'B-PER': 3, 'I-PER': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-LOC': 7, 'I-LOC': 8}
    
    index = 2
    for sentence, _ in train_data:
        for word in sentence:
            if word not in WORD_TO_INDEX:
                WORD_TO_INDEX[word] = index
                index += 1

        # VOCAB.update(sentence)
    # WORD_TO_INDEX = {word: i+2 for i, word in enumerate(VOCAB)}
    WORD_TO_INDEX['<pad>'] = 0
    WORD_TO_INDEX['<unk>'] = 1

def vocab_create2() -> None:
    """
    create WORD_TO_INDEX_GLOVE, GLOVE_VECTOR file 
    convert the data into numerical form
    :param train_data: train read data file
    :return:
    """
    global WORD_TO_INDEX_GLOVE, GLOVE_VECTOR, INDEX_TO_WORD_GLOVE
    
    WORD_TO_INDEX_GLOVE['<pad>'] = 0
    WORD_TO_INDEX_GLOVE['<unk>'] = 1

    with open('glove.6B.100d.txt', 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            line = line.strip().split()
            word = line[0]
            vector = torch.FloatTensor([float(val) for val in line[1:]])
            GLOVE_VECTOR[word] = vector
            WORD_TO_INDEX_GLOVE[word] = i+2

    GLOVE_VECTOR['<pad>'] = torch.zeros(100)
    tensor_list = list(GLOVE_VECTOR.values())
    GLOVE_VECTOR['<unk>'] = torch.mean(torch.stack(tensor_list), dim=0)

    INDEX_TO_WORD_GLOVE = {v: k for k, v in WORD_TO_INDEX_GLOVE.items()}

# ------------------------Encode---------------------------

class Encode1:
    def Encode_Process(self, sentence, label):
        """
        Encode the sentence and labels into numerical form
        """
        sentence = [WORD_TO_INDEX.get(word, 1) for word in sentence]
        label = [TAG_TO_INDEX[lab] for lab in label]
        return sentence, label

    def Encode_Vocab(self, data):
        data = [self.Encode_Process(sentence, label) for sentence, label in data]
        return data
    
class Encode2:
    def Encode_Process(self, sentence, label):
        """
        Encode the sentence and labels into numerical form
        """
        capitalization = []
        for word in sentence:
            if word.lower() == word:
                capitalization.append([0])  # lowercase
            else:
                capitalization.append([1])  # titlecase or mixed case

        sentence = [WORD_TO_INDEX_GLOVE.get(word.lower(), 1) for word in sentence]
        label = [TAG_TO_INDEX[lab] for lab in label]
        return sentence, capitalization, label

    def Encode_Vocab(self, data):
        data = [self.Encode_Process(sentence, label) for sentence, label in data]
        return data
        
# ------------------------Dataset---------------------------

class NERDataset1(data.Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        sentence, label = self.data[index]
        sentence = torch.LongTensor(sentence) # LongTensor
        label = torch.LongTensor(label)
        return sentence, label
    
    def __len__(self):
        return len(self.data)
    
def collate_fn1(batch):
    # batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    sentence, label = zip(*batch)
    sentence = pad_sequence(sentence, batch_first=True, padding_value=0)
    label = pad_sequence(label, batch_first=True, padding_value=-1)
    return sentence, label


class NERDataset2(data.Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        sentence, capitalization, label = self.data[index]
        capitalization = torch.LongTensor(capitalization)
        sentence = torch.LongTensor(sentence)
        label = torch.LongTensor(label)
        return sentence, capitalization, label
        # return sentence, label
    
    def __len__(self):
        return len(self.data)
    
def collate_fn2(batch):
    sentence, capitalization, label = zip(*batch)
    sentence = pad_sequence(sentence, batch_first=True, padding_value=0)
    capitalization = pad_sequence(capitalization, batch_first=True, padding_value=0)
    label = pad_sequence(label, batch_first=True, padding_value=-1)
    return sentence, capitalization, label
    
# ------------------------Model1---------------------------

class BLSTM1(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, lstm_hidden_dim, linear_output_dim, num_layers, dropout):
        super(BLSTM1, self).__init__()
        
        self.embedding = nn.Embedding(input_size, embedding_dim) #padding_idx=0
        self.blstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers=num_layers, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_dim * 2, linear_output_dim)
        self.activation = nn.ELU()
        self.classifier = nn.Linear(linear_output_dim, output_size)
        
    def forward(self, sentence):
        embedded = self.embedding(sentence)
        
        length = torch.sum(sentence != 0, dim=1)
        packed_output = pack_padded_sequence(embedded, length, batch_first=True, enforce_sorted=False)
        blstm_output, _ = self.blstm(packed_output)
        unpacked_output, _ = pad_packed_sequence(blstm_output, batch_first=True)
        
        blstm_output = self.dropout(unpacked_output)
        
        linear_output = self.fc(blstm_output)
        elu_output = self.activation(linear_output)
        
        output = self.classifier(elu_output)
        return output
    
class BLSTM2(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, lstm_hidden_dim, linear_output_dim, num_layers, dropout, glove_embeddings):
        super(BLSTM2, self).__init__()
     
        # self.embedding = nn.Embedding(input_size, embedding_dim)
        self.embedding = nn.Embedding.from_pretrained(glove_embeddings)
       
        self.blstm = nn.LSTM(embedding_dim+1, lstm_hidden_dim, num_layers=num_layers, bidirectional=True) #dropout=dropout
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(lstm_hidden_dim * 2, linear_output_dim)
        self.activation = nn.ELU()
        self.classifier = nn.Linear(linear_output_dim, output_size)
        
    def forward(self, sentence, capitalization):
        
        word_embedded = self.embedding(sentence)

        embedded = torch.cat((word_embedded, capitalization), dim=-1)
        
        length = torch.sum(sentence != 0, dim=1)
        packed_output = pack_padded_sequence(embedded, length, batch_first=True, enforce_sorted=False)
        blstm_output, _ = self.blstm(packed_output)
        unpacked_output, _ = pad_packed_sequence(blstm_output, batch_first=True)
        
        blstm_output = self.dropout(unpacked_output)

        linear_output = self.fc(blstm_output)
        elu_output = self.activation(linear_output)
     
        output = self.classifier(elu_output)
        return output
    
# ------------------------Train Process 1---------------------------

def train_model_checkpoint1(model1, dev_loader):
    model1.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sentence, label in dev_loader:
            output = model1(sentence)
            predicted = output.argmax(dim = -1)
            total += label.numel()
            correct += (predicted == label).sum().item()
    # print("Dev Accuracy %.6f" % (correct/total))
    return (correct/total)

def train_model_process1(train_loader, dev_loader):
    input_size = len(WORD_TO_INDEX)
    output_size = len(TAG_TO_INDEX)
    embedding_dim = 100
    lstm_hidden_dim = 256
    linear_output_dim = 128
    num_layers = 1
    dropout = 0.33
    lr = 0.5
    num_epochs = 20

    model1 = BLSTM1(input_size, output_size, embedding_dim, lstm_hidden_dim, linear_output_dim, num_layers, dropout)
    print(model1)
    optimizer = optim.SGD(model1.parameters(), lr=lr, momentum=0.9) 

    class_weights = torch.tensor([0.7, 1, 1, 1, 1, 1, 1, 1, 1])
    ignore_index = -1 

    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)

    max_accuracy = float("-inf")
    accuracy = 0
    for epoch in range(num_epochs):
        running_loss = 0
        model1.train()
        for sentence, label in train_loader:
            # print(sentence.shape, label.shape)
            optimizer.zero_grad()
            output = model1(sentence)
            output = output.permute(0, 2, 1)
            # print(output.shape, label.shape)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # print('Epoch %d \t Training Loss: %.6f' % (epoch+1, running_loss/len(train_loader)))
        accuracy = train_model_checkpoint1(model1, dev_loader)
        if (accuracy > max_accuracy): 
          max_accuracy = accuracy
          torch.save(model1, "model/blstm1.pt")
        print('Epoch %d \t Training Loss: %.6f \t Dev Accuracy: %.6f' % (epoch+1, running_loss/len(train_loader), max_accuracy))

    print("Model blstm1 Training Completed")
    # Model Saving-
    # torch.save(model1, "model/blstm1.pt")

    
def train_model1(train_data):
    global INDEX_TO_TAG, INDEX_TO_WORD
    encode = Encode1()
    train_data = encode.Encode_Vocab(train_data)

    INDEX_TO_TAG = {index: tag for tag, index in TAG_TO_INDEX.items()}
    INDEX_TO_WORD = {v: k for k, v in WORD_TO_INDEX.items()}

    batch_size = 64
    train_dataset = NERDataset1(train_data)
    train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            collate_fn=collate_fn1)
    
    dev_data = read_data(DEV_PATH)
    encode = Encode1()
    dev_dataset = encode.Encode_Vocab(dev_data)
    dev_dataset = NERDataset1(dev_dataset)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)

    # train_model_process1(train_loader, dev_loader)

def dev_model1(dev_path):
    # Model Loading:
    model1 = torch.load("model/blstm1.pt")
    model1.eval()

    dev_data = read_data(dev_path)
    encode = Encode1()
    dev_dataset = encode.Encode_Vocab(dev_data)
    dev_dataset = NERDataset1(dev_dataset)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)

    with open("dev1.out", "w") as f:
        for i, (sentence, label) in enumerate(dev_loader):
            # print((sentence, label))
            output = model1(sentence)
            # print(output)
            predicted = output.argmax(dim = -1)
            # print(predicted)
            index = 0
            for j in range(len(sentence[0])):
                # print(sentence[0][j].item())
                index = index + 1
                word = dev_data[i][0][j]
                original_tag = INDEX_TO_TAG[label[0][j].item()]
                predicted_tag = INDEX_TO_TAG[predicted[0][j].item()]
                # f.write("{} {} {} {}\n".format(index, word, original_tag, predicted_tag))
                f.write("{} {} {}\n".format(index, word, predicted_tag))
            if (i != len(dev_loader)-1): f.write("\n")

    print("File dev1.out Created")

def test_model1(test_path):
    # Model Loading:
    model1 = torch.load("model/blstm1.pt")
    model1.eval()

    test_data = read_data_test(test_path)
    encode = Encode1()
    test_dataset = encode.Encode_Vocab(test_data)
    test_dataset = NERDataset1(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with open("test1.out", "w") as f:
        for i, (sentence, label) in enumerate(test_loader):
            output = model1(sentence)
            predicted = output.argmax(dim = -1)
            index = 0
            for j in range(len(sentence[0])):
                index = index + 1
                word = test_data[i][0][j]
                predicted_tag = INDEX_TO_TAG[predicted[0][j].item()]
                f.write("{} {} {}\n".format(index, word, predicted_tag))
            if (i != len(test_loader)-1): f.write("\n")

    print("File test1.out Created")

# ------------------------Train Process 2---------------------------

def train_model_process2(train_loader):
    global WORD_TO_INDEX_GLOVE, TAG_TO_INDEX, GLOVE_VECTOR
    input_size = len(WORD_TO_INDEX_GLOVE)
    # print(input_size)
    output_size = len(TAG_TO_INDEX)
    embedding_dim = 100
    lstm_hidden_dim = 256
    linear_output_dim = 128
    num_layers = 1
    dropout = 0.33
    lr = 0.5
    num_epochs = 10

    weights_matrix = torch.zeros((input_size, embedding_dim))
    for word, index in WORD_TO_INDEX_GLOVE.items():
        word = word.lower()
        if word in GLOVE_VECTOR:
            weights_matrix[index] = GLOVE_VECTOR[word]

    model2 = BLSTM2(input_size, output_size, embedding_dim, lstm_hidden_dim, linear_output_dim, num_layers, dropout, weights_matrix)
    print(model2)
    optimizer = optim.SGD(model2.parameters(), lr=lr, momentum=0.9)
    class_weights = torch.tensor([0.7, 1, 1, 1, 1, 1, 1, 1, 1])
    ignore_index = -1 

    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)

    for epoch in range(num_epochs):
        running_loss = 0
        model2.train()
        for sentence, capitalization, label in train_loader:
            optimizer.zero_grad()
            # print(sentence.shape, capitalization.shape)
            output = model2(sentence, capitalization)
            output = output.permute(0, 2, 1)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch %d \t Training Loss: %.6f' % (epoch+1, running_loss/len(train_loader)))

    print("Model blstm2 Training Completed")
    # Model Saving-
    torch.save(model2, "model/blstm2.pt")

def train_model2(train_data):
    encode = Encode2()
    train_data = encode.Encode_Vocab(train_data)

    batch_size = 64
    train_dataset = NERDataset2(train_data)
    train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            collate_fn=collate_fn2)

    # train_model_process2(train_loader)

def dev_model2(dev_path):
    # Model Loading-
    model2 = torch.load("model/blstm2.pt")
    model2.eval()

    dev_data = read_data(dev_path)
    encode = Encode2()
    dev_dataset = encode.Encode_Vocab(dev_data)
    dev_dataset = NERDataset2(dev_dataset)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)

    with open("dev2.out", "w") as f:
        for i, (sentence, capitalization, label) in enumerate(dev_loader):
            # print((sentence, label))
            output = model2(sentence, capitalization)
            # print(output)
            predicted = output.argmax(dim = -1)
            index = 0
            for j in range(len(sentence[0])):
                # print(sentence[0][j].item())
                index = index + 1
                word = dev_data[i][0][j]
                original_tag = INDEX_TO_TAG[label[0][j].item()]
                predicted_tag = INDEX_TO_TAG[predicted[0][j].item()]
                # f.write("{} {} {} {}\n".format(index, word, original_tag, predicted_tag))
                f.write("{} {} {}\n".format(index, word, predicted_tag))
            if (i != len(dev_loader)-1): f.write("\n")

    print("File dev2.out Created")

def test_model2(test_path):
    # Model Loading:
    model2 = torch.load("model/blstm2.pt")
    model2.eval()

    test_data = read_data_test(test_path)
    encode = Encode2()
    test_dataset = encode.Encode_Vocab(test_data)
    test_dataset = NERDataset2(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with open("test2.out", "w") as f:
        for i, (sentence, capitalization, label) in enumerate(test_loader):
            output = model2(sentence, capitalization)
            predicted = output.argmax(dim = -1)
            index = 0
            for j in range(len(sentence[0])):
                index = index + 1
                word = test_data[i][0][j]
                predicted_tag = INDEX_TO_TAG[predicted[0][j].item()]
                f.write("{} {} {}\n".format(index, word, predicted_tag))
            if (i != len(test_loader)-1): f.write("\n")

    print("File test2.out Created")

if __name__ == "__main__":

    # -----Task 1-----
    train_data = read_data(TRAIN_PATH)
    vocab_create1(train_data)
    train_model1(train_data)
    dev_model1(DEV_PATH)
    test_model1(TEST_PATH)

    
    # -----Task 2-----
    train_data = read_data(TRAIN_PATH)
    vocab_create2()
    train_model2(train_data)
    dev_model2(DEV_PATH)
    test_model2(TEST_PATH)



