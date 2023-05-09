"""
@author: advait naik
"""

from collections import defaultdict
import json
import numpy as np

TRAIN_PATH = '../data/train'
DEV_PATH = '../data/dev'
TEST_PATH = '../data/test'
VOCAB = dict()
_TAG = set()
THERSHOLD = 2
TRANSITION = defaultdict(float)
EMISSION = defaultdict(float)

# ----------------------Helper Function-------------------------

def data_preparation(input_data: list[str]) -> list:
    """
    read data line and convert line in format [(word, tag)] 
    :param input_data:
    :return:
    """
    line = []
    data = []
    for sentence in input_data:
        if not sentence.isspace():
            line.append(
                (sentence.split('\t')[1], sentence.split('\t')[2].strip()))
        if sentence.isspace():
            data.append(line)
            line = []
    return data


def data_preparation_test(input_data: list[str]) -> list:
    """
    read data line and convert line in format [(word)] 
    :param input_data:
    :return:
    """
    line = []
    data = []
    for sentence in input_data:
        if not sentence.isspace():
            line.append(sentence.split('\t')[1].strip())
        if sentence.isspace():
            data.append(line)
            line = []
    return data

def readDataFile(path: str) -> str:
    """
    read raw data
    :param path: raw data file path
    :return:
    """
    with open(path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    # print(data)
    return data

# ----------------------Helper Function-------------------------

# ---------------------Task 1--------------------

def readTrainDataFile(train_data: list[str]) -> dict:
    """
    helper function vocabulary creation
    :param train_data: raw train data file
    :return:
    """
    # words = []
    vocab_data = defaultdict(int)
    for sentence in train_data:
        if not sentence.isspace():
            # word.append(sentence.split('\t')[1])
            vocab_data[sentence.split('\t')[1]] += 1
    # print(VocabularyCreation)
    return vocab_data


def createVocab(vocab_data: defaultdict, THERSHOLD) -> dict:
    """
    helper function vocabulary creation
    :param vocab_data: dictionary of words 
    :return:
    """
    Vocab = {}
    Vocab['< unk >'] = 0

    for key, value in vocab_data.items():
        if (value > THERSHOLD-1):
            Vocab[key] = value
        else:
            Vocab['< unk >'] += value

    Vocab = sorted(Vocab.items(), key=lambda x: x[1], reverse=True)
    Vocab = dict(Vocab)
    # print(Vocab)
    # print(Vocab['< unk >'])
    return Vocab


def outputVocabFile(Vocab: dict) -> None:
    """
    helper function vocabulary creation
    :param Vocab: dictionary of words with thershold
    :return:
    """
    vocab = []
    count = 1
    unknown = '< unk >' + '\t' + '0' + '\t' + str(Vocab['< unk >'])
    vocab.append(unknown)

    print('What is the total occurrences of the special token ‘< unk >’ after replacement? ',
          Vocab['< unk >'])
    del Vocab['< unk >']

    for key, value in Vocab.items():
        line = key + '\t' + str(count) + '\t' + str(value)
        vocab.append(line)
        count += 1

    # print(vocab)
    print('What is the total size of your vocabulary? ', len(vocab))

    with open('vocab', 'wt') as file:
        file.write('\n'.join(vocab))
    print('Vocab.txt')


def VocabularyCreation(train_data: list[str]) -> None:
    """
    create vocab.txt file 
    :param train_data: raw train data file
    :return:
    """
    print('-----Task 1-----')
    global VOCAB, THERSHOLD
    print('What is the selected threshold for unknown words replacement? ', THERSHOLD)
    vocab_data = readTrainDataFile(train_data)
    VOCAB = createVocab(vocab_data, THERSHOLD)
    outputVocabFile(VOCAB)

# ---------------------Task 1--------------------

# ---------------------Task 2--------------------

def outputHMMFile() -> None:
    """
    create hmm.json file 
    :param: 
    :return:
    """
    transition = dict()
    emission = dict()

    for key, value in TRANSITION.items():
        transition[str(key)] = value

    for key, value in EMISSION.items():
        emission[str(key)] = value

    hmm = {'transition': transition, 'emission': emission}
    json_hmm = json.dumps(hmm)

    with open('hmm.json', 'w') as file:
        file.write(json_hmm)
    print('hmm.json')

def ModelLearning(train_data: list[str]) -> None:
    """
    create TRANSITION EMISSION 
    :param: 
    :return:
    """
    # print(train_data)
    global TRANSITION, EMISSION, _TAG

    transition_data = defaultdict(float)
    emission_data = defaultdict(float)
    tag_data = defaultdict(float)
    tag_data['< start >'] = 0
    # tag_data['< end >'] = 0

    data = data_preparation(train_data)
    for line in data:
        line = [('', '< start >')] + line
        tag_data['< start >'] += 1
        for index in range(1, len(line)):
            word, tag = line[index]
            _, previous_tag = line[index-1]

            if word not in VOCAB:
                word = '< unk >'

            tag_data[tag] += 1
            transition_key = (previous_tag, tag)
            transition_data[transition_key] += 1

            emission_key = (tag, word)
            emission_data[emission_key] += 1

        # transition_dictionary[(tag, '< end >')] += 1
        # tag_dictionary['< end >'] += 1

    # print(len(emission_data))
    # print(len(transition_data))
    # print(data)

    for transition_key in transition_data.keys():
        _tag = transition_key[0]
        TRANSITION[transition_key] = transition_data[transition_key] / tag_data[_tag]

    for emission_key in emission_data.keys():
        _tag = emission_key[0]
        EMISSION[emission_key] = emission_data[emission_key] / tag_data[_tag]

    for key in tag_data.keys():
        if key != '< start >':
            _TAG.add(key)

    # print(len(_TAG))
    # print(TRANSITION)
    # print(EMISSION)
    print('-----Task 2-----')
    print("How many transition and emission parameters in your HMM?")
    print('Transition Parameters', len(TRANSITION))
    print('Emission Parameters', len(EMISSION))
    outputHMMFile()
# ---------------------Task 2--------------------


class HiddenMarkovModel:
    def __init__(self, dev_data, test_data) -> None:
        self.dev_data = dev_data
        self.test_data = test_data

# ---------------------Output File--------------------
    def predictGreedyDecodingHMM(self, sentence: list) -> list:
        """
        predict tag
        :param sentence:
        :return: 
        """
        predicted_line = []
        word = sentence[0]
        _word = word
        if word not in VOCAB:
            word = '< unk >'
        _predicted_tag = max([_tag_ for _tag_ in _TAG], key=lambda _tag_: TRANSITION[('< start >', _tag_)] * EMISSION[(_tag_, word)])
        predicted_line.append((_word, _predicted_tag))

        for index in range(1, len(sentence)):
            word = sentence[index]
            _word = word
            if word not in VOCAB:
                word = '< unk >'
            predicted_tag = max([__tag for __tag in _TAG], key=lambda __tag: TRANSITION[(_predicted_tag, __tag)] * EMISSION[(__tag, word)])
            
            predicted_line.append((_word, predicted_tag))
            _predicted_tag = predicted_tag

        return predicted_line
    
    def predictViterbiDecodingHMM(self, sentence: list) -> list:
        """
        predict tag
        :param sentence:
        :return: 
        """
        predicted_line = []
        _tags = list(_TAG)
        _tag_num = len(_tags)

        line_num = len(sentence)
        ViterbiMatrix = np.zeros((line_num, _tag_num), dtype=np.float64)
        backtrack = np.zeros((line_num, _tag_num), dtype=int)

        word = sentence[0]
        _word = word
        predicted_line.append([_word])
        if word not in VOCAB:
            word = '< unk >'
        for tag in range(_tag_num):
            ViterbiMatrix[0][tag] = TRANSITION[('< start >', _tags[tag])] * EMISSION[(_tags[tag], word)]
            backtrack[0][tag] = tag

        for word_num in range(1, line_num):
            word = sentence[word_num]
            _word = word
            predicted_line.append([_word])
            if word not in VOCAB:
                word = '< unk >'
            for tag in range(_tag_num):
                ViterbiMatrix[word_num][tag], backtrack[word_num][tag] = max((ViterbiMatrix[word_num-1][prev_tag] * TRANSITION[(_tags[prev_tag], _tags[tag])] * EMISSION[(_tags[tag], word)], prev_tag) for prev_tag in range(_tag_num))
                    
        _tag_index_predicted = [np.argmax(ViterbiMatrix[line_num-1])]
        for word_num in range(line_num-2, -1, -1):
            _tag_index_predicted.insert(0, backtrack[word_num+1][_tag_index_predicted[0]])

        for index in range(len(predicted_line)):
            predicted_line[index].append(_tags[_tag_index_predicted[index]])
        return predicted_line

    def outputPredictionFile(self, algorithm) -> None:
        """
        viterbi.out greedy.out create
        :param algorithm:
        :return: 
        """
        vocab = []
        _test_data = data_preparation_test(self.test_data)

        if algorithm == 'greedy':
            for sentence in _test_data:
                predictedline = self.predictGreedyDecodingHMM(sentence)
                for word_index in range(len(predictedline)):
                    _word, _tag = predictedline[word_index]
                    insertline = str(word_index+1) + '\t' + _word + '\t' + _tag + '\n'
                    vocab.append(insertline)
                vocab.append('\n')

            with open('greedy.out', 'wt') as file:
                file.write(''.join(vocab))

            print('greedy.out')

        if algorithm == 'viterbi':
            for sentence in _test_data:
                predictedline = self.predictViterbiDecodingHMM(sentence)
                for word_index in range(len(predictedline)):
                    _word, _tag = predictedline[word_index]
                    insertline = str(word_index+1) + '\t' + _word + '\t' + _tag + '\n'
                    vocab.append(insertline)
                vocab.append('\n')

            with open('viterbi.out', 'wt') as file:
                file.write(''.join(vocab))

            print('viterbi.out')
# ---------------------Output File--------------------

# ---------------------Task 3--------------------
    def GreedyDecodingHMM(self):
        """
        dev data accuracy and call greedy.out create function
        :param:
        :return: 
        """
        print('-----Task 3-----')
        data = data_preparation(self.dev_data)
        # print(len(data))
        _total = 0
        _matched = 0

        for line in data:
            word, _tag = line[0]
            if word not in VOCAB:
                word = '< unk >'
            _predicted_tag = max([_tag_ for _tag_ in _TAG], key=lambda _tag_: TRANSITION[('< start >', _tag_)] * EMISSION[(_tag_, word)])
            if (_tag == _predicted_tag):
                _matched += 1
            _total += 1

            # _predicted_tag = max([__tag for __tag in _TAG], key=lambda __tag: TRANSITION[('< start >', __tag)])

            for index in range(len(line)):
                word, _tag = line[index]
                if word not in VOCAB:
                    word = '< unk >'
                predicted_tag = max([__tag for __tag in _TAG], key=lambda __tag: TRANSITION[(_predicted_tag, __tag)] * EMISSION[(__tag, word)])
                _predicted_tag = predicted_tag
                if (_tag == predicted_tag):
                    _matched += 1
                _total += 1
        print('What is the accuracy on the dev data?')
        print('Greedy Decoding with HMM: ', _matched/_total)
        self.outputPredictionFile('greedy')

# ---------------------Task 3--------------------


# ---------------------Task 4--------------------
    def ViterbiDecodingHMM(self):
        """
        dev data accuracy and call viterbi.out create function
        :param:
        :return: 
        """
        print('-----Task 4-----')
        data = data_preparation(self.dev_data)
        _total = 0
        _matched = 0
        _tags = list(_TAG)
        _tag_num = len(_tags)

        for sentence in data: 
            # print(sentence)
            _tag_actual = [sentence[0][1]]
            # print(states)
            
            line_num = len(sentence)
            ViterbiMatrix = np.zeros((line_num, _tag_num), dtype=np.float64)
            backtrack = np.zeros((line_num, _tag_num), dtype=int)

            for tag in range(_tag_num):
                word = sentence[0][0]
                if word not in VOCAB:
                    word = '< unk >'
                ViterbiMatrix[0][tag] = TRANSITION[('< start >', _tags[tag])] * EMISSION[(_tags[tag], word)]
                backtrack[0][tag] = tag

            # print(ViterbiMatrix[0])

            for word_num in range(1, line_num):
                # _tag.append(sentence[word_num][1])
                # word = sentence[word_num][0]
                word, tag = sentence[word_num]
                _tag_actual.append(tag)

                # print(word)
                if word not in VOCAB:
                    word = '< unk >'
                for tag in range(_tag_num):
                    ViterbiMatrix[word_num][tag], backtrack[word_num][tag] = max((ViterbiMatrix[word_num-1][prev_tag] * TRANSITION[(_tags[prev_tag], _tags[tag])] * EMISSION[(_tags[tag], word)], prev_tag) for prev_tag in range(_tag_num))
                    
            _tag_index_predicted = [np.argmax(ViterbiMatrix[line_num-1])]
            # print(temp_result)
            # print(temp_result)
            for word_num in range(line_num-2, -1, -1):
                _tag_index_predicted.insert(0, backtrack[word_num+1][_tag_index_predicted[0]])

            # print(backtrack)
            tag_index = 0
            for index in _tag_index_predicted:
                _total += 1
                if (_tags[index] == _tag_actual[tag_index]):
                    _matched += 1
                tag_index += 1
                # results.append(states[index])

            # return _tag,results
        print('What is the accuracy on the dev data?')
        print('Viterbi Decoding with HMM: ', _matched/_total)
        self.outputPredictionFile('viterbi')

# ---------------------Task 4--------------------

if __name__ == "__main__":

    train_data = readDataFile(TRAIN_PATH)

    # -----Task 1-----
    VocabularyCreation(train_data)

    # -----Task 2-----
    ModelLearning(train_data)

    dev_data = readDataFile(DEV_PATH)
    test_data = readDataFile(TEST_PATH)
    hmm = HiddenMarkovModel(dev_data, test_data)

    # -----Task 3-----
    hmm.GreedyDecodingHMM()

    # -----Task 4-----
    hmm.ViterbiDecodingHMM()

