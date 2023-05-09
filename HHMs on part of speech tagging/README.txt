---------------Instruction Run Code----------------------
Command - python HiddenMarkovModel.py

--------------python version-------------
Python 3.10.4

--------------app Structure--------------
HomeworkAssignmentNo2/
├── code/
│   ├── HiddenMarkovModel.py
│   ├── hmm.json
│   ├── README.txt
│   ├── vocab.txt
│   ├── greedy.out
│   └── viterbi.out
│
└── data/
    ├── dev
    ├── test
    └── train

---------------Output Format---------------
-----Task 1-----
What is the selected threshold for unknown words replacement?  2
What is the total occurrences of the special token ‘< unk >’ after replacement?  20011
What is the total size of your vocabulary?  23183
Vocab.txt       // Vocab.txt file created
-----Task 2-----
How many transition and emission parameters in your HMM?
Transition Parameters 1392
Emission Parameters 30303
hmm.json        // hmm.json file created
-----Task 3-----
What is the accuracy on the dev data?
Greedy Decoding with HMM:  0.9318895372130801
greedy.out      // greedy.out file created
-----Task 4-----
What is the accuracy on the dev data?
Viterbi Decoding with HMM:  0.9476816115247703
viterbi.out      // viterbi.out file created