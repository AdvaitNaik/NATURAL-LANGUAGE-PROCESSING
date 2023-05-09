---------------Instruction Run Code----------------------
Extract the glove.6B.100d.gz -> glove.6B.100d.txt

TO generate the prediction file -> dev1.out, dev2.out, test1.out, test2.out 
Command- python NameEntityRecognition.py

---------------Output Format---------------

Output- File dev1.out Created
	File test1.out Created
	File dev2.out Created
	File test2.out Created

--------------python version-------------
Python 3.10.4

--------------app Structure--------------
HW4-Advait Hemant-Naik/
├── code/
│   ├── NameEntityRecognition.py
│   ├── glove.6B.100d.txt
│   ├── README.txt
│   ├── conll03eval.txt
│   ├── dev1.out
│   ├── dev2.out
│   ├── test1.out
│   ├── test2.out
│   ├── Report.pdf
│   └── model/
│  	├── blstm1.pt
│   	└── blstm2.pt
│
└── data/
    ├── dev
    ├── test
    └── train

---------------Train Model---------------

To Train the model
Model1 - Uncomment the line 382 -> train_model_process1(train_loader, dev_loader)
Model2 - Uncomment the line 489 -> train_model_process2(train_loader)