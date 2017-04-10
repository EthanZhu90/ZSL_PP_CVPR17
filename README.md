# ZSLPP
Mohamed Elhoseiny*, Yizhe Zhu*, Han Zhang, Ahmed Elgammal, Link the head to the "peak'': Zero Shot Learning from Noisy Text descriptions at Part Precision, CVPR, 2017


This code is implemente by  Yizhe Zhu and Mohamed Elhoseiny. 

Data:
You can download the data [CUB2011](https://drive.google.com/open?id=0B_8vkk7CF-pwejFFcEp2R1FfRFU) and [NABird](https://drive.google.com/open?id=0B_8vkk7CF-pwOGhpQXFUUXZlQjg). 

Trianed Models:

https://drive.google.com/open?id=0B_8vkk7CF-pwMU5QQUlUOTZFblU  reproduce the results in the paper.  

Raw wikipedia article data and detailed merging information of NABird can be obtained [here](https://drive.google.com/open?id=0B_8vkk7CF-pwckxLQTVkcDBadGc).

Testing, reproducing the results in the paper
---------------------------------------------

ZSL_Test(Dataset = 'CUBird' or 'NABird', splitmode = 'East' or 'Hard', ImgFtSource = 'DET' or 'ATN')

   splitmode = Easy or Hard splits defined in Section 4.1 in the paper


CUNBirds East split in Table1 (ATN means usin groun truth part bozes)
--------------------------------------------------------------------------------
>> ZSL_Test('CUBird', 'Easy', 'ATN')
Dataset: CUB2011   Easy  ATN
Model: trained_models/CUBird_Easy_ATN.mat
Load Testing set
test_acc = 43.5049%  


 >> ZSL_Test('CUBird', 'Easy', 'DET'), :DET means using the detected parts instead of GT parts. 
----------------------------------------------------------------------
Dataset: CUB2011   Easy  DET
Model: trained_models/CUBird_Easy_DET.mat
Load Testing set
test_acc = 37.5725% 

    NABirds Easy split in Table3 :DET means using the detected parts instead of GT parts. 
--------------------------------------------------------------------------------
>> ZSL_Test('NABird', 'Easy')
Dataset: NABird   Easy  DET
Model: trained_models/NABird_Easy_DET.mat
Load Testing set
test_acc = 30.5937% 

NABirds Hard split in Table3
--------------------------------------------------
>> ZSL_Test('NABird', 'Hard')
Dataset: NABird   Hard  DET
Model: trained_models/NABird_Hard_DET.mat
Load Testing set
test_acc = 8.1349% 




Training
---------
>>ZSL_Train(Dateset, Splitmode, ImgFtSource, lambda1, lambda2, GPU_mode)
is the command  to train the model using a particular setting. 
% For example ZSL_Train('CUBird', 'Easy', 'DET', 100000, 10000, true), trains on the CUBirds dataset on the Easy split and using the detected part boxes. 
, lambda1=100000, and lambda2=10000, and GPU_mode=true (using GPU mode for training). If false, the training is done on CPU.





