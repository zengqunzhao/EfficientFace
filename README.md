# EfficientFace

*Zengqun Zhao, Qingshan Liu, Feng Zhou. "[Robust Lightweight Facial Expression Recognition Network with Label Distribution Training](https://drive.google.com/file/d/1yDpyQ1emZ8IObPNZt76ljeW98GP-Dw22/view?usp=sharing)". AAAI'21*

## Requirements

- Python >= 3.6
- PyTorch >= 1.2
- torchvision >= 0.4.0

## Training

- Step 1: download basic emotions dataset of [RAF-DB](http://www.whdeng.cn/raf/model1.html), and make sure it has the structure like the following:

```txt
./RAF-DB/
         train/
               0/
                 train_09748.jpg
                 ...
                 train_12271.jpg
               1/
               ...
               6/
         test/
              0/
              ...
              6/

[Note] 0: Neutral; 1: Happiness; 2: Sadness; 3: Surprise; 4: Fear; 5: Disgust; 6: Anger
```

- Step 2: download pre-trained model from [Google Drive](https://drive.google.com/file/d/1sRS8Vc96uWx_1BSi-y9uhc_dY7mSED6f/view?usp=sharing), and put it into ***./checkpoint***.
- Step 3: change the ***--data*** in *run.sh* to your path
- Step 4: run ``` sh run.sh ```


## Pre-trained Models
- Sept. 16, 2021 Update\
We provide the pre-trained ResNet-18 and ResNet-50 on MS-Celeb-1M (classes number is 12666) for your research.  
The [Google Driver](https://drive.google.com/file/d/1dPCWFaa9xrs6nTVkweTJWyx9qGGdi3qe/view?usp=sharing) for ResNet-18 model. The [Google Driver](https://drive.google.com/file/d/1XweLvXPZPH53xj5Pklj5E3V9-LG9-pLD/view?usp=sharing) for ResNet-50 model.  
The pre-trained ResNet-50 model can be also used for LDG.  
- Nov. 6, 2021 Update\
The fine-tuned LDG models on CAER-S, AffectNet-7, and AffectNet-8 can be downloaded [here](https://drive.google.com/file/d/1tu4996A74PPyZYeUmS-d_9728dvlYDQw/view?usp=sharing), [here](https://drive.google.com/file/d/1FQ1nizEmQ_FxGbk7zzOa4Toe4lGPkAZO/view?usp=sharing), and [here](https://drive.google.com/file/d/16b-Y52Z89FMRysi-gjKNS9z6-rdcijU0/view?usp=sharing), respectively.
- Nov. 12, 2021 Update\
The trained EfficientFace model on RAF-DB, CAER-S, AffectNet-7, and AffectNet-8 can be downloaded [here](https://drive.google.com/file/d/1W_3JT2_c_2R18kPTUUfX5QvQM2km7APC/view?usp=sharing), [here](https://drive.google.com/file/d/1mhdhQUU-ROJNM9kKK_043doVTT2wI-Ua/view?usp=sharing), [here](https://drive.google.com/file/d/1nwerwDyDqC2ia1Eqa-S6lb2SipwL-hTs/view?usp=sharing), and [here](https://drive.google.com/file/d/16nay3FwOLjwNVFG4sKc2TSFwC2TUPNZ8/view?usp=sharing), respectively.
As demonstrated in the paper, the testing accuracy is 88.36\%, 85.87\%, 63.70\%, and 59.89\%, respectively.


## Citation

```txt
@inproceedings{zhao2021robust,
  title={Robust Lightweight Facial Expression Recognition Network with Label Distribution Training},
  author={Zhao, Zengqun and Liu, Qingshan and Zhou, Feng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={4},
  pages={3510--3519},
  year={2021}
}
```


## Note
The samples' number of the CAER-S dataset employed in our work should be: all (69,982 samples), training set (48,995 samples), and test set (20,987 samples). We apologize for the typos in our paper.
