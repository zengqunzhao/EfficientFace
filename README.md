# EfficientFace

*Zengqun Zhao, Qingshan Liu, and Feng Zhou, "[Robust Lightweight Facial Expression Recognition Network with Label Distribution Training](https://zengqunzhao.github.io/doc/pdfs/AAAI2021.pdf)", AAAI'21*


## Requirements
- Python $\geq$3.6
- PyTorch $\geq$1.2
- torchvision $\geq$0.4.0
- numpy
- matplotlib
- datetime
- shutil
- time
- argparse
- os

## Training

- Step 1: download basic emotions dataset of [RAF-DB](http://www.whdeng.cn/raf/model1.html), and make sure it have the structure like following:
 
```
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

- Step 2: download pre-trained model from
   [Google Drive](https://drive.google.com/file/d/1sRS8Vc96uWx_1BSi-y9uhc_dY7mSED6f/view?usp=sharing),
    and put it into ***./checkpoint***.
    
- Step 3: change the ***--data*** in *run.sh* to your path 

- Step 4: run ```sh run.sh ```


## Citation

If you find this repository helpful, use this code or adopt ideas from the paper for your research, please cite:

```
@inproceedings{zhao2021robust,
  title={Robust Lightweight Facial Expression Recognition Network with Label Distribution Training},
  author={Zhao, Zengqun and Liu, Qingshan and Zhou, Feng},
  booktitle={AAAI},
  pages={},
  year={2021}
}
```
