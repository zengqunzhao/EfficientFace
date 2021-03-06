# EfficientFace

*Zengqun Zhao, Qingshan Liu, Feng Zhou. "[Robust Lightweight Facial Expression Recognition Network with Label Distribution Training](https://ojs.aaai.org/index.php/AAAI/article/view/16465)". AAAI'21*

## Requirements

- Python >= 3.6
- PyTorch >= 1.2
- torchvision >= 0.4.0

## Training

- Step 1: download basic emotions dataset of [RAF-DB](http://www.whdeng.cn/raf/model1.html), and make sure it have the structure like following:

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
The samples' number of CAER-S dataset employed in our work should be: all (69,982 samples), training set (48,995 samples), and test set (20,987 samples). We apologize for the typos in our paper.
