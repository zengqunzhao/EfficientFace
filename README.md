# MA-Net
This is an official PyTorch implementation of MA-Net.

## Requirements
- Python 3.6
- PyTorch 1.2
- torchvision 0.4.0
- numpy
- matplotlib
- datetime
- shutil
- time
- argparse
- os

## Usage

### Train
- Step 1: prepare RAF-DB dataset.

- Step 2: download pre-trained model from
   [Google Driver](https://drive.google.com/file/d/1CYqrarqSSxwt6STIYB_Z4qNx8TGrOYSO/view?usp=sharing),
    and put it into *./checkpoint*.
    
- Step 3: change the ***project_path*** and ***data_path*** in main.py.

- Step 4: run ```python main.py ```
