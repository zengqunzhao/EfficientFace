import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from models import EfficientFace
import glob
from PIL import Image
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import time

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    
    # create model
    ## EfficientFace
    model = EfficientFace.efficient_face()
    model.fc = nn.Linear(1024, 7)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load('./checkpoint/.......pth.tar')
    pre_trained_dict = checkpoint['state_dict']
    model.load_state_dict(pre_trained_dict)

    # Data loading code
    data_dir = './test_data'
    image_dir = glob.glob(os.path.join(data_dir, '*'))
    print(image_dir)

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    
    transforms_com = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    softmax = nn.Softmax(dim=1)
    
    model.eval()
    with torch.no_grad():
        for img in image_dir:
            # print(img)
            img_t = Image.open(img).convert('RGB')
            img_t = transforms_com(img_t)
            img_t = img_t.unsqueeze(0)
            img_t = img_t.cuda()
            time_s = time.time()
            output = model(img_t)
            print(time.time()-time_s)
            output = softmax(output)
            output = output.cpu().numpy()[0]
            print(output)

if __name__ == '__main__':
    main()
