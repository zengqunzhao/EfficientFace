import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from models import resnet
from models import EfficientFace
import glob
from PIL import Image
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import cv2
import matplotlib.pyplot as plt

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    # create model
    ## EfficientFace
    model_cla = EfficientFace.efficient_face()
    model_cla.fc = nn.Linear(1024, 7)
    model_cla = torch.nn.DataParallel(model_cla).cuda()
    checkpoint = torch.load('./checkpoint/EfficientFace_Trained_on_CAERS.pth.tar')
    pre_trained_dict = checkpoint['state_dict']
    model_cla.load_state_dict(pre_trained_dict)

    # # LDG
    # model_dis = resnet.resnet50()
    # model_dis.fc = nn.Linear(2048, 7)
    # model_dis = torch.nn.DataParallel(model_dis).cuda()
    # checkpoint = torch.load('./checkpoint/LDG_Pretrained_on_CAERS.tar')
    # model_dis.load_state_dict(checkpoint['state_dict'])

    # Data loading code
    data_dir = './test_data'
    image_dir = glob.glob(os.path.join(data_dir, '*', '*.jpg'))

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
    
    transforms_com = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    softmax = nn.Softmax(dim=1)
    
    label = ('Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger')
    
    final_img = np.zeros((750, 500, 3), np.uint8)
    
    model_cla.eval()
    with torch.no_grad():
        for img in image_dir:
            img_t = Image.open(img).convert('RGB')
            img_t = transforms_com(img_t)
            img_t = img_t.unsqueeze(0)
            img_t = img_t.cuda()
            output = model_cla(img_t)
            output = softmax(output)
            output = output.cpu().numpy()[0]
            print(output)
            
            u = list(img.split('/'))
            text = np.round(output, 2)
            plt.figure(figsize=(5, 2.5))
            plt.bar(range(len(text)), text, tick_label=label)
            plt.title('Distribution of the Prediction', color='red')
            plt.ylim(0, 1)
            plt.savefig('./Vis/' + u[-2] + '_' + u[-1])

            u = list(img.split('/'))
            img_pad = cv2.imread('./Vis/' + u[-2] + '_' + u[-1])
            text = str(text)
            img = cv2.imread(img)
            img = cv2.resize(img, (500, 500))
            cv2.putText(img, text, (0, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255))
            final_img[0:250, 0:500] = img_pad  #H,W
            final_img[250:750, 0:500] = img
            cv2.imwrite('./Vis/' + u[-2] + '_final_' + u[-1], final_img)

class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    pass


if __name__ == '__main__':
    main()