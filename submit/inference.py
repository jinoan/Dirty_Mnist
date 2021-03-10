import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import os
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

import timm
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import imgaug
import random


class config:
    seed = 42
    device = "cuda"    
        
    test_batch_size = 32
    num_workers = 8
    tta = True
    tta_count = 32
    
    IMG_SIZE = 380

    TEST_DATA_PATH = './data/'
    ADAM_MODEL_PATH = './input/adam_model/'
    SAM_MODEL_PATH = './input/sam_model/'
    SAMPLE_SUBMISSION_FILE = './input/sample_submission.csv'
    SUBMISSION_FILE = './output/submission.csv'


# device 설정
device = torch.device(config.device if torch.cuda.is_available() else "cpu")

# seed 설정
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    imgaug.random.seed(seed)

seed_everything(config.seed)


def self_cut_mix(img):
    transformed_img = img.copy()
    l = random.randrange(img.shape[0] // 8, img.shape[0] // 2 + 1)
    
    # cut
    if random.choice([True, False]):
        i1 = random.randrange(l, img.shape[0] - l + 1)
        j1 = random.randrange(0, img.shape[0] - l + 1)
        i2 = random.randrange(0, i1 - l + 1)
        j2 = random.randrange(0, img.shape[0] - l + 1)
    else:
        i1 = random.randrange(0, img.shape[0] - l + 1)
        j1 = random.randrange(l, img.shape[0] - l + 1)
        i2 = random.randrange(0, img.shape[0] - l + 1)
        j2 = random.randrange(0, j1 - l + 1)
        
    p1 = img[i1:i1+l, j1:j1+l].copy()
    p2 = img[i2:i2+l, j2:j2+l].copy()
    
    # rotate piece
    random_rotate_90 = A.RandomRotate90(p=1)
    p1 = random_rotate_90(image=p1)["image"]
    p2 = random_rotate_90(image=p2)["image"]
    
    # mix
    if random.choice([True, False]):
        transformed_img[i1:i1+l, j1:j1+l] = p2
        transformed_img[i2:i2+l, j2:j2+l] = p1
    else:
        transformed_img[i1:i1+l, j1:j1+l] = p1
        transformed_img[i2:i2+l, j2:j2+l] = p2

    return transformed_img

class SelfCutMix(ImageOnlyTransform):
    def __init__(
        self,
        always_apply=False,
        p=1
    ):
        super(SelfCutMix, self).__init__(always_apply, p)
    
    def apply(self, img, **params):
        return self_cut_mix(img)


# numpy를 tensor로 변환하는 ToTensor 정의
class ToTensor(object):
    """numpy array를 tensor(torch)로 변환합니다."""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.FloatTensor(image),
                'label': torch.FloatTensor(label)}


# to_tensor 선언
to_tensor = T.Compose([
                        ToTensor()
                    ])


class DatasetMNIST(torch.utils.data.Dataset):
    def __init__(self,
                 dir_path,
                 meta_df,
                 transforms=to_tensor,  # 미리 선언한 to_tensor를 transforms로 받음
                 augmentations=None):
        
        self.dir_path = dir_path  # 데이터의 이미지가 저장된 디렉터리 경로
        self.meta_df = meta_df  # 데이터의 인덱스와 정답지가 들어있는 DataFrame

        self.transforms = transforms  # Transform
        self.augmentations = augmentations  # Augmentation
        
    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, index):
        # 폴더 경로 + 이미지 이름 + .png => 파일의 경로
        # 참고) "12".zfill(5) => 000012
        #       "146".zfill(5) => 000145
        # cv2.IMREAD_GRAYSCALE : png파일을 채널이 1개인 GRAYSCALE로 읽음
        image = cv2.imread(self.dir_path +\
                           str(self.meta_df.iloc[index,0]).zfill(5) + '.png',
                           cv2.IMREAD_GRAYSCALE)
        
        image = cv2.resize(image, dsize=(config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        # augmentation 적용
        if self.augmentations:
            augmentations = A.Compose([
                A.RandomRotate90(p=1),
                A.GridDistortion(p=0.8),
                SelfCutMix(p=1),
                A.CoarseDropout(p=0.8, max_holes=4, max_height=32, max_width=32, min_holes=1, min_height=16, min_width=16, fill_value=0),
                A.GaussNoise(p=0.75),
            ])
            image = augmentations(image=image)['image']
        
        # 0 ~ 255의 값을 갖고 크기가 (256,256)인 numpy array를
        # 0 ~ 1 사이의 실수를 갖고 크기가 (256,256,1)인 numpy array로 변환
        image = (image/255).astype('float')[..., np.newaxis]

        # 정답 numpy array생성(존재하면 1 없으면 0)
        label = self.meta_df.iloc[index, 1:].values.astype('float')
        sample = {'image': image, 'label': label}

        # transform 적용
        # numpy to tensor
        if self.transforms:
            sample = self.transforms(sample)

        # sample 반환
        return sample


#test Dataset 정의
sample_submission = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)
test_dataset = DatasetMNIST(config.TEST_DATA_PATH, sample_submission)
test_data_loader = DataLoader(
    test_dataset,
    batch_size = config.test_batch_size,
    shuffle = False,
    num_workers = config.num_workers,
    drop_last = False
)


####################
##### Adam 모델 #####
####################

class MultiLabelModel(nn.Module):
    def __init__(self):
        super(MultiLabelModel, self).__init__()
        self.conv2d = nn.Conv2d(1, 3, 3, stride=1, padding=1)
        self.efn_b3 = timm.create_model('efficientnet_b3', pretrained=True, num_classes=1024)
        self.fc = nn.ModuleList([nn.Linear(1024, 1) for i in range(26)])

        nn.init.xavier_uniform_(self.conv2d.weight)
        for i in range(26):
            nn.init.xavier_uniform_(self.fc[i].weight)

    def forward(self, x):
        x = F.relu(self.conv2d(x))
        x = F.relu(self.efn_b3(x))
        xs = []
        for i in range(26):
            xs.append(self.fc[i](x))
        x = torch.cat(xs, dim=1)
        x = torch.sigmoid(x)
        return x


# 모델 선언
model = MultiLabelModel()

model_files = os.listdir(config.ADAM_MODEL_PATH)
best_models = [torch.load(config.ADAM_MODEL_PATH + model_file) for model_file in model_files]

predictions_list = []
# 배치 단위로 추론
prediction_df = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)

# 5개의 fold마다 가장 좋은 모델을 이용하여 예측
for model_index, model in enumerate(best_models):
    print(f'[model: {model_files[model_index]}]')
    if config.tta:
        count = config.tta_count
    else:
        count = 1
    for c in range(count):
        # 0으로 채워진 array 생성
        prediction_array = np.zeros([prediction_df.shape[0],
                                    prediction_df.shape[1] -1])
        with tqdm(test_data_loader,
                total=test_data_loader.__len__(),
                unit="batch") as test_bar:
            for idx, sample in enumerate(test_bar):
                with torch.no_grad():
                    # 추론
                    model.eval()
                    images = sample['image']
                    images = images.to(device)
                    probs = model(images)
                    probs = probs.cpu().detach().numpy()

                    # 예측 결과를 
                    # prediction_array에 입력
                    batch_index = config.test_batch_size * idx
                    prediction_array[batch_index: batch_index + images.shape[0],:]\
                                = probs
                            
        # 채널을 하나 추가하여 list에 append
        predictions_list.append(prediction_array[...,np.newaxis])


####################
##### SAM  모델 #####
####################

class MultiLabelModel(nn.Module):
    def __init__(self):
        super(MultiLabelModel, self).__init__()
        self.conv2d = nn.Conv2d(1, 3, 3, stride=1, padding=1)
        self.efn_b3 = timm.create_model('efficientnet_b3', pretrained=True, num_classes=1024)
        self.fc = nn.ModuleList([nn.Linear(1024, 1) for i in range(26)])

        nn.init.xavier_uniform_(self.conv2d.weight)
        for i in range(26):
            nn.init.xavier_uniform_(self.fc[i].weight)

    def forward(self, x):
        x = F.gelu(self.conv2d(x))
        x = F.gelu(self.efn_b3(x))
        xs = []
        for i in range(26):
            xs.append(self.fc[i](x))
        x = torch.cat(xs, dim=1)
        x = torch.sigmoid(x)
        return x

# 모델 선언
model = MultiLabelModel()

model_files = os.listdir(config.SAM_MODEL_PATH)
best_models = [torch.load(config.SAM_MODEL_PATH + model_file) for model_file in model_files]

# 5개의 fold마다 가장 좋은 모델을 이용하여 예측
for model_index, model in enumerate(best_models):
    print(f'[model: {model_files[model_index]}]')
    if config.tta:
        count = config.tta_count
    else:
        count = 1
    for c in range(count):
        # 0으로 채워진 array 생성
        prediction_array = np.zeros([prediction_df.shape[0],
                                    prediction_df.shape[1] -1])
        with tqdm(test_data_loader,
                total=test_data_loader.__len__(),
                unit="batch") as test_bar:
            for idx, sample in enumerate(test_bar):
                with torch.no_grad():
                    # 추론
                    model.eval()
                    images = sample['image']
                    images = images.to(device)
                    probs = model(images)
                    probs = probs.cpu().detach().numpy()

                    # 예측 결과를 
                    # prediction_array에 입력
                    batch_index = config.test_batch_size * idx
                    prediction_array[batch_index: batch_index + images.shape[0],:]\
                                = probs
                            
        # 채널을 하나 추가하여 list에 append
        predictions_list.append(prediction_array[...,np.newaxis])

# axis = 2를 기준으로 평균
predictions_array = np.concatenate(predictions_list, axis = 2)
predictions_mean = predictions_array.mean(axis = 2)

# 평균 값이 0.5보다 클 경우 1 작으면 0
predictions_mean = (predictions_mean > 0.5) * 1

sample_submission = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)
sample_submission.iloc[:,1:] = predictions_mean
sample_submission.to_csv(config.SUBMISSION_FILE, index = False)