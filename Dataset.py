import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class HistoDataset(Dataset):
    TRAIN_SET = 9
    TEST_SET = 27
    def __init__(self, dataframe, img_dir, flag=TRAIN_SET):
        '''

        :param dataframe: Dataframe holding image id and labels
        :type dataframe: Pandas dataframe
        :param img_dir: Directory containing images
        :type img_dir: String
        :param flag: Flag to set whether the dataset is training or test set
        :type flag: Int
        '''
        self.df = dataframe
        self.img_dir = img_dir
        self.flag = flag
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        img_id = self.df.iloc[item]['id']
        label = float(self.df.iloc[item]['label'])

        img_path = self.img_dir + img_id + '.tif'
        img = Image.open(img_path)
        t_img = self.transforms(img)
        if self.flag == HistoDataset.TEST_SET:
            return t_img
        return t_img, label
