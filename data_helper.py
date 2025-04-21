
from pathlib import Path
from torchvision.io import decode_image
from torch.utils.data import Dataset 
import os
import torch
import logging

# logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(messages)s')
# log = logging.getLogger(__name__)

# create a Dataset class to retrieve the data
class SQLDataset_Informative(Dataset):
    def __init__(self, conn, label_col, img_col='image_path', data_dir=Path(os.path.expanduser('~'), 'CrisisMMD_v2.0','CrisisMMD_v2.0'), 
                 transform=None, target_transform=None):
        '''
        Parameters: 
            conn - a mysql.connector object that will be used to retrieve a cursor 
            label_col - a name of type string that matches the column name in the sql database that labels the image data
            img_col - a name of type string that matches the column name in the sql database that contains the image path
            data_dir - the path that contains the folder containing the image paths in the sql database 
            transform - pytorch image transformations that transform the data
            target_transform - does nothing as of now, so do not specify this
        '''
        self.conn = conn
        self.img_col = img_col
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.data_dir = data_dir

    def __len__(self):
        '''
        Returns the number of images in the database 
        '''
        cursor = self.conn.cursor()
        query = 'SELECT COUNT(image_id) FROM Images'
        cursor.execute(query)
        count = cursor.fetchone()
        cursor.close()

        # log.debug(f'Dataset length: {count}')

        return count[0]
    
    def __getitem__(self, idx):
        '''
        Description: 
            Retrieves a tuple of (torch.tensor, string) where the first object is a 3D tensor of image data and the string is the label
        '''
        # retrieve an image from the sql database

        # log.debug(f'Fetching item at index {idx}')

        cursor = self.conn.cursor()
        try:
            query = f'SELECT {self.img_col}, {self.label_col} FROM Images WHERE idx={idx+1}' # we must add one because python starts at 0 idx but sql starts at 1
            cursor.execute(query)
            
            # read in image
            img_path, label = cursor.fetchone()
            img_path = Path(self.data_dir, img_path)
            image = decode_image(img_path, mode='RGB') # returns (Tensor[image_channels, image_height, image_width])

            
            if label == 'informative':
                label = torch.tensor(1)
            else:
                label = torch.tensor(0)
            # print(f'image shape before transform: {image.shape}')
            # apply transforms on image 
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
        finally:
            cursor.close()

        # print(f'image shape after transform: {image.shape}')
        
        # log.debug(f'image and label from __getitem__: {image, label}')
        # log.debug(f'image and label shapes from __getitem__: {image.shape}, {label.shape}')

        return image, label