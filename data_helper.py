
from pathlib import Path
from torchvision.io import decode_image
from torch.utils.data import Dataset 
import os
import torch
import logging


# create a Dataset class to retrieve the data
class SQLDataset_Informative(Dataset):
    def __init__(self, conn, label_col, img_col='image_path', data_dir=Path(os.path.expanduser('~'), 'CrisisMMD_v2.0','CrisisMMD_v2.0'), 
                 transform=None, target_transform=None, is_train=False, is_test=False, is_val=False):
        '''
        Parameters: 
            conn - a mysql.connector object that will be used to retrieve a cursor 
            label_col - a name of type string that matches the column name in the sql database that labels the image data
            img_col - a name of type string that matches the column name in the sql database that contains the image path
            data_dir - the path that contains the folder containing the image paths in the sql database 
            transform - pytorch image transformations that transform the data
            target_transform - does nothing as of now, so do not specify this
            is_train | is_val | is_test - choose none or one of these; if none chosen, all data is used 
        
        Notes:
            is_train uses 90% of the data
            is_val and is_test each use 5% of the data 
        '''
        assert(not ((is_train and is_test) or (is_train and is_val) or (is_val and is_test)), 'a dataset can only be one of either train, test, or val')

        self.conn = conn
        self.img_col = img_col
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.data_dir = data_dir
        self.is_train = is_train
        self.is_val = is_val
        self.is_test = is_test

        cursor = self.conn.cursor()

        # we need a list of available indices 
        # what idxs are available to use for this database? - depends on the dataset type
        if not (self.is_train or self.is_val or self.is_test): # if no dataset type specified
            query = 'SELECT COUNT(image_id) FROM Images'
            cursor.execute(query)
            count = cursor.fetchone()[0]
            self.possible_sql_idxs = range(count)
        else:
            
            query = 'SELECT COUNT(image_id) FROM Images'
            cursor.execute(query)
            count = cursor.fetchone()[0]
            

            # below, we use +1 because SQL indexing starts at 1
            if is_train: 
                self.possible_sql_idxs = [i+1 for i in range(count) if (((i+1) % 20) < 18)]
            elif is_val:
                self.possible_sql_idxs = [i+1 for i in range(count) if (((i+1) % 20) == 18)]
            else: # must be test
                self.possible_sql_idxs = [i+1 for i in range(count) if (((i+1) % 20) > 18)]
        cursor.close()

    def __len__(self):
        '''
        Returns the number of images in the database 
        '''
        # cursor = self.conn.cursor()

        # if not (self.is_train or self.is_val or self.is_test): # if no dataset type specified 
        #     query = 'SELECT COUNT(image_id) FROM Images'
        # elif self.is_train: 
        #     query = 'SELECT COUNT(image_id) FROM Images WHERE MOD(idx, 20) < 18'
        # elif self.is_val:
        #     query = 'SELECT COUNT(image_id) FROM Images WHERE MOD(idx, 20) = 18'
        # else: # must be test
        #     query = 'SELECT COUNT(image_id) FROM Images WHERE MOD(idx, 20) > 18'

        # cursor.execute(query)
        # count = cursor.fetchone()
        # cursor.close()

        # return count[0]

        return len(self.possible_sql_idxs)
    
    def __getitem__(self, idx):
        '''
        Description: 
            Retrieves a tuple of (torch.tensor, string) where the first object is a 3D tensor of image data and the string is the label
        '''
        # retrieve an image from the sql database

        cursor = self.conn.cursor()

        try:
                query = f'SELECT {self.img_col}, {self.label_col} FROM Images WHERE idx={self.possible_sql_idxs[idx]}' 
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

        return {'image': image, 'label': label}