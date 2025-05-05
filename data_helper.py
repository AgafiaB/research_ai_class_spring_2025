from pathlib import Path
from torchvision.io import decode_image
from torch.utils.data import Dataset 
import os
import torch
import logging
import numpy as np
import cv2


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


# GRAD CAM FUNCTIONS 

def get_conv_layer(model, conv_layer_name):
    for name, layer in model.named_modules():
        if name == conv_layer_name:
            return layer
        
    raise ValueError(f'Layer {(conv_layer_name)} not found in the model')

# TODO: ensure that features.18 is the correct layer for your model
def compute_gradcam(model, img_tensor, class_idx, conv_layer_name='features.18'):
    conv_layer = get_conv_layer(model, conv_layer_name)

    # fwd hook to store activations
    # this will give us a feature map to use for our grad cam
    activations = None
    def forward_hook(module, input, output):
        nonlocal activations # COMMENT: what does this even mean??
        activations = output

    hook = conv_layer.register_forward_hook(forward_hook)

    # compute gradients
    img_tensor.requires_grad_(True)
    preds = model(img_tensor)
    loss = preds[:, class_idx]
    model.zero_grad()
    loss.backward()

    grads = img_tensor.grad.cpu().numpy()
    pooled_grads = np.mean(grads, axis=(0, 2, 3))

    # remove the hook
    hook.remove()

    activations = activations.detach().cpu().numpy()[0]
    for i in range(pooled_grads.shape[0]):
        activations[i, ...] *= pooled_grads[i]

    heatmap = np.mean(activations, axis=0)
    heatmap = np.maximum(heatmap, 0) # essentially a RELU activation fn
    heatmap /= np.max(heatmap)

    return heatmap

def overlay_heatmap(img_path, heatmap, alpha=.4):
    '''
    Returns a cv2 image with the heatmap superimposed onto it 
    '''
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)
    return superimposed_img

def show_gradcam_image(model_datapath, image_datapath, class_idx):
    #class_idx: 0 = non informative, 1 = informative

    #load in the model and image
    model = torch.load(model_datapath, map_location=torch.device('cpu'))
    image = Image.open(image_datapath)

    # Define a transform to convert PIL 
    # image to a Torch tensor
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    img_tensor = transform(image).unsqueeze(0)
    img_tensor = img_tensor.type(torch.float32)

    heatmap = data_helper.compute_gradcam(model, img_tensor, class_idx, conv_layer_name='0.features.18')
    overlay =data_helper.overlay_heatmap(image_datapath, heatmap, alpha=.4)
    img = Image.fromarray(overlay)
    return img
