from torch.utils.data import Dataset, DataLoader
import os, glob
import numpy as np

def gen_Datalist(path, mode):
    adcs = sorted(glob.glob(path + '/adc/%s/*' % mode))
    b1000s = sorted(glob.glob(path + '/b1000/%s/*' % mode))
    masks = sorted(glob.glob(path + '/mask/%s/*' % mode))
      
    np_list={'ADC':adcs, 'b1000':b1000s, 'Mask':masks}
    return np_list

class CustomDset(Dataset):
    def __init__(self, root, mode, img_transform=None, label_transform=None):
        
        patient_list = os.listdir(root)
        if len(patient_list) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"))
                
        self.root = root
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.patient_list = patient_list
        self.Datalist = gen_Datalist(root, mode)
        
    def __getitem__(self, index):
        
        ADC_ = self.Datalist['ADC'][index]
        ADC = np.expand_dims(np.load(ADC_), axis=0)
        b1000_ = self.Datalist['b1000'][index]
        b1000 = np.expand_dims(np.load(b1000_), axis=0)
        mask_ = self.Datalist['Mask'][index]
        mask = np.expand_dims(np.load(mask_), axis=0)
        mask = np.where(mask!=0, 1, 0).astype(np.uint8)
        
        patient = os.path.basename(ADC_).split('.')[0]
        
        if self.img_transform is not None:
            ADC = self.img_transform(ADC)
            b1000 = self.img_transform(b1000)

        if self.label_transform is not None:
            mask = self.label_transform(mask)
            
        return patient, ADC, b1000, mask
        
    def __len__(self):
        return len(self.Datalist['ADC'])