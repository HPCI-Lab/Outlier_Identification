
import random
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch 

from local_datasets.common import get_is_outlier_list, inject_label_outliers, inject_img_outliers

class OutlierMNISTDatasetWrapper(Dataset): 
    def __init__(self, dataset, outlier_num=2, classes=10, samples_per_class=100): 
        super().__init__()

        self.outlier_num = outlier_num
        self.classes = classes


        classes_limit = {k:0 for k in range(self.classes)}
        self.X, self.y = [], []
        for s, l in dataset: 
            if l in list(range(self.classes)): 
                if classes_limit[l] < samples_per_class: 
                    self.X.append(s)
                    self.y.append(l)
                    classes_limit[l] += 1

        self.X = inject_img_outliers(self.X, self.outlier_num)
        self.y = inject_label_outliers(self.y, self.outlier_num, classes)
        self.is_outlier = get_is_outlier_list(classes*samples_per_class, outlier_number=self.outlier_num)

    def __len__(self): 
        return len(self.X)
    
    def __getitem__(self, index):
        s, l = self.X[index], self.y[index]
        return index, self.is_outlier[index], s / 255, F.one_hot(torch.tensor(l), num_classes=self.classes).float()