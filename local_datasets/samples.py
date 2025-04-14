
import numpy as np
from torch.utils.data import Dataset

def generate_data(n_samples_per_class, num_of_clusters, random_state=42, outlier_perc=0.01, size=1, outlier_range=1, outlier_dist=1):
    # np.random.seed(random_state)

    if num_of_clusters not in [2,3,4]: 
        raise AttributeError("num_of_clusters must be in [2,3,4]")
    
    cluster_1 = np.random.randn(n_samples_per_class, 2 * size) + np.array([4, 4] * size)
    labels_1 = np.zeros(n_samples_per_class)

    cluster_2 = np.random.randn(n_samples_per_class, 2 * size) - np.array([4, 4] * size)
    labels_2 = np.ones(n_samples_per_class)

    cluster_3 = np.random.randn(n_samples_per_class, 2 * size) - np.array([-4, 4] * size)
    labels_3 = np.ones(n_samples_per_class) + np.ones(n_samples_per_class)

    cluster_4 = np.random.randn(n_samples_per_class, 2 * size) - np.array([4, -4] * size)
    labels_4 = np.ones(n_samples_per_class) + np.ones(n_samples_per_class) + np.ones(n_samples_per_class)

    n_outliers = int(n_samples_per_class * outlier_perc)  # 5% of total points as outliers
    # print("N_Outliers: ", n_outliers * num_of_clusters)

    outliers = np.random.uniform(low=-outlier_dist*outlier_range, high=-(outlier_dist+1)*outlier_range, size=(n_outliers, 2 * size))
    outlier_labels = np.zeros(n_outliers)

    outliers2 = np.random.uniform(low=outlier_dist*outlier_range, high=(outlier_dist+1)*outlier_range, size=(n_outliers, 2 * size))
    outlier_labels2 = np.ones(n_outliers)

    outliers3 = np.random.uniform(low=outlier_dist*outlier_range, high=(outlier_dist+1)*outlier_range, size=(n_outliers, 2 * size)) * np.array([-1, 1] * size)
    outlier_labels3 = np.ones(n_outliers) + np.ones(n_outliers)

    outliers4 = np.random.uniform(low=outlier_dist*outlier_range, high=(outlier_dist+1)*outlier_range, size=(n_outliers, 2 * size)) * np.array([1, -1] * size)
    outlier_labels4 = np.ones(n_outliers) + np.ones(n_outliers) + np.ones(n_outliers)

    if num_of_clusters == 2: 
        X = np.vstack([
            cluster_1, cluster_2, #cluster_3, cluster_4, 
            outliers, outliers2, #outliers3, outliers4,
        ]).astype(np.float32)
        y = np.hstack([
            labels_1, labels_2, #labels_3, labels_4, 
            outlier_labels, outlier_labels2, #outlier_labels3, outlier_labels4,
        ]).astype(np.int64)
        return X, y
    elif num_of_clusters == 3: 
        X = np.vstack([
            cluster_1, cluster_2, cluster_3, #cluster_4, 
            outliers, outliers2, outliers3, #outliers4,
        ]).astype(np.float32)
        y = np.hstack([
            labels_1, labels_2, labels_3, #labels_4, 
            outlier_labels, outlier_labels2, outlier_labels3, #outlier_labels4,
        ]).astype(np.int64)
        return X, y
    else: 
        X = np.vstack([
            cluster_1, cluster_2, cluster_3, cluster_4, 
            outliers, outliers2, outliers3, outliers4,
        ]).astype(np.float32)
        y = np.hstack([
            labels_1, labels_2, labels_3, labels_4, 
            outlier_labels, outlier_labels2, outlier_labels3, outlier_labels4,
        ]).astype(np.int64)
        return X, y



class OutlierSamplesDataset(Dataset): 
    def __init__(self, n_samples_per_class, classes, outlier_perc=0.01, size=1, outlier_range=1, outlier_dist=1): 
        super().__init__()

        self.X, self.y = generate_data(n_samples_per_class, classes, outlier_perc=outlier_perc, size=size, outlier_range=outlier_range, outlier_dist=outlier_dist)

        self.is_outlier = [i > n_samples_per_class * classes for i in range(len(self.X))]

    def __len__(self): 
        return len(self.X)
    
    def __getitem__(self, index):
        return index, self.is_outlier[index], self.X[index], self.y[index]