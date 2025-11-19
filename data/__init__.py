import os.path as path
from .deep_lesion.deep_lesion import DeepLesion

def get_dataset(dataset_type, **dataset_opts):
    return {
        "deep_lesion": DeepLesion,
    }[dataset_type](**dataset_opts[dataset_type])
