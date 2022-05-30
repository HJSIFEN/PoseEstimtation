import shutil
import os
import numpy as np
from datasets import TestAlphaPose
import torch
def image_classification(model, json_file, image_dir,  save_dir):
    
    target = np.array(['front', 'back', 'left', 'right'])
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        for t in target:
            os.mkdir(os.path.join(save_dir, t))
    
    datasets = TestAlphaPose(json_file)

    for data, image_id in datasets:
        data = torch.tensor(data).float()
        pred = model(data.unsqueeze(0)) 
        shutil.copyfile(os.path.join(image_dir, image_id), os.path.join(save_dir, target[pred.argmax()],image_id))
