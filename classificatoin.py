import shutil
import os
import numpy as np
from datasets import TestRawData

def image_classiciation(model, dir="C:\\Users\\user\\Desktop\\RE-ID\\datasets\\DukeMTMC-reID\\query", save_dir ="C:\\Users\\user\\Desktop\\temp"):
    target = np.array(['front', 'back', 'left', 'right'])
    datasets = TestRawData(dir = dir)

    for t in target:
        if not os.path.exists(os.path.join(save_dir, t)):
            os.mkdir(os.path.join(save_dir, t))

    for data in datasets:
        if data == -1:
            continue

        pred = model(data[0].unsqueeze(0))
        image_path = data[1]
        image_name = image_path.split('\\')[-1]
        
        
        shutil.copyfile(image_path, os.path.join(save_dir, target[pred.argmax()], image_name))
