import shutil
import os

def delete_images(location, dir):   
    # # location
    # location = "C:\Users\papep\Documents\GitHub\msdc-thesis\!tool\temp"
    
    # # directory
    # dir = "animation_images"
    
    # path
    path = os.path.join(location, dir)
    
    # removing directory
    shutil.rmtree(path, ignore_errors = True)