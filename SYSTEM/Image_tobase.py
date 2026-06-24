import base64
import os 
def imageconvert (name, folder_path = "SYSTEM/Data"):
    file_name = os.path.join(folder_path , name)
    if not os.path.exists(file_name):
        print(f"{file_name} not found ")
        return None
    with open(file_name , "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    return image_b64
