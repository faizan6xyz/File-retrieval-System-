import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import os
def run_vision_encoder(name , folder_path = "SYSTME/data"):
    file_name  = os.path.join(folder_path , name )
    if not os.path.exists(file_name):
        print(f"{file_name} not found ")
        return None    
    # 1. Load the Vision Encoder (We use CLIP as an example) , 'openai/clip-vit-base-patch32' is a common, lightweight encoder
    model_name = "openai/clip-vit-base-patch32"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    # 2. Load and Preprocess the image , The processor handles resizing, normalization, and tensor conversion automatically
    image = Image.open(file_name).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    # 3. Run the Encoder (The "Translation" Step) , 
    with torch.no_grad(): # We don't need to calculate gradients for inference
        outputs = model.get_image_features(**inputs)
    # 4. The Result: A Vector Embedding , # This is a list of numbers that represents the "meaning" of the image
    image_embedding = outputs.image_embeds
    print(f"Embedding Shape: {image_embedding.shape}") 
    # Output: torch.Size([1, 512]) -> 512 numbers representing the image
    return image_embedding

if __name__ == "__main__" :
    translation = run_vision_encoder("n.png") 