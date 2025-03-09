import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoProcessor
import io
from PIL import Image
import folder_paths
from pathlib import Path

# Define the directory for saving files related to the MCLLaVA model
# files_for_sa2va_model = Path(folder_paths.folder_names_and_paths["LLavacheckpoints"][0][0]) / "files_for_mcllava"
# files_for_sa2va_model.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

class Sa2VABase:
  RETURN_TYPES = ("STRING", )
  RETURN_NAMES = ("caption", )
  FUNCTION = "run"
  CATEGORY = "Sa2VAWrapper/chat"

  def run(self, images, prompt):
    model_path = snapshot_download("ByteDance/Sa2VA-8B",
                                        # local_dir=files_for_sa2va_model,
                                        force_download=True,  # Set to True if you always want to download, regardless of local copy
                                        local_files_only=False,  # Set to False to allow downloading if not available locally
                                        local_dir_use_symlinks="auto",  # or set to True/False based on your symlink preference
                                        ignore_patterns=["*.bin", "*.jpg", "*.png"]  # Exclude certain file types
                                      ) 
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoProcessor.from_pretrained(
       model_path, 
       trust_remote_code=True
    )
    
    # Convert the input tensor to PIL images
    vid_frames = []
    # Assuming images is a tensor of shape [batch_size, height, width, channels]
    for i in range(images.shape[0]):
        # Convert tensor to numpy array and ensure proper format
        img_array = images[i].cpu().numpy()
        # Ensure values are in the right range for PIL
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
        # Create PIL image
        img = Image.fromarray(img_array)
        vid_frames.append(img)

    result = model.predict_forward(
      video=vid_frames,
      text=prompt,
      tokenizer=tokenizer,
    )

    prediction = result['prediction']

    return (prediction, )
    
  

class GetCaptionFromImages(Sa2VABase):
  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "images": ("IMAGE", ),
        "prompt": ("STRING", {"default": "Describe the scene in great detail. Be sure to describe the movement of the people, animals, or objects through the scene with great detail."})
      },
    }
  
  RETURN_TYPES = ("STRING", )
  RETURN_NAMES = ("caption", )
  FUNCTION = "run"

  CATEGORY = "Sa2VAWrapper/chat"

  def run(self, images, prompt):
    return super().run(images, prompt)



# class GetSegmentationFromVideo(Sa2VABase):

#   @classmethod
#   def INPUT_TYPES(self):
#     return {
#       "required": {
#         "images": ("IMAGE", ),
#       },
#     }
    
