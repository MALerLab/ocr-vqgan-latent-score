import torch
import torchvision
tf = torchvision.transforms.ToTensor()
from omegaconf import OmegaConf
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import PIL

from taming.models.vqgan import EMAVQ

if __name__ == "__main__":
  # model_name = "sqocremavq_16_240_gray_unfit"
  model_name = "gsocremavq_16_128_gray_unfit"
  config_path = list((Path("logs")/ model_name).rglob("*project.yaml"))[0]
  config = OmegaConf.load(config_path)

  model = EMAVQ(**config.model.params)
  
  ckpt_path = list((Path("logs")/ model_name).rglob("*.ckpt"))[0]
  sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
  missing, unexpected = model.load_state_dict(sd, strict=False)
  model.cuda().eval()
  torch.set_grad_enabled(False)
  del sd
  
  # image_path_list = list(Path("/home/sake/userdata/latent_score_dataset/string_quartet/segments").rglob("*/*/images/flattened_resampled/240_gray/*.png"))
  image_path_list = list(Path("/home/sake/userdata/olimpic_dataset/grandstaff-lmx").rglob("**/*.jpg"))
  filtered_pathlist = []
  for p in image_path_list:
    if "distorted" in p.name:
      continue
    elif p.name.startswith("."):
      continue
    else:
      filtered_pathlist.append(p)
  image_path_list = filtered_pathlist
  
  for image_path in tqdm(image_path_list):
    save_path = image_path.parent / (image_path.stem + f":{model_name}_shifted") / (image_path.stem + f":{model_name}_shifted")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # if Path(str(save_path) + ":0_0.pt").exists():
    #   continue
    
    print("Encoding : ", image_path)
    image = PIL.Image.open(image_path).convert("L")
    aspect_ratio = image.width / image.height
    new_width = int(128 * aspect_ratio)
    image = image.resize((new_width, 128))
    resized_image_path = image_path.with_stem(image_path.stem + "_128")
    image.save(resized_image_path)
    image = tf(image).cuda()
    for i in range(16):
      x_shifted_img = image[...,i:image.shape[-1]-15+i]
      y_shifted_imgs = []
      for j in range(4):
        y_shifted_imgs.append(torch.nn.functional.pad(x_shifted_img[:, j:x_shifted_img.shape[-2]-4+j], (0, 0, 4-j, j), mode='replicate'))
      y_shifted_imgs = torch.stack(y_shifted_imgs)
      shifted_quant, shifted_emb_loss, shifted_info = model.encode(y_shifted_imgs.cuda())
      unflattended_out = shifted_info[2].view(shifted_quant.shape[0],shifted_quant.shape[-2],shifted_quant.shape[-1])
      
      for j in range(4):
        torch.save(unflattended_out[j].unsqueeze(0), str(save_path) + f":{j}_{i}.pt")
    # break