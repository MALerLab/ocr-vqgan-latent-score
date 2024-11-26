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
  model_name = "sqocremavq_240_gray_fit"
  config_path = list((Path("logs")/ model_name).rglob("*project.yaml"))[0]
  config = OmegaConf.load(config_path)

  model = EMAVQ(**config.model.params)
  
  ckpt_path = list((Path("logs")/ model_name).rglob("*.ckpt"))[0]
  sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
  missing, unexpected = model.load_state_dict(sd, strict=False)
  model.cuda().eval()
  torch.set_grad_enabled(False)
  del sd
  
  image_path_list = list(Path("/home/sake/userdata/latent_score_dataset/string_quartet/segments").rglob("*/*/images/flattened_resampled/240_gray/*.png"))
  
  for image_path in tqdm(image_path_list):
    save_path = (image_path.parent.parent.parent.parent / "image_tokens" / image_path.parent.parent.name / image_path.parent.name / model_name / image_path.name ).with_suffix(".pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
      continue
    
    print("Encoding : ", image_path)
    image = PIL.Image.open(image_path)
    image = tf(image).unsqueeze(0).cuda()
    image = image.to(torch.float32) * 2.0 - 1.0
    quant, emb_loss, info = model.encode(image)
    info = info[2].view(quant.shape[0],quant.shape[-2],quant.shape[-1])
    
    torch.save(info, save_path)
    # break