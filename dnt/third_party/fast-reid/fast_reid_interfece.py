import os
import cv2
import requests
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
# from torch.backends import cudnn

from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch

__all__ = ['FastReIDInterface', 'setup_cfg', 'postprocess', 'preprocess']

def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False

    cfg.freeze()

    return cfg

def check_weights(weights_path):
    if not os.path.exists(weights_path):
            try:
                url = 'https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/'+os.path.basename(weights_path)
                response = requests.get(url, stream=True, allow_redirects=True)
                with open(weights_path, mode="wb") as file:
                    pbar = tqdm(unit="B", total=int(response.headers['Content-Length']), 
                                desc="Downloading "+os.path.basename(weights_path)+" ...")
                    for chunk in response.iter_content(chunk_size=10 * 1024):
                        file.write(chunk)
                        pbar.update(len(chunk))
            except ValueError:
                raise FileNotFoundError(f"File {weights_path} not found and cannot be downloaded from {url}!")
            
def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features

def preprocess(image, input_size):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[1], input_size[0], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size) * 114
    img = np.array(image)
    r = min(input_size[1] / img.shape[0], input_size[0] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    return padded_img, r

class FastReIDInterface:
    def __init__(self, config_file:str, weights_path:str, device:str='cuda', half:bool=True, batch_size:int=1):
        super(FastReIDInterface, self).__init__()
        if device != 'cpu':
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.half = half
        self.batch_size = batch_size

        check_weights(weights_path)
        
        self.cfg = setup_cfg(config_file, ['MODEL.WEIGHTS', weights_path])
        self.model = build_model(self.cfg)
        self.model.eval()

        Checkpointer(self.model).load(weights_path)

        if self.device == 'cuda':
            if self.half:
                self.model = self.model.eval().to(device='cuda').half()
            else:
                self.model = self.model.eval().to(device='cuda')
        else:
            self.model = self.model.eval()

        self.pH, self.pW = self.cfg.INPUT.SIZE_TEST

    def inference(self, image:np.array, detections:np.array)->np.array:

        if detections is None or np.size(detections) == 0:
            return []

        H, W, _ = np.shape(image)

        batch_patches = []
        patches = []
        for d in range(np.size(detections, 0)):
            tlbr = detections[d, :4].astype(np.int_)
            tlbr[0] = max(0, tlbr[0])                           #top left x   
            tlbr[1] = max(0, tlbr[1])                           #top left y
            tlbr[2] = min(W - 1, tlbr[2])                       #bottom right x
            tlbr[3] = min(H - 1, tlbr[3])                       #bottom right y
            patch = image[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2], :]  #crop the bbox

            # the model expects RGB inputs
            patch = patch[:, :, ::-1]

            # Apply pre-processing to image.
            patch = cv2.resize(patch, tuple(self.cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_LINEAR)
            # patch, scale = preprocess(patch, self.cfg.INPUT.SIZE_TEST[::-1])

            #plt.figure()
            #plt.imshow(patch)
            #plt.show()

            # Make shape with a new batch dimension which is adapted for network input
            patch = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))
            if self.half:
                patch = patch.to(device=self.device).half()
            patches.append(patch)

            if (d + 1) % self.batch_size == 0:
                patches = torch.stack(patches, dim=0)
                batch_patches.append(patches)
                patches = []

        if len(patches):
            patches = torch.stack(patches, dim=0)
            batch_patches.append(patches)

        features = np.zeros((0, 2048))
        # features = np.zeros((0, 768))

        for patches in batch_patches:
            if self.device == 'cuda':
                patches = patches.to(device='cuda')
            else:
                patches = patches.to(device='cpu')
            # Run model
            patches_ = torch.clone(patches)
            pred = self.model(patches)
            pred[torch.isinf(pred)] = 1.0

            feat = postprocess(pred)
            nans = np.isnan(np.sum(feat, axis=1))
            if np.isnan(feat).any():
                for n in range(np.size(nans)):
                    if nans[n]:
                        # patch_np = patches[n, ...].squeeze().transpose(1, 2, 0).cpu().numpy()
                        patch_np = patches_[n, ...]
                        patch_np_ = torch.unsqueeze(patch_np, 0)
                        pred_ = self.model(patch_np_)

                        patch_np = torch.squeeze(patch_np).cpu()
                        patch_np = torch.permute(patch_np, (1, 2, 0)).int()
                        patch_np = patch_np.numpy()

            features = np.vstack((features, feat))

        return features

if __name__=='__main__':
    cwd = os.path.dirname(os.path.abspath(__file__))
    extractor = FastReIDInterface(os.path.join(cwd, 'configs/MOT17/sbs_S50.yml'), 
                                  os.path.join(cwd, '/home/wonstran/dnt/detntrack_0.3alpha/dnt/track/botsort/checkpoint/mot17_sbs_S50.pth'))
    
    image = cv2.imread('/mnt/d/videos/ped2stage/frames/824.jpg')
    detections = np.array([[971, 482, 1016, 611], [913, 482, 963, 614]])
    features = extractor.inference(image, detections)
    print(features)