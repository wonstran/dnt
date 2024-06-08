import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
import os, sys, requests
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../third_party/fast-reid/")))

from model import Net
from fastreid.config import get_cfg # type: ignore
from fastreid.engine import DefaultTrainer # type: ignore
from fastreid.utils.checkpoint import Checkpointer # type: ignore

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


class FastReIDExtractor(object):
    def __init__(self, model_config, model_path, use_cuda=True):
        cfg = get_cfg()
        cfg.set_new_allowed(True)
        cfg.merge_from_file(model_config)
        cfg.MODEL.BACKBONE.PRETRAIN = False
        self.net = DefaultTrainer.build_model(cfg)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"

        if not os.path.exists(model_path):
            try:
                url = 'https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/'+os.path.basename(model_path)
                response = requests.get(url, stream=True, allow_redirects=True)
                with open(model_path, mode="wb") as file:
                    pbar = tqdm(unit="B", total=int(response.headers['Content-Length']), 
                                desc="Downloading "+os.path.basename(model_path)+" ...")
                    for chunk in response.iter_content(chunk_size=10 * 1024):
                        file.write(chunk)
                        pbar.update(len(chunk))
            except ValueError:
                raise FileNotFoundError(f"File {model_path} not found and cannot be downloaded from {url}!")
            
        Checkpointer(self.net).load(model_path)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.net.eval()
        height, width = cfg.INPUT.SIZE_TEST
        self.size = (width, height)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()

if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    img_file = os.path.join(dirname, "demo.jpg")
    imgs = []
    imgs.append(cv2.imread(img_file)[:,:,(2,1,0)])
    cfg_file = os.path.join(dirname, "fastreid.yaml")
    model_1 = os.path.join(dirname, "checkpoint/ckpt.t7")
    model_2 = os.path.join(dirname, "checkpoint/market_bot_R50.pth")
    extr1 = Extractor(model_1, use_cuda=True)
    extr2 = FastReIDExtractor(cfg_file, model_2)
    feature1 = extr1(imgs)
    feature2 = extr2(imgs)
    print(feature1.shape, feature2.shape)

