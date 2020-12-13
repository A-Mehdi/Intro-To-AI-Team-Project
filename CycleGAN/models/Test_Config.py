from utils import Preprocess
from models import ResnetGenerator
import cv2
import torch
import numpy as np

class Face2Cartoon:
    def __init__(self):
        self.pre = Preprocess()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = ResnetGenerator(ngf=32, img_size=256, light=True).to(self.device)
        params = torch.load('./train_model/train-size256-ch32-True-lr0.0001-adv1-cyc50-id1-identity10-cam1000/face2cartoon_params_latest.pt', map_location=self.device)
        self.net.load_state_dict(params['genA2B'])

    def inference(self, img):
        # face alignment
        face_rgba = self.pre.process(img)
        if face_rgba is None:
            print('can not detect face!!!')
            return None

        face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
        face = face_rgba[:, :, :3].copy()
        mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
        face = (face*mask + (1-mask)*255) / 127.5 - 1

        face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
        face = torch.from_numpy(face).to(self.device)

        # start inference
        with torch.no_grad():
            cartoon = self.net(face)[0][0]

        cartoon = np.transpose(cartoon.cpu().numpy(), (1, 2, 0))
        cartoon = (cartoon + 1) * 127.5
        cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
        cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
        return cartoon