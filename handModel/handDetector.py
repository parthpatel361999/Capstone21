import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2

from handModel.data import cfg
from handModel.layers.functions.prior_box import PriorBox
from handModel.utils.nms_wrapper import nms
from handModel.models.faceboxes import FaceBoxes
from handModel.utils.box_utils import decode
from handModel.utils.timer import Timer
from PIL import Image, ImageDraw


class HandDetector():
    def __init__(self):
        torch.set_grad_enabled(False)
        # net and model
        self.net = FaceBoxes(phase='test', size=None, num_classes=2)  # initialize detector

        self.net = load_model(self.net, 'handModel/weights/Final_HandBoxes.pth', False)
        self.net = torch.nn.DataParallel(self.net, [1, 0]).cuda()  # multiprocessing edler look here
        self.net.to(f'cuda:{self.net.device_ids[0]}')
        self.net.eval()
        print('Finished loading model!')
        # print(net)
        cudnn.benchmark = True
        self.device = torch.device(f'cuda:{self.net.device_ids[0]}')
        # net = net.to(device)

        # testing scale
        self.resize = 2

        self._t = {'forward_pass': Timer(), 'misc': Timer()}
        return
    def detectHand(self, frame):
        to_show = frame
        img = np.float32(to_show)

        if self.resize != 1:
            img = cv2.resize(img, None, None, fx=self.resize, fy=self.resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        self._t['forward_pass'].tic()
        out = self.net(img)  # forward pass
        self._t['forward_pass'].toc()
        self._t['misc'].tic()
        priorbox = PriorBox(cfg, out[2], (im_height, im_width), phase='test')
        priors = priorbox.forward()
        priors = priors.to(self.device)
        loc, conf, _ = out
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > 0.95)[0]  # EDLER SCORE THRESHOLD
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:5000]  # top k
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        # keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(dets, 0.01, force_cpu=False)  # nms_threshold
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:1, :]  # keep top k
        self._t['misc'].toc()
        if len(dets) > 0:
            if len(dets[0]) >= 4 and dets[0][4] >= 0.95:
                return dets[0]
            else:
                return None
        else:
            return None

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def draw_hand(hand,mainImg_1):
    lineWidth = 2
    tempImg = np.zeros((mainImg_1.shape[0],mainImg_1.shape[1],3),np.uint8)
    tempImg = cv2.cvtColor(tempImg,cv2.COLOR_BGR2RGB)
    tempImg = Image.fromarray(tempImg)
    tempPILDraw = ImageDraw.Draw(tempImg, mode='RGBA')
    if (hand is not None) and len(hand) >= 4:
        tempPILDraw.line([hand[0], hand[1]], fill=tuple([255, 0, 0]), width=lineWidth)
        tempPILDraw.line([hand[1], hand[2]], fill=tuple([255, 0, 0]), width=lineWidth)
        tempPILDraw.line([hand[2], hand[3]], fill=tuple([255, 0, 0]), width=lineWidth)
        tempPILDraw.line([hand[3], hand[0]], fill=tuple([255, 0, 0]), width=lineWidth)
    tempImg = np.array(tempImg)
    tempImg = cv2.cvtColor(tempImg, cv2.COLOR_RGB2BGR)
    return tempImg