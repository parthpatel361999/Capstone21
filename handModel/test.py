#from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
import cv2
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from utils.timer import Timer



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


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # net and model
    net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector

    net = load_model(net, 'weights/Final_HandBoxes.pth', False)
    net = torch.nn.DataParallel(net, [1,0]).cuda()  # multiprocessing edler look here
    net.to(f'cuda:{net.device_ids[0]}')
    net.eval()
    print('Finished loading model!')
    #print(net)
    cudnn.benchmark = True
    device = torch.device(f'cuda:{net.device_ids[0]}')
    #net = net.to(device)

    # testing scale
    resize = 2

    _t = {'forward_pass': Timer(), 'misc': Timer()}


    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if ret:
            #frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
            #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            #print(frame.shape)
            to_show = frame
            img = np.float32(to_show)

            if resize != 1:
                img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)

            _t['forward_pass'].tic()
            out = net(img)  # forward pass
            _t['forward_pass'].toc()
            _t['misc'].tic()
            priorbox = PriorBox(cfg, out[2], (im_height, im_width), phase='test')
            priors = priorbox.forward()
            priors = priors.to(device)
            loc, conf, _ = out
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.data.cpu().numpy()[:, 1]

            # ignore low scores
            inds = np.where(scores > 0.95)[0] #EDLER SCORE THRESHOLD
            boxes = boxes[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:5000] #top k
            boxes = boxes[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            # keep = py_cpu_nms(dets, args.nms_threshold)
            keep = nms(dets, 0.01, force_cpu=False) #nms_threshold
            dets = dets[keep, :]

            # keep top-K faster NMS
            dets = dets[:1, :] #keep top k
            _t['misc'].toc()

            for i in range(dets.shape[0]):
                cv2.rectangle(to_show, (dets[i][0], dets[i][1]), (dets[i][2], dets[i][3]), [0, 0, 255], 3)
                #print(str(dets[i][0]) + " " + str(dets[i][1]) + " " + str(dets[i][2]) + " " + str(dets[i][3]))
                cv2.putText(to_show, str(dets[i][4]), (int(dets[i][0]) + 5, int(dets[i][1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('image', to_show)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            cv2.waitKey(1)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()