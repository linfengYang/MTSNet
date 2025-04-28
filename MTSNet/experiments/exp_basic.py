import os
import torch
from model import Informer,InDecoder,ASB,DCT,WFTC,NDWF,Encoder,FreTS,MLP,NEW,SDF,OIP,FNL,DRFormer,OLD,DRNEW,DCTRElu,TEST,PathFormer


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Informer':TEST,
            'InDecoder':InDecoder,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0,1' if not self.args.use_multi_gpu else self.args.devices#str( self.args.gpu)#environ可使设备只看见哪几块GPU设备
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
