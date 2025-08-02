
import jittor as jt
from jittor import init
import logging
from jittor import nn
logger = logging.getLogger('NanoDet')

class TIMMWrapper(nn.Module):

    def __init__(self, model_name, features_only=True, pretrained=True, checkpoint_path='', in_channels=3, **kwargs):
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError('timm is not installed, please install it first') from exc
        super(TIMMWrapper, self).__init__()
        self.timm = timm.create_model(model_name=model_name, features_only=features_only, pretrained=pretrained, in_chans=in_channels, checkpoint_path=checkpoint_path, **kwargs)
        self.timm.global_pool = None
        self.timm.fc = None
        self.timm.classifier = None
        feature_info = getattr(self.timm, 'feature_info', None)
        if feature_info:
            logger.info(f'TIMM backbone feature channels: {feature_info.channels()}')

    def execute(self, x):
        outs = self.timm(x)
        if isinstance(outs, (list, tuple)):
            features = tuple(outs)
        else:
            features = (outs,)
        return features
