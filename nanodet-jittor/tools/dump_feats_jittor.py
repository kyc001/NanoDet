# Dump intermediate features from Jittor model for a single image
import os, sys, argparse, numpy as np, jittor as jt, cv2, torch
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
JT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
PT_ROOT = os.path.abspath(os.path.join(JT_ROOT, "../nanodet-pytorch"))
if JT_ROOT not in sys.path: sys.path.insert(0, JT_ROOT)
if PT_ROOT in sys.path: sys.path.remove(PT_ROOT)
from nanodet.util import load_config, cfg, Logger
from nanodet.model.arch import build_model
from nanodet.data.transform.pipeline import Pipeline
from nanodet.util.check_point import load_model_weight
from collections import OrderedDict

from infer_from_pt_ckpt import pt_state_to_jt_checkpoint

def parse():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--img', required=True)
    ap.add_argument('--save', default='result/jt_feats')
    return ap.parse_args()

def main():
    a=parse(); jt.flags.use_cuda=1
    os.makedirs(a.save, exist_ok=True)
    load_config(cfg, a.cfg)
    logger=Logger(save_dir=a.save, name='dump_jt')
    model=build_model(cfg.model); model.eval()
    # load pt->jt
    pt_ckpt=torch.load(a.ckpt, map_location='cpu')
    jt_ckpt=pt_state_to_jt_checkpoint(pt_ckpt, model=model, prefer_avg=True)
    load_model_weight(model, jt_ckpt, logger)
    # prepare meta
    img=cv2.imread(a.img); h,w=img.shape[:2]
    meta=dict(img_info={'file_name':os.path.basename(a.img),'height':h,'width':w,'id':0}, img=img, raw_img=img)
    pipe=Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    meta=pipe(None, meta, cfg.data.val.input_size)
    im=meta['img'].transpose(2,0,1); im=np.ascontiguousarray(im); np.save(os.path.join(a.save,'input_after_pipe.npy'), im[None]) ; meta['img']=jt.array(im).unsqueeze(0)
    # hooks: monkey-patch execute with saved originals to avoid recursion
    feats={}
    bb_org = model.backbone.execute
    fpn_org = model.fpn.execute
    hd_org = model.head.execute
    def hook_backbone(x):
        y = bb_org(x)
        feats['backbone'] = [f if isinstance(f, jt.Var) else f for f in y]
        return y
    def hook_fpn(x):
        y = fpn_org(x)
        feats['fpn'] = [f if isinstance(f, jt.Var) else f for f in y]
        return y
    def hook_head(x):
        y = hd_org(x)
        feats['head_out'] = y
        return y
    model.backbone.execute = hook_backbone
    model.fpn.execute = hook_fpn
    model.head.execute = hook_head
    # forward
    _ = model(meta['img'])
    # save
    np.savez(os.path.join(a.save,'jt_feats.npz'),
             bb0=feats['backbone'][0], bb1=feats['backbone'][1], bb2=feats['backbone'][2],
             fpn0=feats['fpn'][0], fpn1=feats['fpn'][1], fpn2=feats['fpn'][2], fpn3=feats['fpn'][3],
             head=feats['head_out'])
    print('saved to', os.path.join(a.save,'jt_feats.npz'))

if __name__=='__main__': main()

