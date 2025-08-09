# Dump intermediate features from PyTorch model for a single image
import os, sys, argparse, numpy as np, torch, cv2
CUR_DIR=os.path.dirname(os.path.abspath(__file__)); PT_ROOT=os.path.abspath(os.path.join(CUR_DIR,'..'))
if PT_ROOT not in sys.path: sys.path.insert(0, PT_ROOT)
from nanodet.util import load_config, cfg
from nanodet.model.arch import build_model
from nanodet.data.transform.pipeline import Pipeline

def parse():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--img', required=True)
    ap.add_argument('--save', default='workspace/torch_feats')
    ap.add_argument('--device', default='cuda:0')
    return ap.parse_args()

def main():
    a=parse(); device=a.device
    load_config(cfg, a.cfg)
    os.makedirs(a.save, exist_ok=True)
    model=build_model(cfg.model).eval()
    sd=torch.load(a.ckpt, map_location='cpu')
    if 'state_dict' in sd: sd=sd['state_dict']
    msd=model.state_dict(); 
    # strip wrappers
    fixed={}
    for k,v in sd.items():
        if k.startswith('module.'): k=k[7:]
        if k.startswith('model.'): k=k[6:]
        fixed[k]=v
    # load strictly
    model.load_state_dict(fixed, strict=False)
    # prepare img strictly via val Pipeline
    img=cv2.imread(a.img); h,w=img.shape[:2]
    meta=dict(img=img, img_info={'file_name':os.path.basename(a.img),'height':h,'width':w,'id':0})
    pipe=Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    meta=pipe(None, meta, cfg.data.val.input_size)
    im=meta['img'].transpose(2,0,1)[None]
    np.save(os.path.join(a.save,'input_after_pipe.npy'), im)
    im=torch.from_numpy(im).float()
    feats={}
    feats={}
    def cap_backbone(x):
        y=bb_org(x); feats['backbone']=y; return y
    def cap_fpn(x):
        y=fpn_org(x); feats['fpn']=y; return y
    def cap_head(x):
        y=hd_org(x); feats['head_out']=y; return y
    bb_org, fpn_org, hd_org = model.backbone.forward, model.fpn.forward, model.head.forward
    model.backbone.forward = cap_backbone
    model.fpn.forward = cap_fpn
    model.head.forward = cap_head
    with torch.no_grad(): _=model(im)
    # move to cpu numpy
    def tonp(t): return t.detach().float().cpu().numpy()
    bb0,bb1,bb2=[tonp(f) for f in feats['backbone']]
    fpn0,fpn1,fpn2,fpn3=[tonp(f) for f in feats['fpn']]
    head=tonp(feats['head_out'])
    np.savez(os.path.join(a.save,'pt_feats.npz'),
             bb0=bb0,bb1=bb1,bb2=bb2,
             fpn0=fpn0,fpn1=fpn1,fpn2=fpn2,fpn3=fpn3,
             head=head)
    print('saved to', os.path.join(a.save,'pt_feats.npz'))

if __name__=='__main__': main()

