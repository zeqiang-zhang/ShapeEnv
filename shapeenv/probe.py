"""MLP probes used for all latent-probe results of the paper (Appendix C.2).

Probe: Linear(d,256)-ReLU-Dropout(0.2)-Linear(256,256)-ReLU-Dropout(0.2)-Linear(256,C).
Training: 70/30 split (stratified for classification, ``random_state=42``),
Adam (lr 1e-3, weight decay 1e-4), batch 512, at most 100 epochs, early
stopping with patience 10 on the held-out loss, best checkpoint restored.

Requires the optional dependencies ``torch`` and ``scikit-learn``
(``pip install shapeenv[probe]``). The code is copied verbatim from the
paper's probe scripts.
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class MLPClf(nn.Module):
    def __init__(self, d, c, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h,h),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h,c))
    def forward(self,x): return self.net(x)
    def predict(self,x):
        self.eval()
        with torch.no_grad(): return self.forward(torch.tensor(x,dtype=torch.float32).to(next(self.parameters()).device)).argmax(-1).cpu().numpy()


class MLPReg(nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h,h),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h,1))
    def forward(self,x): return self.net(x)
    def predict(self,x):
        self.eval()
        with torch.no_grad(): return self.forward(torch.tensor(x,dtype=torch.float32).to(next(self.parameters()).device)).cpu().numpy().squeeze(-1)


def train_clf(feats, labels, n_cls, device, epochs=100, patience=10):
    """Fit a classification probe. Returns (train_acc, test_acc, model)."""
    Xtr,Xte,ytr,yte = train_test_split(feats,labels,test_size=0.3,random_state=42,stratify=labels)
    m = MLPClf(Xtr.shape[1],n_cls).to(device)
    opt = torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4); crit=nn.CrossEntropyLoss()
    Xt=torch.tensor(Xtr,dtype=torch.float32).to(device); yt=torch.tensor(ytr,dtype=torch.long).to(device)
    Xv=torch.tensor(Xte,dtype=torch.float32).to(device); yv=torch.tensor(yte,dtype=torch.long).to(device)
    ld=DataLoader(TensorDataset(Xt,yt),batch_size=512,shuffle=True)
    bl,bs2,ni=float('inf'),None,0
    for ep in range(epochs):
        m.train()
        for xb,yb in ld: opt.zero_grad();crit(m(xb),yb).backward();opt.step()
        m.eval()
        with torch.no_grad(): tl=crit(m(Xv),yv).item()
        if tl<bl: bl=tl;bs2={k:v.clone() for k,v in m.state_dict().items()};ni=0
        else: ni+=1
        if ni>=patience: break
    if bs2: m.load_state_dict(bs2)
    m.eval()
    with torch.no_grad():
        tra=(m(Xt).argmax(-1)==yt).float().mean().item(); tea=(m(Xv).argmax(-1)==yv).float().mean().item()
    return tra, tea, m


def train_reg(feats, labels, device, epochs=100, patience=10):
    """Fit a regression probe. Returns (train_mse, test_mse, model)."""
    Xtr,Xte,ytr,yte = train_test_split(feats,labels,test_size=0.3,random_state=42)
    m = MLPReg(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4); crit=nn.MSELoss()
    Xt=torch.tensor(Xtr,dtype=torch.float32).to(device); yt=torch.tensor(ytr,dtype=torch.float32).unsqueeze(-1).to(device)
    Xv=torch.tensor(Xte,dtype=torch.float32).to(device); yv=torch.tensor(yte,dtype=torch.float32).unsqueeze(-1).to(device)
    ld=DataLoader(TensorDataset(Xt,yt),batch_size=512,shuffle=True)
    bl,bs2,ni=float('inf'),None,0
    for ep in range(epochs):
        m.train()
        for xb,yb in ld: opt.zero_grad();crit(m(xb),yb).backward();opt.step()
        m.eval()
        with torch.no_grad(): tl=crit(m(Xv),yv).item()
        if tl<bl: bl=tl;bs2={k:v.clone() for k,v in m.state_dict().items()};ni=0
        else: ni+=1
        if ni>=patience: break
    if bs2: m.load_state_dict(bs2)
    m.eval()
    with torch.no_grad():
        trm=crit(m(Xt),yt).item(); tem=crit(m(Xv),yv).item()
    return trm, tem, m
