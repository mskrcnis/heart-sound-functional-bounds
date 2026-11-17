"""The nominal-confidence control for support-response matching."""
from query_training_common import *

def nominal_numpy(p,potentials):
    T=len(p);weights=rewards(T);values=[]
    for pm,d,a in potentials(p):
        path=solve(*prepare(pm,d,a))[3]
        selected=np.asarray(pm,dtype=np.float64)[np.arange(T),path]
        values.append((weights*selected[None,:]).sum(axis=1)/T)
    return np.asarray(values)

def nominal_torch(p,potentials):
    T=len(p);weights=torch.as_tensor(rewards(T),device=p.device,dtype=torch.float64);values=[]
    for cols,(pm,d,a) in zip(COLS,potentials(p.detach().cpu().numpy())):
        path=solve(*prepare(pm,d,a))[3]
        selected=p[:,cols][torch.arange(T,device=p.device),torch.as_tensor(path,device=p.device)].double()
        values.append((weights*selected[None,:]).sum(dim=1)/T)
    return torch.stack(values)
