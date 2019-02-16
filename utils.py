import torch
from torch import nn

def device_templ(tens,cuda=False):
    return tens.cuda() if cuda else tens


def projection(S,k):
    #check whether S is already feasile
    capS = torch.clamp(S,0,1)
    if capS.sum()-k<1e-10:
        return S

    S = torch.flip(S,dims=[0])
    #find the (i,j) region
    for i in range(0,len(S)):
        for j in range(i,len(S)):
            capS = S
            if i>0:
                capS[:i]=0
            if j<len(S)-1:
                capS[j:]=1
            shf = (k-capS[i:j+1].sum())/(j-i+1)
            capS[i:j+1] = capS[i:j+1]+shf
            if capS[i]>=0 and (i==0 or S[i-1]+shf<=0) and capS[j]<=1 and (j==len(S)-1 or S[j+1]+shf>=1): 
                S=capS
                S = torch.flip(S,dims=[0])
                return S

def objective(X,U):
    mat = torch.mm(X,U)
    mat = mat*mat
    objective = mat.sum(1).sum(0).div(X.size(0))
    return objective

def get_syn_data(N = 10000,d = 100,device=device_templ):
    X = nn.init.uniform_(device(torch.FloatTensor(N,d)))
    U,S,V = torch.svd(X)
    S = device(torch.exp(torch.arange(len(S)).float()/10))
    X = torch.mm(torch.mm(U,torch.diag(S)),V.t())
    X_train = X[:int(N*0.7)]
    X_val = X[int(N*0.7):int(N*0.85)]
    X_test = X[int(N*0.85):]
    return X_train,X_val,X_test

#gram schmidt orthogonalization
def gram_schmidt(A):
    #performs qr decomposition and returns Q matrix
    return torch.qr(A)[0]

def sort_and_select(U,S):
    S,idx = torch.sort(S,descending=True)
    U = torch.index_select(U,1,idx)
    return U,S

def incremental_update(U,S,x,max_rank = None):
    # print(U.size(),S.size())
    xproj = torch.matmul(U.transpose(0,1),x)
    xres = x-torch.matmul(U,xproj)
    xresnorm = torch.norm(xres)
    xres = xres.div(xresnorm.clamp(min=1e-6))

    eta=1
    Q = torch.cat([torch.diag(S)+torch.ger(xproj,xproj).mul(eta),eta*xresnorm*xproj.unsqueeze(1)],dim=1)
    Q = torch.cat([Q,torch.cat([xproj.mul(xresnorm*eta),(eta*xresnorm**2).unsqueeze(0)]).unsqueeze(0)])

    S2,U2 = torch.symeig(Q,eigenvectors=True)
    U = torch.matmul(torch.cat([U,xres.unsqueeze(1)],dim=1),U2)
    S = S2
    U,S = sort_and_select(U,S)
    if max_rank is not None and S.size()>max_rank:
        S = S[:max_rank]
        U = U[:,:max_rank]
    return U,S

def rank1_update(U,S,x,eta,eps):
    xproj = torch.matmul(U.transpose(0,1),x)
    xres = x-torch.matmul(U,xproj)
    xresnorm = torch.norm(xres)
    # print(xres,xresnorm)
    xres = xres.div(xresnorm.clamp(min=1e-6))
    #eps thresholding ensures that the residual is added to the orthogonal list only if the component is significant enough.
    #when U is full rank, then explicit updates are not performed to SVD vectors of Q
    if U.size(1)>=U.size(0) or xresnorm<eps:
        Q = torch.diag(S)+torch.ger(xproj,xproj).mul(eta)
    else:
        Q = torch.cat([torch.diag(S)+torch.ger(xproj,xproj).mul(eta),eta*xresnorm*xproj.unsqueeze(1)],dim=1)
        Q = torch.cat([Q,torch.cat([xproj.mul(xresnorm*eta),(eta*xresnorm**2).unsqueeze(0)]).unsqueeze(0)])
        U = torch.cat([U,xres.unsqueeze(1)],dim=1)
    S2,U2 = torch.symeig(Q,eigenvectors=True)

    # print(Q.size(),U.size(),xres.size(),U2.size(),S2.size())
    U = torch.matmul(U,U2)
    S = S2
    U,S = sort_and_select(U,S)
    # print(U,S)
    return U,S

def stochastic_power_update(U,x,eta):
    #eta learning rate
    U = U + torch.mm(torch.ger(x,x),U).mul(eta)
    return U

def msg(U,S,k,x,eta,eps,beta):
    #beta l1 regularization const
    #eps threshold on norm of residuals for non-trivial updates
    #eta learning rate
    U,S = rank1_update(U,S,x,eta,eps)
    S = S-beta*eta 
    S =projection(S,k)
    return U,S


