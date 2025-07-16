import torch.nn as nn
import torch
from torch.autograd import grad
from dgllife.model.gnn.attentivefp  import AttentiveFPGNN
from dgllife.model.readout.attentivefp_readout import AttentiveFPReadout
R=8.3144621;

class NN_ECS(nn.Module):
    def __init__(self,):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(2,32)
        
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32,32)
        
        self.fc4 = nn.Linear(32,32)
        self.fc5 = nn.Linear(32,32)

        self.fc_final = nn.Linear(32,2)
        
        self.mess_pass=AttentiveFPGNN(node_feat_size=12,edge_feat_size=5,num_layers=1,graph_feat_size=16,dropout=0)
        self.read_out=AttentiveFPReadout(16,num_timesteps=1,dropout=0)
    def forward(self,g1,n1,e1,g2,n2,e2,x):
        h1=self.mess_pass(g1,n1,e1)
        h1=self.read_out(g1,h1)    
        h2=self.mess_pass(g2,n2,e2)
        h2=self.read_out(g2,h2)
        
        h=h1*h2/(torch.sum(h1**2,dim=1)**0.5*torch.sum(h2**2,dim=1)**0.5).view(-1,1)  
        h=torch.concatenate((h,torch.ones_like(h)),dim=-1)
        
        
        out = self.fc1(x)
        
        out2 = self.fc2(out)
        out3=torch.tanh(out2)
        out4=self.fc3(out3)+out
        out5=torch.tanh(out4)
        
        out6=self.fc4(out5)
        out7=torch.tanh(out6)
        out8=self.fc5(out7)+out5
                
        ff=torch.mul(h,out8)
        ff=self.fc_final(ff)
        return ff
n_eos=[0.03982797,1.812227,-2.537512,-0.5333254,0.1677031,-1.323801,-0.6694654,0.8072718,-0.7740229,
   -0.01843846,1.407916,-0.4237082,-0.2270068,-0.805213,0.00994318,-0.008798793]
t_eos=[1.0,0.223,0.755,1.24,0.44,2.0,2.2,1.2,1.5,0.9,1.33,1.75,2.11,1.0,1.5,1.0]
d_eos=[4,1,1,2,3,1,3,2,2,7,1,1,3,3,2,1]
l_eos=[0]*5+[2,2,1,2,1]+[0]*6
eta_eos=[0]*10+[1.0,1.61,1.24,9.34,5.78,3.08]
beta_eos=[0]*10+[1.21,1.37,0.98,171,47.4,15.4]
gamma_eos=[0]*10+[0.943,0.642,0.59,1.2,1.33,0.64]
epsilon_eos=[0]*10+[0.728,0.87,0.855,0.79,1.3,0.71]
def cal_a00(tr,dr):  #
    A00=0
    tau=1/tr;delta=dr
    for i in range(5):
        A00+=n_eos[i]*delta**d_eos[i]*tau**t_eos[i]
    for i in range(5,10):
        A00+=n_eos[i]*delta**d_eos[i]*tau**t_eos[i]*torch.exp(-1*delta**l_eos[i])
    for i in range(10,16):
        A00+=n_eos[i]*delta**d_eos[i]*tau**t_eos[i]*torch.exp(-eta_eos[i]*(delta-epsilon_eos[i])**2-beta_eos[i]*(tau-gamma_eos[i])**2)
    return A00
def cal_a10(tr,dr):  
    A10=0
    tau=1/tr;delta=dr
    for i in range(5):
        A10+=n_eos[i]*delta**d_eos[i]*tau**t_eos[i]*t_eos[i]   
    for i in range(5,10):
        A10+=n_eos[i]*delta**d_eos[i]*tau**t_eos[i]*torch.exp(-1*delta**l_eos[i])*t_eos[i]
    for i in range(10,16):
        A10+=n_eos[i]*delta**d_eos[i]*tau**t_eos[i]*torch.exp(-eta_eos[i]*(delta-epsilon_eos[i])**2
                                                              -beta_eos[i]*(tau-gamma_eos[i])**2)*(t_eos[i]-2*beta_eos[i]*tau*(tau-gamma_eos[i]))
    return A10
def cal_a01(tr,dr):  
    A01=0
    tau=1/tr;delta=dr
    for i in range(5):
        A01+=n_eos[i]*delta**d_eos[i]*tau**t_eos[i]*d_eos[i]
    for i in range(5,10):
        A01+=n_eos[i]*delta**d_eos[i]*tau**t_eos[i]*torch.exp(-1*delta**l_eos[i])*(d_eos[i]-l_eos[i]*delta**l_eos[i])
    for i in range(10,16):
        A01+=n_eos[i]*delta**d_eos[i]*tau**t_eos[i]*+\
        torch.exp(-eta_eos[i]*(delta-epsilon_eos[i])**2-beta_eos[i]*(tau-gamma_eos[i])**2)*(d_eos[i]-2*eta_eos[i]*delta*(delta-epsilon_eos[i]))
    return A01
def cal_derivatives(model,g_tar,g_ref,input_,tc,dc):#
    out=model(g_tar,g_tar.ndata['atom'],g_tar.edata['bond'],g_ref,g_ref.ndata['atom'],g_ref.edata['bond'],input_,)
    etha=grad(out[:,0],input_,grad_outputs=torch.ones_like(out[:,0]),create_graph=True,retain_graph=True,)[0]
    phi=grad(out[:,1],input_,grad_outputs=torch.ones_like(out[:,1]),create_graph=True,retain_graph=True,)[0]
    
    etha_t=etha[:,0]*1/tc;etha_rho=etha[:,1]*1/dc
    phi_t=phi[:,0]*1/tc; phi_rho=phi[:,1]*1/dc
    
    Ft=(tc/382.51*etha_t)*(input_[:,0]*tc)/(tc/382.51*out[:,0])
    Frho=(tc/382.51*etha_rho)*(input_[:,1]*dc)/(tc/382.51*out[:,0])
    
    Ht=(4.29/dc*phi_t)*(input_[:,0]*tc)/(4.29/dc*out[:,1])
    Hrho=(4.29/dc*phi_rho)*(input_[:,1]*dc)/(4.29/dc*out[:,1])
    
    return Ft,Frho,Ht,Hrho
def cal_pressure(model,g_tar,g_ref,t,d,tc,dc):   
    input_=torch.concatenate((t.reshape(-1,1)/tc,d.reshape(-1,1)/dc),axis=-1).float()
    input_.requires_grad_(True)
    out=model(g_tar,g_tar.ndata['atom'],g_tar.edata['bond'],g_ref,g_ref.ndata['atom'],g_ref.edata['bond'],input_,)
    Ft,Frho,Ht,Hrho=cal_derivatives(model,g_tar,g_ref,input_,tc,dc)
    
    tr_new=input_[:,0]/out[:,0]
    dr_new=input_[:,1]*out[:,1]
    
    Z_ref=cal_a01(tr_new,dr_new);ur_ref=cal_a10(tr_new,dr_new)
    Z_tar=ur_ref*Frho+Z_ref*(1+Hrho)
    p=(Z_tar+1)*d*R*t
    return p*1e-3  #MPa

def cal_energy(model,g_tar,g_ref,t,d,tc,dc):#t,d : pytorch tensors of size [n];
    input_=torch.concatenate((t.reshape(-1,1)/tc,d.reshape(-1,1)/dc),axis=-1).float()
    input_.requires_grad_(True)
    
    out=model(g_tar,g_tar.ndata['atom'],g_tar.edata['bond'],g_ref,g_ref.ndata['atom'],g_ref.edata['bond'],input_,)
    Ft,Frho,Ht,Hrho=cal_derivatives(model,g_tar,g_ref,input_,tc,dc)
    tr_new=input_[:,0]/out[:,0];
    dr_new=input_[:,1]*out[:,1]
    
    Ar00=cal_a00(tr_new,dr_new)
    Ar10=cal_a10(tr_new,dr_new)
    Ar01=cal_a01(tr_new,dr_new)
    Zr_ref=Ar01
    ur_ref=Ar10
    hr_ref=Ar10+Ar01
    sr_ref=Ar10-Ar00
    gr_ref=Ar00+Ar01
    sr=sr_ref-ur_ref*Ft-Zr_ref*Ht
    hr=hr_ref+ur_ref*(Frho-Ft)+Zr_ref*(Hrho-Ht)
    ur=ur_ref*(1-Ft)-Zr_ref*Ht 
    gr=gr_ref+Zr_ref*Hrho+ur_ref*Frho
    return sr.detach()*R*1e-3,hr.detach()*R*t*1e-3  