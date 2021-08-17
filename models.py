import numpy as np
from model_functions import laplacian

def sir_spatial_euler(S,I,R,P,beta,gamma,r0,dr,dt):
    
    Sk1=-(I+(r0*r0/8.0)*laplacian(I,dr))*(beta*S/P)*dt
    Rk1=gamma*I*dt
    Ik1=-(Sk1+Rk1)
    
    Sn=Sk1+S
    In=Ik1+I
    Rn=Rk1+R
    
    Sn[np.where(Sn<0)[0]]=0
    In[np.where(In<0)[0]]=0
    Rn[np.where(Rn<0)[0]]=0
    
    return Sn,In,Rn

def sir_spatial_rk(S,I,R,P,beta,gamma,r0,dr,dt):
    
    #1st step    
    Sk1=-(I+(r0*r0/8.0)*laplacian(I,dr))*(beta*S/P)*dt
    Rk1=gamma*I*dt
    Ik1=-(Sk1+Rk1)

    #2nd step
    Sn=Sk1/2+S
    In=Ik1/2+I
    Rn=Rk1/2+R
    Sk2=-(In+(r0*r0/8.0)*laplacian(In,dr))*(beta*Sn/P)*dt
    Rk2=gamma*In*dt
    Ik2=-(Sk2+Rk2)

    
    #3rd step
    Sn=Sk2/2+S
    In=Ik2/2+I
    Rn=Rk2/2+R
    Sk3=-(In+(r0*r0/8.0)*laplacian(In,dr))*(beta*Sn/P)*dt
    Rk3=gamma*In*dt
    Ik3=-(Sk3+Rk3)


    #3rd step
    Sn=Sk3*dt+S
    In=Ik3*dt+I
    Rn=Rk3*dt+R
    Sk4=-(In+(r0*r0/8.0)*laplacian(In,dr))*(beta*Sn/P)*dt
    Rk4=gamma*In*dt
    Ik4=-(Sk4+Rk4)
    
    Sn=S+1/6*(Sk1+2*Sk2+2*Sk3+Sk4)
    In=I+1/6*(Ik1+2*Ik2+2*Ik3+Ik4)
    Rn=R+1/6*(Rk1+2*Rk2+2*Rk3+Rk4)

    Sn[np.where(Sn<0)[0]]=0
    In[np.where(In<0)[0]]=0
    Rn[np.where(Rn<0)[0]]=0
    
    return Sn,In,Rn   

def sir(S,I,R,beta,gamma,N):
    Sn=(-beta*S*I)+S
    In=(beta*S*I-gamma*I)+I
    Rn=gamma*I+R

    if Sn<0:
        Sn=0
    if In<0:
        In=0
    if Rn<0:
        Rn=0

    scale=N/(Sn+In+Rn)
    return Sn*scale,In*scale,Rn*scale

def seir(S,E,I,R,beta,gamma,a,N):
    Sn=(-beta*S*I)+S
    En=(beta*S*I-a*E)+E
    In=(a*E-gamma*I)+I
    Rn=gamma*I+R

    if Sn<0:
        Sn=0
    if En<0:
        En=0
    if In<0:
        In=0
    if Rn<0:
        Rn=0

    scale=N/(Sn+En+In+Rn)
    return Sn*scale,En*scale,In*scale,Rn*scale

def sim_sir(S,I,R,beta,gamma,n_days,beta_decay=None):
    N=S+I+R
    s,i,r=[S],[I],[R]
    for days in range(n_days):
        S,I,R=sir(S,I,R,beta,gamma,N)
        if beta_decay:
            beta*=(1-beta_decay)
        s.append(S)
        i.append(I)
        r.append(R)
    s,i,r=np.array(s),np.array(i),np.array(r)
    return s,i,r

def sim_seir(S,E,I,R,beta,gamma,a,n_days,beta_decay=None):
    N=S+E+I+R
    s,e,i,r=[S],[E],[I],[R]
    for days in range(n_days):
        S,E,I,R=seir(S,E,I,R,beta,gamma,a,N)
        if beta_decay:
            beta*=(1-beta_decay)
        s.append(S)
        e.append(E)
        i.append(I)
        r.append(R)
    s,e,i,r=np.array(s),np.array(e),np.array(i),np.array(r)
    return s,e,i,r
