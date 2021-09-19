def model_parameters(S,I,R,Tr=14.0,Td=6.0,Sd=0.0):
    gamma=1/Tr
    g=2**(1/Td)-1
    N=S+I+R
    beta=(g+gamma)*N/S*(1-Sd)
    return beta,gamma

def location_to_initial_values(lat,lon):
    N=1000
    I=1
    R=0
    S=N-I-R
    
    return S,I,R
