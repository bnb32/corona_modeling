import tools

#simulation length
n_days=30

#state
state_long="New York"
state_short="NY"
print("State: %s" %state_long)

#county
county=None
print("County: %s" %county)

#get number of positive cases
I0=tools.location_to_cases(state_short)
print("Current cases: %s" %I0)

#get population
N=tools.location_to_population(state_long,county=county)
print("Population: %s"%N)

#estimate doubling time from case trend
Td=tools.location_to_doubling_time(state_short)
print("Doubling time: %s" %Td)

#number of recovered and susceptible
R=0
S=N-I0-R

#get beta and gamma from initial values
#Sd is social distancing percentage
#Tr is recovery time in days

beta,gamma=tools.model_parameters(S,I0,R,Tr=14,Td=Td,Sd=0.0)
print("Basic reproduction number: %s"%(beta/gamma))

#run simulation
St,It,Rt=tools.sim_sir(S,I0,R,beta,gamma,n_days)

#plot infected curve
#saved in ../../web/<state>_<county>_infected.png
tools.plot_infected(It,state_long,county=county)
