from covid.misc import laplacian, params
from scipy.optimize import curve_fit
#, cases_to_intercepts, cases_to_infection_rate, vec_diff, get_infection_rate_ramp, cost_func
#import covid.fetch as fetch


import numpy as np

#logger = get_logger()
#parallel_execute=False
#parallel_execute=True

class CompartmentalModel:
    def __init__(self,compartment_names,parameters=None):
        self.compartment_number = len(compartment_names)
        self.transfer_matrix = np.zeros((self.compartment_number,self.compartment_number),dtype=object)
        self.compartment_names = {compartment_names[i]:i for i in range(self.compartment_number)}
        self.compartment_index = {i:compartment_names[i] for i in range(self.compartment_number)}
        self.compartment_series = {}
        
        if parameters is None:
            self.params = {}
            self.define_parameters()
        else:
            self.params = parameters

        self.params['n_substeps'] = 20
        self.params['dt'] = 1.0/self.params['n_substeps']

        self.define_transfer_matrix()

    def define_parameters(self):
        return

    def update_parameters(self,params):
        self.params = params
        self.define_transfer_matrix()
    
    def define_transfer_matrix(self):
        return

    def update_transfer_matrix(self,last_compartments):
        return

    def get_data(self):
        data = {'dates':[],
                'deaths':[],
                'infected':[],
                'hospitalized':[],
                'recovered':[]}

    def store_data(self,data):
        self.data = data
        return

    def get_parameters(self):
        return

    def transfer_value(self,to_compartment,from_compartment):
        return self.transfer_matrix[self.compartment_names[to_compartment],
                                    self.compartment_names[from_compartment]]
    
    def update_transfer_value(self,value,to_compartment,from_compartment):
        self.transfer_matrix[self.compartment_names[to_compartment],
                             self.compartment_names[from_compartment]] = value

    def last_compartment_value(self,compartment):
        return self.compartment_series[compartment][-1]

    def fit_parameters(self,data,lower_bounds,upper_bounds):
        days = [i for i in range(len(data))]
        popt,pconv = curve_fit(self.model_fit_function,
                               days,data,method='trf',
                               bounds=(lower_bounds,
                                       upper_bounds))
        return popt

    def model_fit_function(self,t,**kwargs):
        for k,v in kwargs.items():
            self.params[k] = v
        results = self.run_model(len(t))
        return results

    def initialize(self,initial_values):
        for name in self.compartment_names:
            self.compartment_series[name] = [initial_values[name]]

    def model_eqns(self,last_compartments):
        
        self.update_transfer_matrix(last_compartments)

        next_compartments = []
        for i in range(self.compartment_number):
            entry = 0
            for j in range(self.compartment_number):
                entry += self.transfer_matrix[i,j]*last_compartments[j]
            next_compartments.append(entry*self.params['dt'])

        return next_compartments
        
    def rk_step(self,last_compartments):

        rk_step=[0.5,0.5,1,0]
        tmp = [x for x in last_compartments]
        compartments_k = [[] for i in range(self.compartment_number)]

        for t in range(4):
            
            tmp = self.model_eqns(tmp)
            for i in range(self.compartment_number):
                compartments_k[i].append(tmp[i])
            tmp = np.multiply(tmp,rk_step[t]) + last_compartments

        stencil=[1,2,2,1]
        next_compartments = last_compartments + 1.0/6.0*np.tensordot(compartments_k,stencil,axes=(1,0))
        next_compartments[next_compartments < 0]=0
    
        return next_compartments

    def run_model(self,n_days):
        
        last_compartments = [self.compartment_series[name][0] for name in self.compartment_names]
        for day in range(n_days):
            for t in range(self.params['n_substeps']):
                last_compartments=self.rk_step(last_compartments)
                for i,name in enumerate(self.compartment_names):
                    self.compartment_series[name].append(last_compartments[i])

        return self.compartment_series

class SIR(CompartmentalModel):
    def __init__(self,parameters=None):
        super().__init__(['S','I','R'],parameters)
    
    def define_parameters(self):

        self.params['infection_rate'] = 3*14.0
        self.params['recovery_rate'] = 1.0/14.0

    def define_transfer_matrix(self):
        
        self.update_transfer_value(-self.params['recovery_rate'],'I','I')
        self.update_transfer_value(self.params['recovery_rate'],'R','I')

    def update_transfer_matrix(self,last_compartments):

        self.params['N'] = sum(last_compartments)
        alpha = self.params['infection_rate']*last_compartments[self.compartment_names['I']]/self.params['N']
        
        self.update_transfer_value(-alpha,'S','S') 
        self.update_transfer_value(alpha,'I','S')

class SIRV(CompartmentalModel):
    def __init__(self,parameters=None):
        super().__init__(['S','I','R','V'],parameters)
    
    def define_parameters(self):

        self.params['infection_rate'] = 3*14.0
        self.params['recovery_rate'] = 1.0/14.0
        self.params['vaccination_rate'] = 1.0/7.0

    def define_transfer_matrix(self):
        
        self.update_transfer_value(-self.params['recovery_rate'],'I','I')
        self.update_transfer_value(self.params['recovery_rate'],'R','I')
        self.update_transfer_value(self.params['vaccination_rate'],'V','S')

    def update_transfer_matrix(self,last_compartments):

        self.params['N'] = sum(last_compartments)
        alpha = self.params['infection_rate']*last_compartments[self.compartment_names['I']]/self.params['N']
        
        self.update_transfer_value(-alpha-self.params['vaccination_rate'],'S','S') 
        self.update_transfer_value(alpha,'I','S')

class SIRD(CompartmentalModel):
    def __init__(self,parameters=None):
        super().__init__(['S','I','R','D'],parameters)
    
    def define_parameters(self):

        self.params['infection_rate'] = 3*14.0
        self.params['recovery_rate'] = 1.0/14.0
        self.params['death_rate'] = 1.0/14.0

    def define_transfer_matrix(self):
        
        self.update_transfer_value(-self.params['recovery_rate']-self.params['death_rate'],'I','I')
        self.update_transfer_value(self.params['recovery_rate'],'R','I')
        self.update_transfer_value(self.params['death_rate'],'D','I')

    def update_transfer_matrix(self,last_compartments):

        self.params['N'] = sum(last_compartments)
        alpha = self.params['infection_rate']*last_compartments[self.compartment_names['I']]/self.params['N']
        
        self.update_transfer_value(-alpha,'S','S') 
        self.update_transfer_value(alpha,'I','S')

class Spatial_SIR(CompartmentalModel):
    def __init__(self,parameters=None):
        super().__init__(['S','I','R'],parameters)
        
    def define_parameters(self):

        self.params['infection_rate'] = 3*14.0
        self.params['recovery_rate'] = 1.0/14.0
        
        self.params['grid_size_x'] = 20
        self.params['grid_size_y'] = 20
        self.params['dx'] = 1
        self.params['dy'] = 1
        self.params['nx'] = int(self.params['grid_size_x']/self.params['dx'])
        self.params['ny'] = int(self.params['grid_size_y']/self.params['dy'])
        self.params['r0'] = 1.0

    def define_transfer_matrix(self):
        
        self.update_transfer_value(-self.params['recovery_rate'],'I','I')
        self.update_transfer_value(self.params['recovery_rate'],'R','I')

    def update_transfer_matrix(self,last_compartments):

        self.params['N'] = sum(last_compartments)
        alpha = self.params['infection_rate']/self.params['N']*(last_compartments[self.compartment_names['I']]+
                self.params['r0']**2/8.0*laplacian(last_compartments[self.compartment_names['I']],self.params['dx'],self.params['dy']))
        
        self.update_transfer_value(-alpha,'S','S') 
        self.update_transfer_value(alpha,'I','S')

    def example(self):

        S = np.zeros((self.params['nx'],self.params['ny']))
        I = np.zeros((self.params['nx'],self.params['ny']))
        R = np.zeros((self.params['nx'],self.params['ny']))
        
        S[:,:] = 10
        midx = self.params['nx']//2
        midy = self.params['ny']//2
        I[midx-2:midx+2,midy-2:midy+2] = 3
        
        self.compartment_series['S'] = [S]
        self.compartment_series['I'] = [I]
        self.compartment_series['R'] = [R]
    
    def initialize(self,initial_values=None):
        
        if initial_values is None:
            self.example()

class SEIR(CompartmentalModel):
    def __init__(self,parameters=None):
        super().__init__(['S','E','I','R'],parameters)
    
    def define_parameters(self):
        
        self.params['infection_rate'] = 3*14.0
        self.params['latency_rate'] = 1.0/7.0
        self.params['recovery_rate'] = 1.0/7.0

    def define_transfer_matrix(self):

        self.update_transfer_value(-self.params['latency_rate'],'E','E')
        self.update_transfer_value(self.params['latency_rate'],'I','E')
        self.update_transfer_value(-self.params['recovery_rate'],'I','I')
        self.update_transfer_value(self.params['recovery_rate'],'R','I')

    def update_transfer_matrix(self,last_compartments):

        self.params['N'] = sum(last_compartments)
        alpha = self.params['infection_rate']*last_compartments[self.compartment_names['I']]/self.params['N']
        
        self.update_transfer_value(-alpha,'S','S') 
        self.update_transfer_value(alpha,'E','S')

class SEAIQHRD(CompartmentalModel):
    def __init__(self,parameters=None):
        super().__init__(['S','E','A','I','Q','H','R','D'],parameters)
       
    def define_parameters(self):

        self.params['infection_rate'] = 3*14.0
        
        self.params['E_to_A_rate'] = 1.0/4.0
        
        self.params['A_to_I_rate'] = 1.0/7.0
        self.params['A_to_R_rate'] = 1.0/3.0
        
        self.params['I_to_Q_rate'] = 1.0/2.0
        self.params['I_to_H_rate'] = 1.0/3.0
        self.params['I_to_R_rate'] = 1.0/3.0
        self.params['I_to_D_rate'] = 1.0/3.0
        
        self.params['Q_to_H_rate'] = 1.0/5.0
        self.params['Q_to_R_rate'] = 1.0/3.0
        self.params['Q_to_D_rate'] = 1.0/3.0
        
        self.params['H_to_R_rate'] = 1.0/3.0
        self.params['H_to_D_rate'] = 1.0/3.0

        self.params['E_decay_rate'] = self.params['E_to_A_rate']
        self.params['A_decay_rate'] = np.sum([self.params['A_to_I_rate'],
                                              self.params['A_to_R_rate']])
        self.params['H_decay_rate'] = np.sum([self.params['H_to_R_rate'],
                                              self.params['H_to_D_rate']])
        self.params['Q_decay_rate'] = np.sum([self.params['Q_to_H_rate'],
                                              self.params['Q_to_R_rate'],
                                              self.params['Q_to_D_rate']])
        self.params['I_decay_rate'] = np.sum([self.params['I_to_Q_rate'],
                                              self.params['I_to_H_rate'],
                                              self.params['I_to_R_rate'],
                                              self.params['I_to_D_rate']])

    def define_transfer_matrix(self):

        self.update_transfer_value(-self.params['A_decay_rate'],'A','A')
        self.update_transfer_value(-self.params['E_decay_rate'],'E','E')
        self.update_transfer_value(-self.params['I_decay_rate'],'I','I')
        self.update_transfer_value(-self.params['Q_decay_rate'],'Q','Q')
        self.update_transfer_value(-self.params['H_decay_rate'],'H','H')
        
        self.update_transfer_value(self.params['E_to_A_rate'],'A','E')
        self.update_transfer_value(self.params['A_to_I_rate'],'I','A')
        self.update_transfer_value(self.params['A_to_R_rate'],'R','A')
        self.update_transfer_value(self.params['I_to_Q_rate'],'Q','I')
        self.update_transfer_value(self.params['I_to_H_rate'],'H','I')
        self.update_transfer_value(self.params['I_to_D_rate'],'D','I')
        self.update_transfer_value(self.params['I_to_R_rate'],'R','I')
        self.update_transfer_value(self.params['Q_to_H_rate'],'H','Q')
        self.update_transfer_value(self.params['Q_to_R_rate'],'R','Q')
        self.update_transfer_value(self.params['Q_to_D_rate'],'D','Q')
        self.update_transfer_value(self.params['H_to_R_rate'],'R','H')
        self.update_transfer_value(self.params['H_to_D_rate'],'D','H')

    def update_transfer_matrix(self,last_compartments):

        self.params['N'] = sum(last_compartments)
        div=self.params['N']
        div-=last_compartments[self.compartment_names['Q']]
        div-=last_compartments[self.compartment_names['H']]
        div-=last_compartments[self.compartment_names['D']]
            
        alpha = self.params['infection_rate']*(last_compartments[self.compartment_names['I']]+
                                               last_compartments[self.compartment_names['A']])/div
        
        self.update_transfer_value(-alpha,'S','S')
        self.update_transfer_value(alpha,'E','S')

