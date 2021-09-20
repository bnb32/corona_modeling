from covid.misc import get_logger, laplacian, cases_to_intercepts, cases_to_beta, vec_diff, get_beta_ramp, cost_func
import covid.fetch as fetch

import numpy as np

logger = get_logger()
parallel_execute=False
#parallel_execute=True

class params:
    pass

class CompartmentalModel:
    def __init__(self,compartment_names):
        self.compartment_number = len(compartment_names)
        self.transfer_matrix = np.zeros((self.compartment_number,self.compartment_number))
        self.compartment_names = {compartment_names[i]:i for i in range(self.compartment_number)}
        self.compartment_index = {i:compartment_names[i] for i in range(self.compartment_number)}
        self.compartment_series = [[0] for i in range(self.compartment_number)]
        self.params = params()
        self.params.n_substeps = 20
        self.params.dt = 1.0/self.params.n_substeps
        self.params.running_update = False

    def update_parameters(self):
        return
    
    def update_transfer_matrix(self,last_compartments):
        return

    def get_data(self):
        data = {'dates':[],
                'deaths':[],
                'infected':[],
                'hospitalized':[],
                'recovered':[]

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
        return self.compartment_series[-1][self.compartment_names[compartment]]

    def initialize(self,initial_values):
        self.compartment_series = [initial_values]

    def model_eqns(self,last_compartments):
        
        self.update_transfer_matrix(last_compartments)

        return np.dot(self.transfer_matrix,last_compartments)*self.params.dt
        
    def rk_step(self,last_compartments):

        if self.params.running_update:
            self.update_parameters()
        
        rk_step=[0.5,0.5,1,0]
        tmp = [x for x in last_compartments]
        compartments_k = [[] for i in range(self.compartment_number)]

        for t in range(4):
            
            tmp = self.model_eqns(tmp)
            for i in range(self.compartment_number):
                compartments_k[i].append(tmp[i])
            tmp = np.multiply(tmp,rk_step[t]) + np.array(last_compartments)

        stencil=[1,2,2,1]
        next_compartments = []
        for i in range(self.compartment_number):
            entry = last_compartments[i] + 1.0/6.0*np.dot(compartments_k[i],stencil)
            if entry < 0.0: entry=0.0
            next_compartments.append(entry)
    
        return next_compartments

    def run_model(self,n_days):
        
        last_compartments = self.compartment_series[0]
        for day in range(n_days):
            for t in range(self.params.n_substeps):
                last_compartments=self.rk_step(last_compartments)
                self.compartment_series.append(last_compartments)

        self.results = {}
        for i in range(self.compartment_number):
            self.results[self.compartment_index[i]] = np.array([self.compartment_series[t][i] for t in range(len(self.compartment_series))])
        
        return self.results

class SIR_general(CompartmentalModel):
    def __init__(self):
        super().__init__(['S','I','R'])
        
        self.params.beta = 3*14.0
        self.params.state = 'Washington'
        self.params.county = None
        self.params.n_days = 10

        self.params.I_to_R_rate = 1.0/14.0
        
        self.update_transfer_value(-self.params.I_to_R_rate,'I','I')
        self.update_transfer_value(self.params.I_to_R_rate,'R','I')

    def update_transfer_matrix(self,last_compartments):

        self.params.N = sum(last_compartments)
        alpha = self.params.beta*last_compartments[self.compartment_names['I']]/self.params.N
        
        self.update_transfer_value(-alpha,'S','S') 
        self.update_transfer_value(alpha,'I','S')

class SEAIQHRD_general(CompartmentalModel):
    def __init__(self):
        super().__init__(['S','E','A','I','Q','H','R','D'])
        self.params.beta = 3*14.0
        
        self.params.E_to_A_rate = 1.0/4.0
        
        self.params.A_to_I_rate = 1.0/7.0
        self.params.A_to_R_rate = 1.0/3.0
        
        self.params.I_to_Q_rate = 1.0/2.0
        self.params.I_to_H_rate = 1.0/3.0
        self.params.I_to_R_rate = 1.0/3.0
        self.params.I_to_D_rate = 1.0/3.0
        
        self.params.Q_to_H_rate = 1.0/5.0
        self.params.Q_to_R_rate = 1.0/3.0
        self.params.Q_to_D_rate = 1.0/3.0
        
        self.params.H_to_R_rate = 1.0/3.0
        self.params.H_to_D_rate = 1.0/3.0

        self.params.E_decay_rate = self.params.E_to_A_rate
        self.params.A_decay_rate = np.mean([self.params.A_to_I_rate,
                                            self.params.A_to_R_rate])
        self.params.H_decay_rate = np.mean([self.params.H_to_R_rate,
                                            self.params.H_to_D_rate])
        self.params.Q_decay_rate = np.mean([self.params.Q_to_H_rate,
                                            self.params.Q_to_R_rate,
                                            self.params.Q_to_D_rate])
        self.params.I_decay_rate = np.mean([self.params.I_to_Q_rate,
                                            self.params.I_to_H_rate,
                                            self.params.I_to_R_rate,
                                            self.params.I_to_D_rate])
        
        self.update_transfer_value(-self.params.A_decay_rate,'A','A')
        self.update_transfer_value(-self.params.E_decay_rate,'E','E')
        self.update_transfer_value(-self.params.I_decay_rate,'I','I')
        self.update_transfer_value(-self.params.Q_decay_rate,'Q','Q')
        self.update_transfer_value(-self.params.H_decay_rate,'H','H')
        
        self.update_transfer_value(self.params.E_to_A_rate,'A','E')
        self.update_transfer_value(self.params.A_to_I_rate,'I','A')
        self.update_transfer_value(self.params.A_to_R_rate,'R','A')
        self.update_transfer_value(self.params.I_to_Q_rate,'Q','I')
        self.update_transfer_value(self.params.I_to_H_rate,'H','I')
        self.update_transfer_value(self.params.I_to_D_rate,'D','I')
        self.update_transfer_value(self.params.I_to_R_rate,'R','I')
        self.update_transfer_value(self.params.Q_to_H_rate,'H','Q')
        self.update_transfer_value(self.params.Q_to_R_rate,'R','Q')
        self.update_transfer_value(self.params.Q_to_D_rate,'D','Q')
        self.update_transfer_value(self.params.H_to_R_rate,'R','H')
        self.update_transfer_value(self.params.H_to_D_rate,'D','H')

    def update_transfer_matrix(self,last_compartments):

        self.params.N = sum(last_compartments)
        div=self.params.N
        div-=last_compartments[self.compartment_names['Q']]
        div-=last_compartments[self.compartment_names['H']]
        div-=last_compartments[self.compartment_names['D']]
            
        alpha = self.params.beta*(last_compartments[self.compartment_names['I']]+last_compartments[self.compartment_names['A']])/div
        
        self.update_transfer_value(-alpha,'S','S')
        self.update_transfer_value(alpha,'E','S')

