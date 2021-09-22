from covid.models import SIR, SIRV
from covid.fetch import Dataset
from covid.postprocessing import plot_comparison, plot_compartment_comparison

sirv = SIRV()

ds = Dataset({'state':'Washington','n_days':150})

data = ds.get_data()

sirv.initialize({'S':data['S'][0],
                'I':data['I'][0],
                'R':data['R'][0],
                'V':data['V'][0]})

sirv.fit_compartment = 'I'

sirv.get_parameters_from_data({'S':data['S'],
                              'I':data['I'],
                              'R':data['R'],
                              'V':data['V']})

output = sirv.run_model(n_days=300)

plot_comparison(sim_data=output,raw_data=data,params=ds.params)

plot_compartment_comparison(sim_data=output,raw_data=data,params=ds.params,compartment='I')
