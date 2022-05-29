"""Example model"""
from covid.models import SIRVD
from covid.fetch import Dataset
from covid.postprocessing import (plot_compartments_comparison,
                                  plot_compartment_comparison,
                                  plot_compartments, plot_compartment)


model = SIRVD()

ds = Dataset({'state': 'Washington', 'n_days': 50})

data = ds.get_data()

#model.fit_derivatives = False
#model.fit_compartments = ['I', 'D']

model.get_parameters_from_data(data)

#print(f'model parameters: \n{pd.DataFrame(model.params, index = [0])}')
print(f'model parameters: {model.params}')

model.initialize({k: data[k][-1] for k in model.compartment_names})

output = model.run_model(n_days = 90)

plot_compartments(sim_data = output, raw_data = data, params = ds.params)

plot_compartment(sim_data = output, raw_data = data, params = ds.params, compartment = 'I')

model.initialize({k: data[k][0] for k in model.compartment_names})

model.start_step = 0

output = model.run_model(n_days = 300)

plot_compartments_comparison(sim_data = output, raw_data = data, params = ds.params)

plot_compartment_comparison(sim_data = output, raw_data = data, params = ds.params, compartment = 'I')
