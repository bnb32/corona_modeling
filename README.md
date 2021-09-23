## COVID MODELING ##

Install package:
```
pip install -e .
```
 
Instantiate model:
```
from covid.models import SIR
model = SIR()
```

Instantiate dataset: 
```
from covid.fetch import Dataset
ds = Dataset({'state':<state>,'n_days':<n_days>})
```

Get data:
```
data = ds.get_data()
```

Fit model parameters: 
```
model.fit_parameters_from_data(data)
```

Run model:
```
model.initialize({k:data[k][0] for k in model.compartment_names})
output = model.run_model(n_days=<n_days>)
```

Plot output of specific compartment:
```
from covid.postprocessing import plot_compartment
plot_compartment(sim_data=output,raw_data=data,params=ds.params,<compartment_name>)
```

Change model parameters:
```
model.update_parameters(<params>)
```
