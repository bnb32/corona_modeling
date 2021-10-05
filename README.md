## COVID MODELING ##

Install package:
```bash
pip install -e .
```
 
Instantiate model:
```python
from covid.models import SIR
model = SIR()
```

Instantiate dataset: 
```python
from covid.fetch import Dataset
ds = Dataset({'state':<state>,'n_days':<n_days>})
```

Get data:
```python
data = ds.get_data()
```

Fit model parameters: 
```python
model.fit_parameters_from_data(data)
```

Run model:
```python
model.initialize({k:data[k][0] for k in model.compartment_names})
output = model.run_model(n_days=<n_days>)
```

Plot output of specific compartment:
```python
from covid.postprocessing import plot_compartment
plot_compartment(sim_data=output,raw_data=data,params=ds.params,<compartment_name>)
```

Change model parameters:
```python
model.update_parameters(<params>)
```
