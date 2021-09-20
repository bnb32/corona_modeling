## COVID MODELING ##

Run `pip install -e .`

All models are in covid.models.

For example: `from covid.models import Spatial_SIR`

Instantiate model `spatial_sir = Spatial_SIR()` and intialize using `spatial_sir.initialize(<initial_values_dictionary>)`

Run model using `output = spatial_sir.run_model(<n_days>)`

Plot output of specific compartment with `output[compartment_name]`

Can change model parameters with `spatial_sir.update_parameters(<params>)`
