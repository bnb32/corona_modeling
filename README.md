## COVID MODELING ##

Run `pip install -e .`

All models are in covid.models.

For example: `from covid.models import Spatial_SIR`

Instantiate model `spatial_sir = Spatial_SIR()` and intialize using `spatial_sir.initialize(<initial_values_dictionary>)`

Instantiate dataset with `from covid.fetch import Dataset` and `ds = Dataset({'state':<state>,'n_days':<n_days>})`

Get data with `data = ds.get_data()`

Fit model parameters using `spatial_sir.fit_parameters_from_data(<data_dictionary>)`

Run model using `output = spatial_sir.run_model(<n_days>)`

Plot output of specific compartment with `output[<compartment_name>]`

Can change model parameters with `spatial_sir.update_parameters(<params>)`
