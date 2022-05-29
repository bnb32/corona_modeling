**************
COVID MODELING
**************

Install package:

.. code-block:: bash

    pip install -e .


Instantiate model:
.. code-block:: python

    from covid.models import SIR
    model = SIR()

Instantiate dataset:
.. code-block:: python

    from covid.fetch import Dataset
    ds = Dataset({'state': <state>, 'n_days': <n_days>})

Get data:
.. code-block:: python

    data = ds.get_data()

Fit model parameters:
.. code-block:: python

    model.fit_parameters_from_data(data)

Run model:
.. code-block:: python

    model.initialize({k: data[k][0] for k in model.compartment_names})
    output = model.run_model(n_days=<n_days>)

Plot output of specific compartment:
.. code-block:: python

    from covid.postprocessing import plot_compartment
    plot_compartment(sim_data=output, raw_data=data, params=ds.params, <compartment_name>)

Change model parameters:
.. code-block:: python

    model.update_parameters(<params>)
