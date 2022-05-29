from covid.models import SIR, SIRV
from covid.fetch import Dataset
from covid.postprocessing import plot_compartments_comparison, plot_compartment_comparison, plot_compartment, plot_compartments

import pickle
import argparse
import pandas as pd


def covid_argparse():

    parser = argparse.ArgumentParser(description='Plot SIRV Model')
    parser.add_argument('-state', default='Washington')
    parser.add_argument('-fit_compartments', default=['S','I','R','V'])
    parser.add_argument('-plot_compartment', default='I')
    parser.add_argument('-sim_days', default=100, type=int)
    parser.add_argument('-data_days', default=100, type=int)
    parser.add_argument('-out_file', type=str)
    return parser

if __name__ == '__main__':
    parser = covid_argparse()
    args = parser.parse_args()

    model = SIRV()

    ds = Dataset({'state':args.state,'n_days':args.data_days})

    data = ds.get_data()

    model.fit_compartments = args.fit_compartments

    model.get_parameters_from_data(data)

    print(f'model parameters: {pd.DataFrame(model.params,index=[0])}')

    model.initialize({k:data[k][-1] for k in model.compartment_names})

    output = model.run_model(n_days=args.sim_days)

    plot_compartments(sim_data=output,raw_data=data,params=ds.params)

    plot_compartment(sim_data=output,raw_data=data,params=ds.params,compartment=args.compartment)

    model.initialize({k:data[k][0] for k in model.compartment_names})

    model.start_step = 0

    output = model.run_model(n_days=args.sim_days)

    plot_compartments_comparison(sim_data=output,raw_data=data,params=ds.params)

    plot_compartment_comparison(sim_data=output,raw_data=data,params=ds.params,compartment=args.compartment)

    with open(args.out_file, 'wb') as fh:
        pickle.dump(output, fh)
