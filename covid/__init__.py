from covid.models import SIR
from covid.fetch import Dataset
from covid.postprocessing import plot_comparison, plot_compartment_comparison
import argparse

def main():

    parser = argparse.ArgumentParser(description='Plot SIR Model')
    parser.add_argument('-state', default='Washington')
    parser.add_argument('-compartment', default='I')
    parser.add_argument('-sim_days', default=100, type=int)
    parser.add_argument('-data_days', default=100, type=int)
    args = parser.parse_args()

    sir = SIR()

    ds = Dataset({'state':args.state,'n_days':args.data_days})

    data = ds.get_data()

    sir.initialize({'S':data['S'][0],
                    'I':data['I'][0],
                    'R':data['R'][0]})

    sir.fit_compartment = 'I'

    sir.get_parameters_from_data({'S':data['S'],
                                  'I':data['I'],
                                  'R':data['R']})

    output = sir.run_model(n_days=args.sim_days)

    plot_comparison(sim_data=output,raw_data=data,params=ds.params)

    plot_compartment_comparison(sim_data=output,raw_data=data,params=ds.params,compartment=args.compartment)

    return output

if __name__=='__main__':
    main()
