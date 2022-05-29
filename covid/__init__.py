"""Covid module"""
import argparse


def covid_argparse():
    """Parse args for running compartmental model"""
    parser = argparse.ArgumentParser(description='Plot SIRV Model')
    parser.add_argument('-state', default='Washington')
    parser.add_argument('-fit_compartments', default=['S', 'I', 'R', 'V'])
    parser.add_argument('-plot_compartment', default='I')
    parser.add_argument('-sim_days', default=100, type=int)
    parser.add_argument('-data_days', default=100, type=int)
    parser.add_argument('-out_file', type=str)
    return parser
