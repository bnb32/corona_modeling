import covid.models
import covid.fetch
import covid.postprocessing as display

from datetime import date, datetime, timedelta
from os import environ, path
from tempfile import gettempdir
import json
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from scipy import stats
import logging
from decimal import Decimal
from sys import stdout

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sh = logging.StreamHandler(stdout)
formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)
logging.getLogger('boto').setLevel(logging.WARN)
logging.getLogger('boto3').setLevel(logging.WARN)
logging.getLogger('botocore').setLevel(logging.WARN)
logging.getLogger('s3transfer').setLevel(logging.WARN)
logging.getLogger('urllib3').setLevel(logging.WARN)

def get_logger():
    return logger

def json_handler(o):
    if isinstance(o, datetime):
        return o.isoformat()
    elif isinstance(o, Decimal):
        return float(o)
    else:
        logger.warning('Unknown Type in json_handler: ' + str(o))
        return str(o)
    
logger = get_logger()

def run_comp_and_plot(state,county=None,
                     n_days=30,
                     n_ramp=3,
                     Tea=0.5,
                     Tiq=0.5,
                     Tai=5.0,
                     Tir=14.0,
                     Tid=20.0,
                     Tih=1.0,
                     piq=0.9,
                     pai=0.6,
                     rai=0.4,
                     Sd=0.6,
                     Sd_period=0,
                     Sd_delay=12,
                     detection_rate=0.2,
                     data_days=30,
                     n_substeps=10,
                     refit=False):

    logger.info('Running run_comp_and_plot')
    
    model = models.SEAIQHRD(state,county,
                            n_days,n_substeps,n_ramp,
                            Tea,Tai,Tir,Tiq,Tid,
                            Tih,pai,piq,rai,Sd,
                            Sd_delay,detection_rate,
                            data_days,refit)
    '''

    model = models.SIR(state,county,
                       n_days,n_substeps,n_ramp,
                       Tir,Sd,Sd_delay,detection_rate,
                       data_days,refit)
    
    '''
    
    #get data and initial values
    model.initialize()    
    model.define_parameters()
    
    #get model parameters
    model.update_parameters()
    print("beta,Sd,Sd_delay,n_ramp: %s,%s,%s,%s" %(model.params.beta,model.params.Sd,model.params.Sd_delay,model.params.n_ramp))

    #run model
    model.simulate()

    #error on data
    err=model.calculate_error()
    print("Data difference: %s" %err)
    
    #plot results
    #comp_plot_path = display.plot_comparison_SIR(model)
    comp_plot_path = display.plot_comparison_SEAIQHRD(model)
    
    return comp_plot_path
