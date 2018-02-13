from lifelines import CoxPHFitter
import pandas as pd


def fitcoxmodel(classification, T, E, pid, verbose=True):
    # Convert the inputs to PD dataframe
    data = dict()
    data['T'] = T
    data['E'] = E
    data['Cov'] = classification
    data = pd.DataFrame(data=data, index=pid)

    # Create the COX fitter
    cph = CoxPHFitter()
    cph.fit(data, duration_col='T', event_col='E')

    if verbose:
        cph.print_summary()

    # Retreive the coefficient
    s = cph.summary
    coef = s['coef']['Cov']
    CI = [s['lower 0.95']['Cov'], s['upper 0.95']['Cov']]
    p = s['p']['Cov']

    return coef, CI, p
