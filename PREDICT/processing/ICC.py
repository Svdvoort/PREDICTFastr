import numpy as np
from scipy.linalg import pinv


def ICC(M, ICCtype='inter'):
    '''
    Input:
        M is matrix of observations. Rows: patients, columns: observers.
        type: ICC type, currently "inter" or "intra".
    '''

    n, k = M.shape

    SStotal = np.var(M, ddof=1) * (n*k - 1)
    MSR = np.var(np.mean(M, 1), ddof=1) * k
    MSW = np.sum(np.var(M, 1, ddof=1)) / n
    MSC = np.var(np.mean(M, 0), ddof=1) * n
    MSE = (SStotal - MSR * (n - 1) - MSC * (k -1)) / ((n - 1) * (k - 1))

    if ICCtype == 'intra':
        r = (MSR - MSW) / (MSR + (k-1)*MSW)
    elif ICCtype == 'inter':
        r = (MSR - MSE) / (MSR + (k-1)*MSE + k*(MSC-MSE)/n)
    else:
        raise ValueError('No valid ICC type given.')

    return r


def ICC_anova(Y, ICCtype='inter', more=False):
    '''
    Adopted from Nipype with a slight alteration to distinguish inter and intra.
    the data Y are entered as a 'table' ie subjects are in rows and repeated
    measures in columns
    One Sample Repeated measure ANOVA
    Y = XB + E with X = [FaTor / Subjects]
    '''

    [nb_subjects, nb_conditions] = Y.shape
    dfc = nb_conditions - 1
    dfe = (nb_subjects - 1) * dfc
    dfr = nb_subjects - 1

    # Compute the repeated measure effect
    # ------------------------------------

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(nb_conditions), np.ones((nb_subjects, 1)))  # sessions
    x0 = np.tile(np.eye(nb_subjects), (nb_conditions, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(np.dot(np.dot(X, pinv(np.dot(X.T, X))), X.T), Y.flatten('F'))
    residuals = Y.flatten('F') - predicted_Y
    SSE = (residuals ** 2).sum()

    residuals.shape = Y.shape

    MSE = SSE / dfe

    # Sum square session effect - between colums/sessions
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * nb_subjects
    MSC = SSC / dfc / nb_subjects

    session_effect_F = MSC / MSE

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    # ICC(3,1) = (mean square subjeT - mean square error) / (mean square subjeT + (k-1)*-mean square error)
    if ICCtype == 'intra':
        ICC = (MSR - MSE) / (MSR + dfc*MSE)
    elif ICCtype == 'inter':
        ICC = (MSR - MSE) / (MSR + dfc*MSE + nb_conditions*(MSC-MSE)/nb_subjects)
    else:
        raise ValueError('No valid ICC type given.')

    e_var = MSE  # variance of error
    r_var = (MSR - MSE) / nb_conditions  # variance between subjects

    if more:
        return ICC, r_var, e_var, session_effect_F, dfc, dfe
    else:
        return ICC
