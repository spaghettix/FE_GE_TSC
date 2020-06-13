


__author__ = 'Stefano Mauceri'
__email__ = 'mauceri.stefano@gmail.com'



# =============================================================================
# IMPORT
# =============================================================================



import numpy as np
np.seterr(all='raise')
import pandas as pd
from math import ceil
from statsmodels.tsa.ar_model import AutoReg
from scipy.stats import kurtosis, linregress, skew



# =============================================================================
# TIME SERIES CLASSIFICATION
# =============================================================================



def extract(T, lb, ub, bool_, function):

    if ub is None:
        if bool_:
            return function(T[:, :lb])
        else:
            return function(T[:, lb:])
    else:
        lb, ub = sorted([lb, ub])
        return function(T[:, lb:ub])



def Above0(X):
    return (X > 0).sum(axis=1, keepdims=True)



def AbsSum(X):
    return np.sum(np.abs(X), axis=1, keepdims=True)



def ARCoeff(X):
    return np.apply_along_axis(lambda x:AutoReg(x, lags=1, seasonal=False, trend='n').fit().params, 1, X)



def AQ(x, y):
    return np.divide(x, np.sqrt(1 + np.power(y, 2)))



def Autocorrelation(X):
    try:
        return np.array([pd.Series(x).autocorr() for x in X]).reshape(-1,1)
    except:
        return np.zeros((X.shape[0], 1))



def Below0(X):
    return (X < 0).sum(axis=1, keepdims=True)



def CID(X):
    return np.sqrt(np.sum(np.power(np.diff(X, axis=1), 2), axis=1, keepdims=True))



def FFT(X):
    coef = np.fft.fft(X, axis=1)
    freqs = np.fft.fftfreq(X.shape[1])
    return freqs[np.argsort(np.abs(coef), axis=1)[:,-1:]]



def Kurtosis(X):
    return kurtosis(X, axis=1).reshape(-1,1)



def LinearTrend(X):
    time_axis = np.arange(X.shape[1])
    return np.array([linregress(time_axis, x)[0] for x in X]).reshape(-1,1)



def Max(X):
    return np.max(X, axis=1, keepdims=True)



def Mean(X):
    return np.mean(X, axis=1, keepdims=True)



def MeanAbsChanges(X):
    return np.mean(np.diff(X, axis=1), axis=1, keepdims=True)



def Median(X):
    return np.median(X, axis=1, keepdims=True)



def Min(X):
    return np.min(X, axis=1, keepdims=True)



def Skewness(X):
    return skew(X, axis=1).reshape(-1,1)



def STD(X):
    return np.std(X, axis=1, keepdims=True)



def _tras(x, lag=1):
    n = x.shape[0]
    if 2 * lag >= n:
        return 0
    else:
        return np.mean((np.roll(x, 2 * -lag) * np.roll(x, 2 * -lag) * np.roll(x, -lag) -
                        np.roll(x, -lag) * x * x)[0:(n - 2 * lag)])



def TRAS(X):
    return np.array([_tras(x) for x in X]).reshape(-1,1)



# =============================================================================
# OTHER FUNCTIONS
# =============================================================================



def return_one_percent(num, pop_size):
    """
    Returns either one percent of the population size or a given number,
    whichever is larger.

    :param num: A given number of individuals (NOT a desired percentage of
    the population).
    :param pop_size: A given population size.
    :return: either one percent of the population size or a given number,
    whichever is larger.
    """

    # Calculate one percent of the given population size.
    percent = int(round(pop_size/100))

    # Return the biggest number.
    if percent < num:
        return num
    else:
        return percent


def return_percent(num, pop_size):
    """
    Returns [num] percent of the population size.

    :param num: A desired percentage of the population.
    :param pop_size: A given population size.
    :return: [num] percent of the population size.
    """

    return int(round(num * pop_size / 100))


def aq(a, b):
    """aq is the analytic quotient, intended as a "better protected
    division", from: Ji Ni and Russ H. Drieberg and Peter I. Rockett,
    "The Use of an Analytic Quotient Operator in Genetic Programming",
    IEEE Transactions on Evolutionary Computation.

    :param a: np.array numerator
    :param b: np.array denominator
    :return: np.array analytic quotient, analogous to a / b.

    """
    return a / np.sqrt(1.0 + b**2.0)


def Pdiv(x, y):
    """
    Koza's protected division is:

    if y == 0:
      return 1
    else:
      return x / y

    but we want an eval-able expression. The following is eval-able:

    return 1 if y == 0 else x / y

    but if x and y are Numpy arrays, this creates a new Boolean
    array with value (y == 0). if doesn't work on a Boolean array.

    The equivalent for Numpy is a where statement, as below. However
    this always evaluates x / y before running np.where, so that
    will raise a 'divide' error (in Numpy's terminology), which we
    ignore using a context manager.

    In some instances, Numpy can raise a FloatingPointError. These are
    ignored with 'invalid = ignore'.

    :param x: numerator np.array
    :param y: denominator np.array
    :return: np.array of x / y, or 1 where y is 0.
    """
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(y == 0, np.ones_like(x), x / y)
    except ZeroDivisionError:
        # In this case we are trying to divide two constants, one of which is 0
        # Return a constant.
        return 1.0


def rlog(x):
    """
    Koza's protected log:
    if x == 0:
      return 1
    else:
      return log(abs(x))

    See pdiv above for explanation of this type of code.

    :param x: argument to log, np.array
    :return: np.array of log(x), or 1 where x is 0.
    """
    with np.errstate(divide='ignore'):
        return np.where(x == 0, np.ones_like(x), np.log(np.abs(x)))


def ppow(x, y):
    """pow(x, y) is undefined in the case where x negative and y
    non-integer. This takes abs(x) to avoid it.

    :param x: np.array, base
    :param y: np.array, exponent
    :return: np.array x**y, but protected

    """
    return np.abs(x)**y


def ppow2(x, y):
    """pow(x, y) is undefined in the case where x negative and y
    non-integer. This takes abs(x) to avoid it. But it preserves
    sign using sign(x).

    :param x: np.array, base
    :param y: np.array, exponent
    :return: np.array, x**y, but protected
    """
    return np.sign(x) * (np.abs(x) ** y)


def psqrt(x):
    """
    Protected square root operator

    :param x: np.array, argument to sqrt
    :return: np.array, sqrt(x) but protected.
    """
    return np.sqrt(np.abs(x))


def psqrt2(x):
    """
    Protected square root operator that preserves the sign of the original
    argument.

    :param x: np.array, argument to sqrt
    :return: np.array, sqrt(x) but protected, preserving sign.
    """
    return np.sign(x) * (np.sqrt(np.abs(x)))


def plog(x):
    """
    Protected log operator. Protects against the log of 0.

    :param x: np.array, argument to log
    :return: np.array of log(x), but protected
    """
    return np.log(1.0 + np.abs(x))


def ave(x):
    """
    Returns the average value of a list.

    :param x: a given list
    :return: the average of param x
    """

    return np.mean(x)


def percentile(sorted_list, p):
    """
    Returns the element corresponding to the p-th percentile
    in a sorted list

    :param sorted_list: The sorted list
    :param p: The percetile
    :return: The element corresponding to the percentile
    """

    return sorted_list[ceil(len(sorted_list) * p / 100) - 1]


def binary_phen_to_float(phen, n_codon, min_value, max_value):
    """
    This method converts a phenotype, defined by a
    string of bits in a list of float values

    :param phen: Phenotype defined by a bit string
    :param n_codon: Number of codons per gene, defined in the grammar
    :param min_value: Minimum value for a gene
    :param max_value: Maximum value for a gene
    :return: A list os float values, representing the chromosome
    """

    i, count, chromosome = 0, 0, []

    while i < len(phen):
        # Get the current gene from the phenotype string.
        gene = phen[i:(i + n_codon)]

        # Convert the bit string in gene to an float/int
        gene_i = int(gene, 2)
        gene_f = float(gene_i) / (2 ** n_codon - 1)

        # Define the variation for the gene
        delta = max_value[count] - min_value[count]

        # Append the float value to the chromosome list
        chromosome.append(gene_f * delta + min_value[count])

        # Increment the index and count.
        i = i + n_codon
        count += 1

    return chromosome


def ilog(n, base):
    """
    Find the integer log of n with respect to the base.

    >>> import math
    >>> for base in range(2, 16 + 1):
    ...     for n in range(1, 1000):
    ...         assert ilog(n, base) == int(math.log(n, base) + 1e-10), '%s %s' % (n, base)
    """
    count = 0
    while n >= base:
        count += 1
        n //= base
    return count


def sci_notation(n, prec=3):
    """
    Represent n in scientific notation, with the specified precision.

    >>> sci_notation(1234 * 10**1000)
    '1.234e+1003'
    >>> sci_notation(10**1000 // 2, prec=1)
    '5.0e+999'
    """
    base = 10
    exponent = ilog(n, base)
    mantissa = n / base**exponent
    return '{0:.{1}f}e{2:+d}'.format(mantissa, prec, exponent)


# =============================================================================
# END
# =============================================================================



if __name__ == '__main__':

    np.random.seed(123)

    X = np.random.randn(10, 100)

    Y = extract(X, 30, 80, True, lambda x: CID(x))
    print(Y.shape)

    Y = extract(X, 30, None, True, lambda x: Mean(x))
    print(Y.shape)


