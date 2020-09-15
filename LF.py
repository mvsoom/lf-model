####################################################
###  Implementation of LF model -- see LF.ipynb  ###
####################################################
from numpy import *
from matplotlib.pyplot import *

from scipy.optimize import newton
from scipy.signal import find_peaks

εMIN = 0.001
εMAX = 150.
εN = 2000
αMIN = 0.001
αMAX = 3.
αN = 7500

def find_largest_root(f, xmin, xmax, N, visual=False):
    x = linspace(xmin, xmax, N)
    fabs = abs(f(x))

    i_x0 = x0 = None

    try:
        i_minima,_ = find_peaks(-fabs)
        i_x0 = max(i_minima)
        x0 = x[i_x0]
        x0 = newton(f, x0, tol=1.e-5, maxiter=1000) # Polishing
    finally:
        if visual: # Show as much as we can
            plot(x, fabs, label="|f(x)|")
            if i_x0: plot(x[i_x0], abs(f(x[i_x0])), 'xr', label="minima")
            if x0: plot(x0, abs(f(x0)), '*g', label="largest root")
            legend()
            show()
    
    return x0

def solve_ε(T0, Te, Ta, visual=False):
    Tb = T0-Te
    
    def f(ε): return ε*Ta-1+exp(-ε*Tb)
    
    return find_largest_root(f, εMIN, εMAX, εN, visual=visual)

def solve_α(Ee, T0, Te, Tp, Ta, ε, visual=False):
    ωg = pi/Tp
    Tb = T0-Te
    
    def E0(α): return -Ee*exp(-α*Te)/sin(ωg*Te)
    
    def Ao(α):
        f1 = E0(α)*exp(α*Te)/sqrt(ωg**2+α**2)
        f2 = sin(ωg*Te-arctan(ωg/α))
        t1 = E0(α)*ωg/(ωg**2+α**2)
        return f1*f2 + t1

    def Ar(α):
        f1 = -Ee/(ε**2*Ta)
        f2 = 1-exp(-ε*Tb)*(1+ε*Tb)
        return f1*f2
    
    def f(α):
        return Ao(α) + Ar(α)
    
    return find_largest_root(f, αMIN, αMAX, αN, visual=visual)

def Ug_prime(dt, Ee, T0, Te, Tp, Ta, visual=False):
    """Calculate U_g' by the LF model
    
    Implementation of the LF model is based on:
        Gobl, Christer. "Reshaping the Transformed LF Model: Generating the Glottal
            Source from the Waveshape Parameter Rd." INTERSPEECH. 2017.
        Doval, Boris, Christophe d'Alessandro, and Nathalie Henrich. "The spectrum
            of glottal flow models." Acta acustica united with acustica 92.6 (2006):
            1026-1046. (Note that our α is denoted by 'a' in this paper)

    The time parameters in (T0, Te, Tp, Ta) are given in the units of dt.
    If visual=True the function, denoted by f, of which the (ε,α) solutions are the roots
    are shown for diagnostic purposes.
    
    Troubleshooting:
        The LF model is quite unpleasant to handle numerically due to equations for (ε,α).
        
        In case the Newton-Raphson polishing does not converge, set visual=True and
        alter this module's (εMIN, εMAX, εN, αMIN, αMAX, αN)-values. The required number
        of points on the grid (i.e. εN and αN) for locating the minimum can be surprisingly
        high for some combinations values of (T0, Te, Tp, Ta).

            * For Ta ≃ 0.01 εMAX should be about 150, while for Ta >= 0.10 a value of
                εMAX=50 is enough.
            * Some combinations of the LF parameters give insoluble equations for α, e.g.
                small values of Tp (≃ 1). In general, Tp should be quite close to Te (which
                is realistic).
    
    Args:
        dt (float): Time step
        Ee, T0, Te, Tp, Ta (float): LF model parameters
        visual (bool): Show plots for (ε,α) calulcations, in that order

    Returns:
        t (array): Sample times of U_g' lying in [0, T0) and spaced apart by dt
        Ugp (array): U_g'
    """
    ε = solve_ε(T0, Te, Ta, visual=visual)
    assert ε > 0.
    
    α = solve_α(Ee, T0, Te, Tp, Ta, ε, visual=visual)
    assert α > 0.
    
    def sine(t): return Ug_prime_sine(t, Ee, T0, Te, Tp, α)
    def up(t): return Ug_prime_up(t, Ee, T0, Te, Ta, ε)
    
    t = arange(0., T0+dt, dt)
    Ugp = piecewise(t, [t <= Te, Te < t], [sine, up])
    return t, Ugp

def Ug_prime_sine(t, Ee, T0, Te, Tp, α):
    ωg = pi/Tp
    E0 = -Ee*exp(-α*Te)/sin(ωg*Te)
    sine = E0*exp(α*t)*sin(ωg*t)
    
    return sine

def Ug_prime_up(t, Ee, T0, Te, Ta, ε):
    Tb = T0-Te
    
    f1 = -Ee/(ε*Ta)
    f2 = exp(-ε*(t-Te))-exp(-ε*Tb)
    
    return f1*f2

###################################
#  LF model parameter utilities  #
##################################
def conv_R_param(T0, Rk, Rg, Fa):
    """Make sure T0 [time] and Fa [1/time] are in congruent units"""
    Ta = 1./(2*pi*Fa)
    Tp = T0/(2*Rg)
    Te = Tp*(1+Rk)
    return Te, Tp, Ta

def gen_param(s):
    """Generate a dict of typical LF model parameters for either s="male vowel" or
    s="female vowel". Note that the unit of time is (msec).
    
    Values taken from [1, p. 121].
    
      [1] Fant, Gunnar. "The LF-model revisited. Transformations and frequency domain analysis."
          Speech Trans. Lab. Q. Rep., Royal Inst. of Tech. Stockholm 2.3 (1995): 40.
    """
    if s=="male vowel":
        Ee = 1.
        T0 = 1000/120. # msec
        Rk = 0.30
        Rg = 1.20
        Fa = .700 # kHz
    elif s=="female vowel":
        Ee = 1.
        T0 = 1000/200. # msec
        Rk = 0.30
        Rg = 1.00
        Fa = .400 # kHz
    else: raise ValueError
    Te, Tp, Ta = conv_R_param(T0, Rk, Rg, Fa)
    return dict(Ee=Ee, T0=T0, Te=Te, Tp=Tp, Ta=Ta)

##############################################
#  Generate LF trains -- see LF-train.ipynb  #
##############################################
from warnings import warn

def max_length_in_P(Pt):
    return max([len(atleast_1d(v)) for v in Pt.values()])

def time_varying_P(Pt, n):
    for k, v in Pt.items():
        v = atleast_1d(v)
        v = pad(v, (0, n-len(v)), "edge")
        yield (k, v)

def LF_prime_train(dt, Pt, n=None, Tpad=0., full_output=False, diagnostic_plots=False):
    """Generate train of succesive GF' pulses à la LF
    
    The time parameters in Pt (T0, Te, Tp, Ta) and Tpad are given in the units of dt.
    The returned arrays (t, to, tp, te) are in units of dt.
    
    Troubleshooting:
        In case the glottal flow obtained by summing the train values starts to drift instead
        of steadily returning to zero, decrease dt.
    
        In case the numerical routine that solves for (ε,α) fails, warning is emitted with
        the current values of the LF parameters (Ee, T0, Te, Tp, Ta) (in that order). In addition,
        if diagnostic_plots=True, diagnostic plots are shown.
        
        See also the doc for Ug_prime().
    
    Args:
        dt (float): Time step
        Pt (dict or str): Specifies time-varying parameters of the LF model. If a string,
            Pt = gen_param(Pt).
        n (int): Number of pulses if this cannot be determined from Pt
        Tpad (float or (2,) array_like): Zero-pad the resulting train left and right
            by this amount. If a single number zero pad left and right by equal amounts
        full_output (bool): See "Returns"
        diagnostic_plots (bool): Show (ε,α) diagnostic plots in case of root solving error
    
    Returns:
        If full_output=False, only return:
        
            t (array): Train sample times spaced apart by dt
            train (array): Train samples
        
        If full_output=True, additionally return:
        
            to (array): Time instants of glottal opening
            tp (array): Time instants of maximum glottal flow (GF)
            te (array): Time instants of maximum excitation (GF')
            Pvary (dict): Dict containing the time-varying parameter values derived from Pt and n
                that were used to generate y
    """
    if isinstance(Pt, str): Pt = gen_param(Pt)
    
    n = max(n if n else 0, max_length_in_P(Pt))
    
    Pvary = dict(time_varying_P(Pt, n))
    
    Ugps = []

    for Ee, T0, Te, Tp, Ta in zip(Pvary["Ee"], Pvary["T0"], Pvary["Te"], Pvary["Tp"], Pvary["Ta"]):
        try:
            _, Ugp = Ug_prime(dt, Ee, T0, Te, Tp, Ta)
            Ugps.append(Ugp)
        except Exception as e:
            warn(f"Exception triggered by LF parameters {(Ee, T0, Te, Tp, Ta)}. Stop.")
            if diagnostic_plots: # Retrigger exception, now with plots
                try: LF.Ug_prime(dt, Ee, T0, Te, Tp, Ta, visual=True)
                except: pass
            raise
    
    train = hstack(Ugps)
    train = pad(train, (asarray(Tpad)/dt).astype(int), "constant")
    
    t = arange(len(train))*dt
    
    offset = atleast_1d(Tpad)[0]
    to = offset + hstack([array(0.), cumsum([len(Ugp) for Ugp in Ugps[:-1]])])*dt
    tp = to + Pvary["Tp"]
    te = to + Pvary["Te"]

    return (t, train) if not full_output else (t, train, to, tp, te, Pvary)