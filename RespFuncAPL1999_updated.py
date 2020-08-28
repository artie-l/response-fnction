import numpy as np
import matplotlib.pyplot as plt
import quadpy

plt.rcParams.update({'font.size': 14})
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


"""Constants"""
c = 299792458  # m/s, speed of light
hbar = 5.29E-12


"""Essential Parameters creation"""

d = 500E-6 # Crystal Thickness
crystal = 'ZnTe' # Crystal name. ZnTe or GaP are acceptable

Frequency = np.linspace(0.1E12, 10E12, 10000)[:, np.newaxis] # THz freq. range
Omega = 2 * np.pi * Frequency

lambda0 = np.arange(700E-9, 900.01E-9, 0.1E-9) # Sampling wavelength range
lamcenter = 800E-9
lammin = 740E-9
lammax = 840E-9
omegacenter = c * 2 * np.pi / lamcenter


def AutocorCalc(omega, dw):
    return (np.exp(-(1 / dw ** 2) * (2 * omega ** 2 - 2 * omega * (2 * omegacenter + Omega)
                                     + omegacenter ** 2 + (omegacenter + Omega) ** 2)) / (np.pi * dw ** 2)
            )


def nOpticCalc(wavelength):
    if crystal == 'ZnTe':
        return np.sqrt(9.92 + 0.42530 / ((wavelength * 10 ** 6) ** 2 - 0.37766 ** 2)
                       + 8414.13 / ((
                                            wavelength * 10 ** 6) ** 2 - 56.5 ** 2))
        # https://refractiveindex.info/?shelf=main&book=ZnTe&page=Li
        # return np.sqrt(4.27+3.01*(wavelength*10**6)**2/((wavelength*10**6)**2-0.142))
    elif crystal == 'GaP':
        # return np.sqrt(4.1705+4.9113/(1-(0.1174/(wavelength*10**6)**2))
        #             +1.9928/(1-(756.46/(wavelength*10**6)**2))) #https://sci-hub.tw/10.1063/1.372050
        return np.sqrt(2.680 + 6.40 * (wavelength * 10 ** 6) ** 2 / ((wavelength * 10 ** 6) ** 2 - 0.0903279))


def ngroupNIRcalc(lam):
    n = nOpticCalc(lambda0)
    dndl = np.zeros(n.shape, np.float)
    dndl[0:-1] = np.diff(n) / np.diff(lambda0)
    dndl[-1] = (n[-1] - n[-2]) / (lambda0[-1] - lambda0[-2])
    index = np.searchsorted(lambda0, lam)
    return c / (c / n * (1 - (lambda0 / n) * dndl) ** -1)[index]


def nTHzcalc():
    """GaP epsiloninf=9.075, hw_to=367.3 cm-1, hw_lo=403.0 cm-1, gamma = 4.3 cm-1"""
    """ZnTe epsiloninf=6.7, hw_to=177 cm-1, hw_lo=206 cm-1, gamma = 3.01 cm-1"""
    if crystal == 'GaP':
        return np.sqrt(
            9.075 * (1 + (403 ** 2 - 367.3 ** 2) / (367.3 ** 2 - (hbar * Omega) ** 2 - 1j * hbar * Omega * 4.3)))
        # return np.sqrt(8.7+(1.8*(10.98E12*2*np.pi)**2/((10.98E12*2*np.pi)**2-Omega**2-1j*(0.02E12*2*np.pi)*Omega)))
    elif crystal == 'ZnTe':
        # return np.sqrt(7.4+(2.7*(5.3E12*2*np.pi)**2/((5.3E12*2*np.pi)**2-Omega**2-1j*(0.09E12*2*np.pi)*Omega)))
        return np.sqrt(7.44 + (2.58 * (5.32E12 * 2 * np.pi) ** 2 / (
                (5.32E12 * 2 * np.pi) ** 2 - Omega ** 2 - 2j * (0.025E12 * 2 * np.pi) * Omega)))
        # return np.sqrt((1+(206**2-177**2)/(177**2-(hbar*Omega)**2-1j*hbar*Omega*3.01))*6.7)


def r41THzcalc():
    """GaP epsiloninf=9.05, hw_to=367.3 cm-1, hw_lo=403.0 cm-1, gamma = 4.3 cm-1"""
    """ZnTe epsiloninf=6.7, hw_to=177 cm-1, hw_lo=206 cm-1, gamma = 3.01 cm-1"""
    if crystal == 'GaP':
        return 1E-12 * (1 - 0.47 * (1 - ((hbar * Omega) ** 2 - 1j * hbar * Omega * 4.3) / (367.3) ** 2) ** -1)
        # return 1E-12*(1+-0.53*(10.98E12*2*np.pi)**2/((10.98E12*2*np.pi)**2-Omega**2-1j*(0.02E12*2*np.pi)*Omega))
    elif crystal == 'ZnTe':
        # return 4.25E-12*(1-0.07*(1-((hbar*Omega)**2-1j*hbar*Omega*3.01)/(177)**2)**-1)
        return 4.25E-12 * (1 + -0.07 * (5.3E12 * 2 * np.pi) ** 2 / (
                (5.3E12 * 2 * np.pi) ** 2 - Omega ** 2 - 1j * (0.09E12 * 2 * np.pi) * Omega))


def RespFuncThz(lam):
    ngr = ngroupNIRcalc(lam)
    if crystal == 'GaP' and lamcenter == 835E-9:
        ngr = 3.556
    return c * (np.exp(-1j * Omega * d * (ngr - nTHz) / c) - 1) / (-1j * d * Omega * (ngr - nTHz)) * (2 / (nTHz + 1))
    # return c*(np.exp(-1j*Omega*d*(ngr-nTHz)/c)-1)/(-1j*d*Omega*(ngr-nTHz))#*(2/(nTHz + 1))


"""Calculation of Geometric Response function"""

nTHz = nTHzcalc()
r41Thz = r41THzcalc()[:, 0]

GeomRespFuncCentralFreq = RespFuncThz(lamcenter)[:, 0]
GeomRespFuncEnvelope = np.mean(RespFuncThz((np.arange(lammin, lammax, 1E-9))), axis=1)
GeomRespFuncEnvelopeshort = np.mean(RespFuncThz((np.arange(797E-9, 803E-9, 1E-9))), axis=1)

"""Integration... Unnecessary in this case"""

# GeomRespFuncEnvelopeRaw, _ = quadpy.quad(RespFuncThz, lammin, lammax, epsabs=1E-1, epsrel=1E-1, limit=200)
# Normvalue=GeomRespFuncCentralFreq[0]/GeomRespFuncEnvelopeRaw[0]
# GeomRespFuncEnvelope=GeomRespFuncEnvelopeRaw*Normvalue

"""Calculation of Response function"""

RespFuncCentralFreq = GeomRespFuncCentralFreq * r41Thz
RespFuncEnvelope = GeomRespFuncEnvelope * r41Thz

"""Plotting graphs"""

"""Geometric response function"""
plt.figure(0, dpi=233)
plt.plot(Frequency * 1E-12, np.abs(GeomRespFuncCentralFreq), label='$\lambda_0= $' + np.str(lamcenter * 1E9) + ' nm',
         linewidth=2)
plt.plot(Frequency * 1E-12, np.abs(GeomRespFuncEnvelopeshort),
         label='$\Delta\lambda_s= $' + np.str(797) + ' to ' + np.str(803) + ' nm', linewidth=2)
plt.plot(Frequency * 1E-12, np.abs(GeomRespFuncEnvelope),
         label='$\Delta\lambda_s= $' + np.str(lammin * 1E9) + ' to ' + np.str(lammax * 1E9) + ' nm', linewidth=2)

plt.autoscale(enable=True, axis='x', tight=True)
plt.tight_layout()
plt.legend(prop={'size': 12})
plt.xlabel('Frequency $\Omega/2\pi$, THz')
plt.ylabel('$|h_{geom}(\omega,\Omega)|$, AU')
plt.title('Geometric response function of ' + str(d * 1E6) + ' $\mu$m ' + crystal, fontsize=11)

with open('Geom_Response_Function_' + crystal + '_' + str(int(d * 1E6)) + 'mum' + '.txt', 'wb') as f:
    np.savetxt(f, np.column_stack((Frequency, np.abs(GeomRespFuncCentralFreq))), fmt='%s', delimiter='\t')

"""Complex Refractive index"""

if 1:
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_dpi(233)
    fig.set_size_inches(12, 6)
    axes[0].plot(Frequency * 1E-12, np.real(nTHz), label='Real part of $n(\Omega)$, ' + crystal, linewidth=2.5)
    axes[0].set_ylabel('$Re(n(\Omega))$', fontsize=15)
    axes[1].plot(Frequency * 1E-12, np.imag(nTHz), 'r', label='Imag part of $n(\Omega)$, ' + crystal, linewidth=2.5)
    axes[1].set_ylabel('$Im(n(\Omega))$', fontsize=15)
    for ax in axes:
        ax.set_xlabel('Frequency $\Omega/2\pi$, THz', fontsize=16)
        ax.legend(prop={'size': 12})
        ax.autoscale(enable=True, axis='x', tight=True)

"""complex Electrooptic coefficient"""

if 1:
    plt.figure(2, dpi=233)
    plt.plot(Frequency * 1E-12, np.abs(r41Thz) * 1E12, label=np.str(lamcenter * 1E9) + ' nm')
    plt.legend(prop={'size': 12})
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.yscale('log')
    plt.xlabel('Frequency $\Omega/2\pi$, THz')
    plt.ylabel('|$r_{41}(\omega$)|, AU')
    plt.title('|$r_{41}(\omega$)| of ' + str(d * 1E6) + ' $\mu$m ' + crystal, fontsize=11)

"""Electrooptic response function"""

if 1:
    plt.figure(3, dpi=233)
    plt.plot(Frequency * 1E-12, np.abs(RespFuncCentralFreq) / np.abs(RespFuncCentralFreq[0]),
             label='$\lambda_0$ = ' + np.str(lamcenter * 1E9) + ' nm')
    plt.legend(prop={'size': 12})
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel('Frequency $\Omega/2\pi$, THz')
    plt.ylabel('$|h_{EOS}(\omega)|$, AU')
    plt.title('Response function of ' + str(d * 1E6) + ' $\mu$m ' + crystal, fontsize=11)

"""Boxcar pulse shape"""

if 1:
    plt.figure(4, dpi=233)
    x = np.linspace(760, 840, 1000)
    y = (780 < x) * (x < 820)  # + np.random.random(len(x))*.1
    plt.plot(x, y, linewidth=2.5)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel('Wavelength, nm')
    plt.ylabel('Spectrum, AU')

"""Geometric response function with pulse duaration"""
if 1:
    FWHM1 = 165 * 10 ** -15
    dw1 = np.sqrt(2 * np.pi) / FWHM1
    FWHM = 10 * 10 ** -15
    dw = np.sqrt(2 * np.pi) / FWHM
    Atc = AutocorCalc(omegacenter, dw)[:, 0]
    Atc = Atc / np.max(Atc)
    Atc1 = AutocorCalc(omegacenter, dw1)[:, 0]
    Atc1 = Atc1 / np.max(Atc1)
    plt.figure(5, dpi=233)
    plt.plot(Frequency * 1E-12, np.abs(GeomRespFuncCentralFreq), label='$\\tau_s$ = $1/\\infty$ fs', linewidth=2)
    plt.plot(Frequency * 1E-12, np.abs(GeomRespFuncCentralFreq) * Atc1, label='$\\tau_{s}$ = 165 fs', linewidth=2)
    plt.plot(Frequency * 1E-12, np.abs(GeomRespFuncCentralFreq) * Atc, label='$\\tau_{s}$ = 10 fs', linewidth=2)
    plt.yscale('log')

    plt.legend(prop={'size': 12})
    plt.xlabel('Frequency $\Omega/2\pi$, THz')
    plt.ylabel('$|h_{geom}(\omega,\Omega)|$, AU')
    plt.title('Geometric response function of ' + str(d * 1E6) + ' $\mu$m ' + crystal, fontsize=11)

    plt.autoscale(enable=True, axis='x', tight=True)
    plt.tight_layout()
