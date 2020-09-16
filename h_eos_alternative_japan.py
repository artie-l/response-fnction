# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 18:17:08 2020

@author: Artem
"""


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


c = 299792458
epsilon0=8.85418781762039*1E-12
hbar=5.29E-12


def r41THzcalc():
    """GaP epsiloninf=9.05, hw_to=367.3 cm-1, hw_lo=403.0 cm-1, gamma = 4.3 cm-1"""
    """ZnTe epsiloninf=6.7, hw_to=177 cm-1, hw_lo=206 cm-1, gamma = 3.01 cm-1"""
    if crystal == 'GaP':
        return 0.8E-12*(1 + -0.47*(1-((hbar*Omega)**2-1j*hbar*Omega*4.3)/(367.3)**2)**-1) 
        # return 1E-12*(1 + -0.53*(10.98E12*2*np.pi)**2/((10.98E12*2*np.pi)**2-Omega**2-1j*(0.02E12*2*np.pi)*Omega))
    elif crystal == 'ZnTe':
        return 4E-12*(1-0.07*(1-((hbar*Omega)**2-1j*hbar*Omega*3.01)/(177)**2)**-1)
        # return 4E-12*(1 + -0.07*(5.3E12*2*np.pi)**2/((5.3E12*2*np.pi)**2-Omega**2-1j*(0.09E12*2*np.pi)*Omega))
    
def nOpticCalc(wavelength):
    if crystal == 'ZnTe':
        return np.sqrt(9.92+0.42530/((wavelength*10**6)**2-0.37766**2)
                    +8414.13/((wavelength*10**6)**2-56.5**2)) #https://refractiveindex.info/?shelf=main&book=ZnTe&page=Li
        # return np.sqrt(4.27+3.01*(wavelength*10**6)**2/((wavelength*10**6)**2-0.142))
    elif crystal == 'GaP':
      #  return np.sqrt(4.1705+4.9113/(1-(0.1174/(wavelength*10**6)**2))
                      # +1.9928/(1-(756.46/(wavelength*10**6)**2))) #https://sci-hub.tw/10.1063/1.372050
        return np.sqrt(2.680+6.40*(wavelength*10**6)**2/((wavelength*10**6)**2-0.0903279)) 
    
def nTHzcalc():
    """GaP epsiloninf=9.075, hw_to=367.3 cm-1, hw_lo=403.0 cm-1, gamma = 4.3 cm-1"""
    """ZnTe epsiloninf=6.7, hw_to=177 cm-1, hw_lo=206 cm-1, gamma = 3.01 cm-1"""
    if crystal == 'GaP':
        return np.sqrt(9.075*(1+(403**2-367.3**2)/(367.3**2-(hbar*Omega)**2-1j*hbar*Omega*4.3)))
        # return np.sqrt(8.7+(1.8*(10.98E12*2*np.pi)**2/((10.98E12*2*np.pi)**2-Omega**2-1j*(0.02E12*2*np.pi)*Omega))) 
    elif crystal == 'ZnTe':
        return np.sqrt(7.4+(2.7*(5.32E12*2*np.pi)**2/((5.32E12*2*np.pi)**2-Omega**2-1j*(0.09E12*2*np.pi)*Omega)))
        # return np.sqrt(7.44+(2.58*(5.32E12*2*np.pi)**2/((5.32E12*2*np.pi)**2-Omega**2-1j*(0.025E12*2*np.pi)*Omega)))
        # return np.sqrt((1+(206**2-177**2)/(177**2-(hbar*Omega)**2-1j*hbar*Omega*3.01))*6.7)    

def ngroupSimulation(omega): #obtained by fitting the grouprefractive index calculated "properly" (see function above)
    if crystal == 'ZnTe':
        return 1.40614+1.90542E-15*omega-9.05723E-31*omega**2+1.81359E-46*omega**3
    if crystal == 'GaP':
        return 2.6168+6.0452E-16*omega-2.44269E-31*omega**2+6.75494E-47*omega**3    

def finite_pulse_effect():
    return np.exp(-(Omega*FWHM/4)**2)

def thz_wavenumber():
    return Omega*nTHzcalc()/c

def alpha_coeff_calc(thz_wavenumver):
    return np.real(thz_wavenumver) - Omega/(c/ngroupSimulation(omegacenter))

def loss_funtion():
    k_thz = thz_wavenumber()
    alpha = alpha_coeff_calc(np.real(k_thz))
    k_im = np.imag(k_thz)
    return 1/(np.sqrt(alpha**2 + k_im**2))*(np.sqrt(np.cos(alpha*d/2)**2*np.sinh(k_im*d/2)**2 + np.sin(alpha*d/2)**2*np.cosh(k_im*d/2)**2))


# Calculation

lamcenter = 805E-9 # Center wavelength, meters
FWHM = 165*10**-15 # pulse duration, seconds
max_thz_freq = 4.5E12 # Hz. Will be calculated up to this frequency. Should be > than FFT sampling frequency

crystal = 'ZnTe' # Crystal type. GaP or ZnTe
d = 500E-6 # Crystal thickness, meters

dw = 2*np.sqrt(2*np.log(2))/FWHM 
dl = np.sqrt(1/dw)

lammin = lamcenter - dl
lammax = lamcenter + dl

lambda0 = np.arange(lammin, lammax + 0.01E-9, 0.01E-9)

omega0 = c * 2 * np.pi / lambda0
omegacenter = c * 2 * np.pi / lamcenter
omegamin = c * 2 * np.pi / lammax
omegamax = c * 2 * np.pi / lammin

Frequency = np.linspace(0.0001E12, max_thz_freq, 1500) # THz frequency range, THz
Omega = 2 * np.pi * Frequency[:, np.newaxis]
    

crystal = 'ZnTe'
d = 500e-6

n=nOpticCalc(lamcenter)
chi41 = r41THzcalc() * n**4 / - 2 / epsilon0
h_znte = chi41*finite_pulse_effect()*loss_funtion()

crystal = 'GaP'
d = 200e-6

n=nOpticCalc(lamcenter)
chi41 = r41THzcalc() * n**4 / - 2 / epsilon0
h_GaP = chi41*finite_pulse_effect()*loss_funtion()

plt.figure(figsize=(7,5), dpi=236)
plt.plot(Frequency*1E-12, np.abs(h_znte)/np.abs(h_GaP))
plt.title("ZnTe GaP ratio")
plt.xlabel("Frequency, THz")
plt.ylabel("Ratio, Arb. Units")