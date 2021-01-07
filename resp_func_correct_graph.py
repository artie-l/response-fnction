# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 23:36:32 2021

@author: Artem
"""

import numpy as np
import matplotlib.pyplot as plt
import quadpy
plt.rcParams.update({'font.size': 14})
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from scipy.interpolate import interp1d

c = 299792458
hbar=5.29E-12
epsilon0=8.85418781762039 * 1E-12


def omegatowavelength(omega):
    return 2*np.pi*c/omega

def nOpticCalc(wavelength):
    lam_micron = wavelength * 1E6 # conversion to micrometers
    if crystal == 'ZnTe':
        return np.sqrt(4.27 + 3.01 * lam_micron ** 2 / (lam_micron ** 2 - 0.142)) # Marple 1964
        # return np.sqrt(9.92 + 0.42530/(lam_micron ** 2 - 0.37766 **2 )
                    # + 8414.13 / (lam_micron ** 2 - 56.5 ** 2)) #https://refractiveindex.info/?shelf=main&book=ZnTe&page=Li
    if crystal == 'GaP':
        return np.sqrt(2.68 + 6.40 * lam_micron ** 2 / (lam_micron ** 2 - 0.0903279)) #https://sci-hub.tw/https://aip.scitation.org/doi/10.1063/1.351012
        # return np.sqrt(4.1705 + 4.9113 / (1 - (0.1174 / lam_micron ** 2))
                       # + 1.9928 / (1 - (756.46 / lam_micron**2))) #https://sci-hub.tw/10.1063/1.372050
    
def ngroupSimulation(omega): #obtained by fitting the grouprefractive index calculated "properly" (see Refr_index_calc.py)
    if crystal == 'ZnTe':
        # return 1.40614+1.90542E-15*omega-9.05723E-31*omega**2+1.81359E-46*omega**3
        return (8.71045E-6 * np.exp(-omega/-3.23162E14) + 0.02678 * np.exp(-omega/-1.00538E15) + 0.03299 * np.exp(-omega/-1.0054E15) + 2.60492) 
    if crystal == 'GaP':
        return 0.03455  * np.exp(-omega/-1.23815E15) + 0.03581  * np.exp(-omega/-1.23816E15) + 0.03654 * np.exp(-omega/-1.23818E15) + 2.85176

def nTHzcalc():
    """GaP epsiloninf=9.075, hw_to=367.3 cm-1, hw_lo=403.0 cm-1, gamma = 4.3 cm-1"""
    """ZnTe epsiloninf=6.7, hw_to=177 cm-1, hw_lo=206 cm-1, gamma = 3.01 cm-1"""
    if crystal == 'GaP':
        return np.sqrt(9.075 * (1 + (403**2 - 367.3**2) / (367.3**2 - (hbar*Omega)**2 - 1j*hbar*Omega*4.3)))
        # return np.sqrt(8.7 + (1.8*(10.98E12*2*np.pi)**2/((10.98E12*2*np.pi)**2-Omega**2-1j*(0.02E12*2*np.pi)*Omega))) 
    elif crystal == 'ZnTe':
        # return np.sqrt(7.4+(2.7*(5.32E12*2*np.pi)**2/((5.32E12*2*np.pi)**2-Omega**2-1j*(0.09E12*2*np.pi)*Omega)))
        # return np.sqrt(7.44+(2.58*(5.32E12*2*np.pi)**2/((5.32E12*2*np.pi)**2-Omega**2-1j*(0.025E12*2*np.pi)*Omega)))
        return np.sqrt(6.7 * (1 + (206**2 - 177**2) / (177**2 - (hbar*Omega)**2 - 1j*hbar*Omega*3.01)))    
        
def r41THzcalc():
    """GaP epsiloninf=9.05, hw_to=367.3 cm-1, hw_lo=403.0 cm-1, gamma = 4.3 cm-1"""
    """ZnTe epsiloninf=6.7, hw_to=177 cm-1, hw_lo=206 cm-1, gamma = 3.01 cm-1"""
    if crystal == 'GaP':
        return 1E-12*(1 - 0.53*(1 - ((hbar*Omega)**2 - 1j*hbar*Omega*4.3) / 367.3**2)**-1)*1.7 #In the paper they say -0.53 +- 0.03
        # return 1E-12*(1 + -0.53*(10.98E12*2*np.pi)**2/((10.98E12*2*np.pi)**2-Omega**2-1j*(0.02E12*2*np.pi)*Omega)) * 4
    elif crystal == 'ZnTe':
        return 4E-12*(1 - 0.07*(1-((hbar*Omega)**2-1j*hbar*Omega*3.01) / 177**2)**-1)
        # return 4E-12*(1 + -0.07*(5.3E12*2*np.pi)**2/((5.3E12*2*np.pi)**2-Omega**2-1j*(0.09E12*2*np.pi)*Omega))
        
def PhaseMatchingCoeffCalc(omega):
    vgr=c/ngroupSimulation(omega)
    vph=c/nTHzcalc()
    return (np.exp(1j*(vph**-1-vgr**-1)*Omega*d)-1)/((vph**-1-vgr**-1)*Omega)

def AutocorCalc(omega):
    return np.exp(-(1/dw**2)*(2*omega**2-2*omega*(2*omegacenter+Omega)+omegacenter**2+(omegacenter+Omega)**2))/(np.sqrt(2*np.pi)*dw)
    
def TransmissionCoeffCalc(omega):
    n=nOpticCalc(omegatowavelength(omega))
    nOmega=nOpticCalc(omegatowavelength(omega-Omega))
    return np.conj(2/(1+n)) * 2/(1+nOmega) * np.abs(2*n / (1+n))**2

def Alltogethercalc(omega):
    n=nOpticCalc(omegatowavelength(omega))
    chi41 = r41THzcalc() * n**4 / 4 # there should be a minus!!!
    return TransmissionCoeffCalc(omega)*AutocorCalc(omega)*PhaseMatchingCoeffCalc(omega)*chi41*omega*omega*c/(n*omega)

def t12_THz():
    return np.transpose(2/(1+nTHzcalc()))[0]

def full_response_function(Omega):
    first_part, _ = quadpy.quad(Alltogethercalc, omegamin, omegamax, epsabs=5E-2, epsrel=5E-2, limit=1000)
    integr_coeff_1 = 4 * 1j * t12_THz() / c ** 2 
    Omega *= -1
    second_part, _ = quadpy.quad(Alltogethercalc, omegamin, omegamax, epsabs=5E-2, epsrel=5E-2, limit=1000)
    integr_coeff_2 = 4 * 1j * t12_THz() / c ** 2
    Omega *= -1
    full_resp_func = np.conj(first_part * integr_coeff_1) + second_part * integr_coeff_2
    return Frequency, full_resp_func

lamcenter = 790E-9 # Carrier wavelength, meters
FWHM = 10*10**-15 # Pulse duration, seconds

max_thz_freq = 60E12 # Hz. EOS Will be calculated up to this frequency. Should be < FFT sampling frequency

dw = 2*np.sqrt(2*np.log(2))/FWHM 
dl = np.sqrt(1/dw) * 2

lammin = lamcenter - dl
lammax = lamcenter + dl

lambda0 = np.arange(lammin, lammax + 0.01E-9, 0.01E-9)

omega0 = c * 2 * np.pi / lambda0
omegacenter = c * 2 * np.pi / lamcenter
omegamin = c * 2 * np.pi / lammax
omegamax = c * 2 * np.pi / lammin

Frequency = np.linspace(0.0001E12, max_thz_freq, 2000) # THz frequency range, THz
Omega = 2 * np.pi * Frequency[:, np.newaxis] # Angular THz frequency matrix

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(11,11.5), dpi=333)
offset_1 = 2.2*1E-9
offset_2 = 3

crystal = 'ZnTe' # Crystal type. GaP or ZnTe
d = 7E-6 # Crystal thickness, meters

freq, znte_7 = full_response_function(Omega) # Response function calculation

ax1.plot(Frequency*1E-12, np.abs(znte_7), 'g',  linewidth='2.5', linestyle=(0, (3, 1, 1, 1, 1, 1)))
ax2.plot(Frequency*1E-12, np.unwrap(np.angle(znte_7)), 'g',  linewidth='2.5', linestyle=(0, (3, 1, 1, 1, 1, 1)))


d = 10E-6 # Crystal thickness, meters

freq, znte_10 = full_response_function(Omega) # Response function calculation

ax1.plot(Frequency*1E-12, np.abs(znte_10) - offset_1, 'r',  linewidth='2.5')
ax2.plot(Frequency*1E-12, np.unwrap(np.angle(znte_10)) - offset_2, 'r',  linewidth='2.5')

d = 30E-6 # Crystal thickness, meters

freq, znte_30 = full_response_function(Omega) # Response function calculation

ax1.plot(Frequency*1E-12, np.abs(znte_30) - offset_1 * 2.3, 'b',  linewidth='2.5', linestyle=(0, (5, 1)))
ax2.plot(Frequency*1E-12, np.unwrap(np.angle(znte_30)) - offset_2 * 2, 'b',  linewidth='2.5', linestyle=(0, (5, 1)))

d = 300E-6 # Crystal thickness, meters

freq, znte_300 = full_response_function(Omega) # Response function calculation

ax1.plot(Frequency*1E-12, np.abs(znte_300) - offset_1 * 3.5, 'black',  linewidth='2.5', linestyle=(0, (5, 1)))
ax2.plot(Frequency*1E-12, np.unwrap(np.angle(znte_300)) - offset_2 * 3, 'black',  linewidth='2.5', linestyle=(0, (5, 1)))

crystal = 'GaP' # Crystal type. GaP or ZnTe
d = 250E-6 # Crystal thickness, meters

freq, gap_250 = full_response_function(Omega) # Response function calculation


ax1.plot(Frequency*1E-12, np.abs(gap_250) - offset_1 * 4.7, 'green',  linewidth='2.5', linestyle=(0, (5, 1)))
ax2.plot(Frequency*1E-12, np.angle(gap_250) - offset_2 * 4, 'green',  linewidth='2.5', linestyle=(0, (5, 1)))

ax1.hlines([-offset_1, -offset_1 * 2.3,-offset_1 * 3.5, offset_1 * 4.7, 0],0,60,linestyles='dashed')

props = dict(boxstyle='square', facecolor='None')
ax1.text(55,np.max(np.abs(znte_7))*0.95,'(c)', fontsize=23.5, ha='center')
ax2.text(55,-0.675,'(d)', fontsize=23.5, ha='center')
ax1.text(30,np.max(np.abs(znte_7))*0.9,'ZnTe', fontsize=23.5, bbox=props, ha='center')
ax1.text(30,np.max(np.abs(znte_7)) - offset_1 * 5.15 ,'GaP', fontsize=23.5, bbox=props, ha='center')
ax1.text(35,offset_1/1.85,'d = 7$\mu m$', fontsize=19.5, ha='left')
ax1.text(35,-offset_1/1.95,'d = 10$\mu m$', fontsize=19.5, ha='left')
ax1.text(35,-1.7*offset_1,'d = 30$\mu m$', fontsize=19.5, ha='left')
ax1.text(35,-2.95*offset_1,'d = 300$\mu m$', fontsize=19.5, ha='left')
ax1.text(35,-4.2*offset_1,'d = 250$\mu m$', fontsize=19.5, ha='left')


ax1.set_ylim(-1.05*1E-8,np.max(np.abs(znte_7))*1.1)
ax2.set_ylim(-14,-0.25)
ax1.set_xlim(0,60)
ax1.set_yticks([])
ax2.set_yticks([])

ax1.set_xlabel('Frequency, THz', fontsize=20.5)
ax2.set_xlabel('Frequency, THz', fontsize=20.5)
ax1.set_ylabel('|$h_{EOS}(\omega, \Omega)$|, Arb. Units', fontsize=21.5)
ax2.set_ylabel('arg$h_{EOS}(\omega, \Omega)$, Arb. Units', fontsize=21.5 )
ax2.yaxis.set_label_position("right")