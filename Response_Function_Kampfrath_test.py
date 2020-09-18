# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:54:00 2020

@author: artml
"""
import numpy as np
import matplotlib.pyplot as plt
import quadpy
plt.rcParams.update({'font.size': 14})
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from scipy.interpolate import interp1d

# constants

c = 299792458
hbar=5.29E-12
epsilon0=8.85418781762039*1E-12
 
# calc functions

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
        return np.sqrt(7.44+(2.58*(5.32E12*2*np.pi)**2/((5.32E12*2*np.pi)**2-Omega**2-1j*(0.025E12*2*np.pi)*Omega)))
        # return np.sqrt(6.7 * (1 + (206**2 - 177**2) / (177**2 - (hbar*Omega)**2 - 1j*hbar*Omega*3.01)))    
        
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

    # try to calculate in full

def AutocorCalc(omega):
    return np.exp(-(1/dw**2)*(2*omega**2-2*omega*(2*omegacenter+Omega)+omegacenter**2+(omegacenter+Omega)**2))/(np.pi*dw**2)
    
def TransmissionCoeffCalc(omega):
    n=nOpticCalc(omegatowavelength(omega))
    nOmega=nOpticCalc(omegatowavelength(omega-Omega))
    return np.conj(2/(1+n)) * 2/(1+nOmega) * np.abs(2*n / (1+n))**2

def Alltogethercalc(omega):
    n=nOpticCalc(omegatowavelength(omega))
    chi41 = r41THzcalc() * n**4 / - 2 / epsilon0
    return TransmissionCoeffCalc(omega)*AutocorCalc(omega)*PhaseMatchingCoeffCalc(omega)*chi41*omega*omega*c/(n*omega)

def t12_THz():
    return np.transpose(2/(1+nTHzcalc()))[0]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# def full_response_function(Omega):
#     first_part, _ = quadpy.quad(Alltogethercalc, omegamin, omegamax, epsabs=5E-2, epsrel=5E-2, limit=1000)
#     integr_coeff_1 = 4 * 1j * t12_THz() / c ** 2 
#     Omega *= -1
#     second_part, _ = quadpy.quad(Alltogethercalc, omegamin, omegamax, epsabs=5E-2, epsrel=5E-2, limit=1000)
#     integr_coeff_2 = 4 * 1j * t12_THz() / c ** 2
#     full_resp_func = np.conj(first_part * integr_coeff_1) + second_part * integr_coeff_2
#     two_sided_frf = np.append(full_resp_func, np.conj(np.flip(full_resp_func)))
#     # two_sided_frf = np.append(np.conj(first_part * integr_coeff_1), np.flip(second_part * integr_coeff_2))
#     two_sided_freq = np.append(Frequency, np.flip(Frequency*-1))
#     return two_sided_freq, two_sided_frf

def full_response_function(Omega):
    first_part, _ = quadpy.quad(Alltogethercalc, omegamin, omegamax, epsabs=5E-2, epsrel=5E-2, limit=1000)
    integr_coeff_1 = 4 * 1j * t12_THz() / c ** 2 
    Omega *= -1
    second_part, _ = quadpy.quad(Alltogethercalc, omegamin, omegamax, epsabs=5E-2, epsrel=5E-2, limit=1000)
    integr_coeff_2 = 4 * 1j * t12_THz() / c ** 2
    full_resp_func = np.conj(first_part * integr_coeff_1) + second_part * integr_coeff_2
    # two_sided_frf = np.append(full_resp_func, np.conj(np.flip(full_resp_func)))
    # two_sided_frf = np.append(np.conj(first_part * integr_coeff_1), np.flip(second_part * integr_coeff_2))
    # two_sided_freq = np.append(Frequency, np.flip(Frequency*-1))
    return Frequency, full_resp_func

# Calculation

lamcenter = 805E-9 # Carrier wavelength, meters
FWHM = 165*10**-15 # Pulse duration, seconds
max_thz_freq = 5E12 # Hz. Will be calculated up to this frequency. Should be > FFT sampling frequency
Tx=25E-3
G = 1 # arbitrary scaling factor
di_factor =  1 # EOS sensetivity improvement. It is noted in the lab notebook

crystal = 'GaP' # Crystal type. GaP or ZnTe
d = 200E-6 # Crystal thickness, meters

dw = 2*np.sqrt(2*np.log(2))/FWHM 
dl = np.sqrt(1/dw)

lammin = lamcenter - dl
lammax = lamcenter + dl

# lammin = 720E-9
# lammax = 860E-9

lambda0 = np.arange(lammin, lammax + 0.01E-9, 0.01E-9)

omega0 = c * 2 * np.pi / lambda0
omegacenter = c * 2 * np.pi / lamcenter
omegamin = c * 2 * np.pi / lammax
omegamax = c * 2 * np.pi / lammin

Frequency = np.linspace(0.0001E12, max_thz_freq, 1500) # THz frequency range, THz
Omega = 2 * np.pi * Frequency[:, np.newaxis]

freq, resp_func = full_response_function(Omega) # Response function calculation
resp_func = resp_func * G * di_factor / Tx # Accounting for the PD gain and lam/4 sens. improvement

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(12,8), dpi=236)
ax1.plot(freq[:len(freq)//2] * 1E-12, np.abs(resp_func[:len(freq)//2]), linewidth=3)
ax2.plot(freq[:len(freq)//2] * 1E-12, np.unwrap(np.angle(resp_func[:len(freq)//2])), linewidth=3)


fig.suptitle("{} $\mu m$ thick {}".format(d*1E6, crystal))
ax1.set_xlabel('Frequency, THz', fontsize=18)
ax2.set_xlabel('Frequency, THz', fontsize=18)
ax1.set_ylabel("$|h_{EOS}(\Omega)|$, m/V", fontsize=18)
ax2.set_ylabel("arg$(h_{EOS})$", fontsize=18)
ax2.yaxis.set_label_position("right")

ax1.autoscale()
ax2.autoscale()
# crystal = 'GaP' # Crystal type. GaP or ZnTe
# d = 200E-6 # Crystal thickness, microns
# freq, resp_func_GaP = full_response_function(Omega)
# plt.plot(freq*1E-12, np.abs(resp_func/resp_func_GaP))

with open('{} mum {} response function.txt'.format(d*1E6, crystal), 'w+') as file:
    np.savetxt(file, np.column_stack((freq, resp_func)), delimiter="\t")

# Import your data here

file_path = '03_06_2020_ZP70_mean_16_X.txt'
try:
    data=np.loadtxt(file_path, delimiter='\t', dtype=np.float64)
    time=data[:,0]
    signal=data[:,1]
    signal -= signal[0]
    signal /= Tx
    fft = np.fft.rfft(signal, norm='ortho') # FFT
    fftfreq = np.fft.rfftfreq(signal.size, d=time[1]-time[0]) # Freq. scale calculation
    
    resp_func_interpolation = interp1d(freq, resp_func, kind='zero', fill_value="extrapolate")
    final_resp_func = resp_func_interpolation(fftfreq * 1E12) # Getting response function in tact with measured data
    
    Cut_off_Frequency=find_nearest(fftfreq, 3) # filter out values above detection bandwidth
    
    final_resp_func[Cut_off_Frequency:] = final_resp_func[0]
    test = np.fft.irfft(fft/final_resp_func)
    reconstructed_signal = np.fft.irfft(fft/final_resp_func, n=len(time), norm='ortho')/(2*np.pi)
    plt.figure("reconstructed")
    plt.title('Reconstructed signal')
    plt.plot(time, signal/np.max(signal))
    plt.plot(time, reconstructed_signal/np.max(reconstructed_signal))
    plt.xlabel("Time delay, ps")
    plt.ylabel("THz field, V/m")
    with open('{} mum {} reconstructed.txt'.format(d*1E6, crystal), 'w+') as file:
        np.savetxt(file, np.column_stack((time, reconstructed_signal)), delimiter="\t")
except:
    print("Something wrong... Check the filepath")
    
    
# Gamma calc
n = nOpticCalc(omegatowavelength(omegacenter))
t_cryst = 2/(1+n)
dT_T = np.arcsin(np.max(np.abs(signal)))
E_thz = dT_T * lamcenter / (d * t_cryst * 1E-12 * 2 * np.pi * n**3) * 10**-5

print(np.max(np.abs(reconstructed_signal))*10**-5)



# dT_T = np.arcsin(0.44)
# E_thz_tanaka = dT_T * 780E-9 / (300E-6 * 0.46 * 0.88E-12 * 2 * np.pi * 3.2**3 * 0.7**6) * 10**-5