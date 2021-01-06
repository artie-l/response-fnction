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

c = 299792458; epsilon0=8.85418781762039*1E-12; m=9.10938356*1E-31; e=1.60217662*1E-19; Na=6.02214076*1E23; w0=1E16; hbar=5.29E-12
 
dlatticeZnTe=6.1E-10; dlatticeGaP=5.4495E-10

lambda0 = np.arange(650E-9,950.01E-9,0.01E-9); lamcenter = 790E-9; lammin=700.8E-9; lammax=950.2E-9

omega0 = c/lambda0*2*np.pi; omegacenter = c*2*np.pi/lamcenter; omegamin = c*2*np.pi/lammax; omegamax = c*2*np.pi/lammin

FWHM = 10*10**-15; dw = 2*np.sqrt(2*np.log(2))/FWHM 

crystal = 'ZnTe'

d=7E-6

Frequency = np.linspace(0.01E12,60E12,1000); Omega = 2*np.pi*Frequency[:, np.newaxis]


def Ncalc():
    if crystal == 'ZnTe':
        return 6.34*Na/192.99*1E6
    elif crystal == 'GaP':
        return 4.138*Na/100.697*1E6

N=Ncalc()

def omegatowavelength(omega):
    return 2*np.pi*c/omega

def nOpticCalc(wavelength):
    if crystal == 'ZnTe':
        return np.sqrt(9.92+0.42530/((wavelength*10**6)**2-0.37766**2)
                    +8414.13/((wavelength*10**6)**2-56.5**2)) #https://refractiveindex.info/?shelf=main&book=ZnTe&page=Li
        # return np.sqrt(4.27+3.01*(wavelength*10**6)**2/((wavelength*10**6)**2-0.142))
    elif crystal == 'GaP':
        # return np.sqrt(4.1705+4.9113/(1-(0.1174/(wavelength*10**6)**2))
        #             +1.9928/(1-(756.46/(wavelength*10**6)**2))) #https://sci-hub.tw/10.1063/1.372050
        return np.sqrt(2.680+6.40*(wavelength*10**6)**2/((wavelength*10**6)**2-0.0903279))
    
def ngroupNIRcalc(omega):
    n=nOpticCalc(omegatowavelength(omega0))
    dndl = np.zeros(n.shape,np.float)
    dndl[0:-1]=np.diff(n)/np.diff(lambda0)
    dndl[-1] = (n[-1] - n[-2])/(lambda0[-1] - lambda0[-2])
    index=np.where(np.isclose(omega0 , omega))[0][0]
    return c/(c/n*(1-(lambda0/n)*dndl)**-1)[index]

def ngroupSimulation(omega): #obtained by fitting the grouprefractive index calculated "properly" (see function above)
    if crystal == 'ZnTe':
        return 1.40614+1.90542E-15*omega-9.05723E-31*omega**2+1.81359E-46*omega**3
    if crystal == 'GaP':
        return 2.6168+6.0452E-16*omega-2.44269E-31*omega**2+6.75494E-47*omega**3    

def nTHzcalc():
    """GaP epsiloninf=9.075, hw_to=367.3 cm-1, hw_lo=403.0 cm-1, gamma = 4.3 cm-1"""
    """ZnTe epsiloninf=6.7, hw_to=177 cm-1, hw_lo=206 cm-1, gamma = 3.01 cm-1"""
    if crystal == 'GaP':
        return np.sqrt(9.075*(1+(403**2-367.3**2)/(367.3**2-(hbar*Omega)**2-1j*hbar*Omega*4.3)))
        # return np.sqrt(8.7+(1.8*(10.98E12*2*np.pi)**2/((10.98E12*2*np.pi)**2-Omega**2-1j*(0.02E12*2*np.pi)*Omega))) 
    elif crystal == 'ZnTe':
        return np.sqrt(7.4+(2.7*(5.3E12*2*np.pi)**2/((5.3E12*2*np.pi)**2-Omega**2-1j*(0.09E12*2*np.pi)*Omega)))
        # return np.sqrt((1+(206**2-177**2)/(177**2-(hbar*Omega)**2-1j*hbar*Omega*3.01))*6.7)    
        
def PhaseMatchingCoeffCalc(omega):
    if crystal == 'ZnTe':    
        vgr=c/ngroupSimulation(omega)
        vph=c/nTHzcalc()
    elif crystal == 'GaP':
        vgr=c/ngroupSimulation(omega)
        vph=c/nTHzcalc()
    return (np.exp(1j*(vph**-1-vgr**-1)*Omega*d)-1)/((vph**-1-vgr**-1)*Omega)

def AutocorCalc(omega):
    return np.exp(-(1/dw**2)*(2*omega**2-2*omega*(2*omegacenter+Omega)+omegacenter**2+(omegacenter+Omega)**2))/(np.sqrt(2*np.pi)*dw*2)

def TransmissionCoeffCalc(omega):
    n=nOpticCalc(omegatowavelength(omega))
    nTHz=nOpticCalc(omegatowavelength(omega-Omega))
    return np.conj(2/(1+n))*np.abs(2*n/(1+n))**2*2/(1+nTHz)


epsilon=nTHzcalc()*nTHzcalc()


def CouplingTermCalc(omega):
    nOpt=nOpticCalc(omegatowavelength(omega))
    nOptsq=nOpt*nOpt
    chii=epsilon/epsilon0-nOptsq
    chie=nOptsq-1
    deltai=deltae=m*w0*w0*epsilon0*epsilon0/(N*N*e*e*e*dlatticeZnTe)
    return chie*chie*(deltae*chie+deltai*chii)*omega*omega*c/(nOpt*omega)*epsilon0

def r41THzcalc():
    """GaP epsiloninf=9.05, hw_to=367.3 cm-1, hw_lo=403.0 cm-1, gamma = 4.3 cm-1"""
    """ZnTe epsiloninf=6.7, hw_to=177 cm-1, hw_lo=206 cm-1, gamma = 3.01 cm-1"""
    if crystal == 'GaP':
        return 1E-12*(1-0.47*(1-((hbar*Omega)**2-1j*hbar*Omega*4.3)/(367.3)**2)**-1)
        # return 1E-12*(1+-0.53*(10.98E12*2*np.pi)**2/((10.98E12*2*np.pi)**2-Omega**2-1j*(0.02E12*2*np.pi)*Omega)) 
    elif crystal == 'ZnTe':
        # return 4.25E-12*(1-0.07*(1-((hbar*Omega)**2-1j*hbar*Omega*3.01)/(177)**2)**-1)
        return 4.25E-12*(1+-0.07*(5.3E12*2*np.pi)**2/((5.3E12*2*np.pi)**2-Omega**2-1j*(0.09E12*2*np.pi)*Omega)) 


# def Alltogethercalc(omega):
#     nOpt=nOpticCalc(omegatowavelength(omega))
#     return TransmissionCoeffCalc(omega)*AutocorCalc(omega)*PhaseMatchingCoeffCalc(omega)*r41THzcalc()*omega*c**2/(nOpt*omega)

offset_factor = 1000

def Alltogethercalc1(omega):
    return TransmissionCoeffCalc(omega)*AutocorCalc(omega)*PhaseMatchingCoeffCalc(omega)*CouplingTermCalc(omega) / offset_factor


def t12_THz():
    return np.transpose(2/(1+np.sqrt(epsilon)))[0]

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(11,11.5), dpi=236)

d=7E-6

Alltogether1, _ = quadpy.quad(Alltogethercalc1, omegamin, omegamax, epsabs=1E-1, epsrel=1E-1, limit=1000)
Alltogetherconj1=np.conj(Alltogether1*4*1j*t12_THz()/c**2)

Omega = Omega * - 1
Alltogether1, _ = quadpy.quad(Alltogethercalc1, omegamin, omegamax, epsabs=1E-1, epsrel=1E-1, limit=1000)
Alltogetherminus1=Alltogether1*4*1j*t12_THz()/c**2

Alltogether1 = Alltogetherconj1 + Alltogetherminus1
Omega = 2*np.pi*Frequency[:, np.newaxis]

ax1.plot(Frequency*1E-12, np.abs(Alltogether1)/2, 'g',  linewidth='2.5', linestyle=(0, (3, 1, 1, 1, 1, 1)))
ax2.plot(Frequency*1E-12, np.angle(Alltogether1) - 2, 'g',  linewidth='2.5', linestyle=(0, (3, 1, 1, 1, 1, 1)))

d=10E-6

Alltogether2, _ = quadpy.quad(Alltogethercalc1, omegamin, omegamax, epsabs=1E-1, epsrel=1E-1, limit=1000)
Alltogetherconj2=np.conj(Alltogether2*4*1j*t12_THz()/c**2)

Omega = Omega * - 1
Alltogether2, _ = quadpy.quad(Alltogethercalc1, omegamin, omegamax, epsabs=1E-1, epsrel=1E-1, limit=1000)
Alltogetherminus2=Alltogether2*4*1j*t12_THz()/c**2

Alltogether2 = Alltogetherconj2 + Alltogetherminus2
Omega = 2*np.pi*Frequency[:, np.newaxis]

ax1.plot(Frequency*1E-12, np.abs(Alltogether2)/2 - 1.5E-11, 'r',  linewidth='2.5')
ax2.plot(Frequency*1E-12, np.angle(Alltogether2) - 8, 'r',  linewidth='2.5')

d=30E-6

Alltogether3, _ = quadpy.quad(Alltogethercalc1, omegamin, omegamax, epsabs=1E-1, epsrel=1E-1, limit=1000)
Alltogetherconj3=np.conj(Alltogether3*4*1j*t12_THz()/c**2)

Omega = Omega * - 1
Alltogether3, _ = quadpy.quad(Alltogethercalc1, omegamin, omegamax, epsabs=1E-1, epsrel=1E-1, limit=1000)
Alltogetherminus3=Alltogether3*4*1j*t12_THz()/c**2

Alltogether3 = Alltogetherconj3 + Alltogetherminus3

ax1.plot(Frequency*1E-12, np.abs(Alltogether3)/2 - 3E-11, 'b',  linewidth='2.5', linestyle=(0, (5, 1)))
ax2.plot(Frequency*1E-12, np.angle(Alltogether3) - 14, 'b',  linewidth='2.5', linestyle=(0, (5, 1)))

ax1.hlines([-3E-11,-1.5E-11,0],0,60,linestyles='dashed')

props = dict(boxstyle='square', facecolor='None')
ax1.text(55,2.3125E-11,'(a)', fontsize=23.5, ha='center')
ax2.text(55,1.3,'(b)', fontsize=23.5, ha='center')
ax1.text(30,2.2E-11,'ZnTe', fontsize=23.5, bbox=props, ha='center')
ax1.text(40,1E-11,'d = 7$\mu m$', fontsize=19.5, ha='left')
ax1.text(40,-0.75E-11,'d = 10$\mu m$', fontsize=19.5, ha='left')
ax1.text(40,-2.25E-11,'d = 30$\mu m$', fontsize=19.5, ha='left')


ax1.set_ylim(-3.2E-11,2.5E-11)
ax2.set_ylim(-18,2)
ax1.set_xlim(0,60)
ax1.set_yticks([])
ax2.set_yticks([])

ax1.set_xlabel('Frequency, THz', fontsize=20.5)
ax2.set_xlabel('Frequency, THz', fontsize=20.5)
ax1.set_ylabel('|$h_{EOS}(\omega, \Omega)$|, Arb. Units', fontsize=21.5)
ax2.set_ylabel('arg$h_{EOS}(\omega, \Omega)$, Arb. Units', fontsize=21.5 )
ax2.yaxis.set_label_position("right")
