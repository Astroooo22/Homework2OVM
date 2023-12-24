import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

###########IMPLIED VOLATILITY###############
#investigating FTSE 100 implied volatility as a function of exercise price (1A)
exercise_prices = [5125,5225,5325,5425,5525,5625,5725,5825]
option_prices = [485,415,350,290.5,236,189.5,149,115]
implied_vol = []
S_0 = 5420.3
r = 0.05
T = 4/12

for i in range(len(exercise_prices)):
    sigma_0 = np.sqrt(2*abs((np.log(S_0/exercise_prices[i])+r*T)/T))
    sigma = sigma_0
    n = 0
    while n < 10:
        d1 = (np.log(S_0/exercise_prices[i])+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = (np.log(S_0/exercise_prices[i])+(r-0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2)
        vega = S_0*np.sqrt(T)*1/np.sqrt(2*np.pi)*np.exp(-0.5*d1**2)
        C = S_0*Nd1-exercise_prices[i]*np.exp(-r*T)*Nd2
        sigma = sigma-(C-option_prices[i])/vega
        n += 1
    implied_vol.append(sigma)

plt.figure() 
plt.plot(exercise_prices,implied_vol,color='black')
plt.scatter(exercise_prices,implied_vol,color='black',marker='D')
plt.xlabel('Exercise price')
plt.ylabel('Implied volatility')
plt.axvline(S_0,linestyle='--',linewidth=1,color='black')
plt.title('FTSE 100, 22 August 2001 modified')
plt.savefig('1A.png')

#investigating paypal implied volatility as a function of exercise price (1B)
paypal_strike = [60,61,62,63,64,65,66,67]
paypal_prices = [3.30,2.37,1.54,0.94,0.54,0.30,0.16,0.09]
implied_vol_paypal = []
S_0 = 63.13
r = 0.05
T = 3/252
#calls of 19/12 with expiry 22/12

for i in range(len(paypal_strike)):
    sigma_0 = np.sqrt(2*abs((np.log(S_0/paypal_strike[i])+r*T)/T))
    sigma = sigma_0
    n = 0
    while n < 10:
        d1 = (np.log(S_0/paypal_strike[i])+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = (np.log(S_0/paypal_strike[i])+(r-0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2)
        vega = S_0*np.sqrt(T)*1/np.sqrt(2*np.pi)*np.exp(-0.5*d1**2)
        C = S_0*Nd1-paypal_strike[i]*np.exp(-r*T)*Nd2
        sigma = sigma-(C-paypal_prices[i])/vega
        n += 1
    implied_vol_paypal.append(sigma)

plt.figure()
plt.plot(paypal_strike,implied_vol_paypal,color='black')
plt.scatter(paypal_strike,implied_vol_paypal,color='black',marker='D')
plt.xlabel('Exercise price')
plt.ylabel('Implied volatility')
plt.axvline(S_0,linestyle='--',linewidth=1,color='black')
plt.title('PYPL, 19 December 2023')
plt.savefig('1B.png')

#investigating paypal implied volatility as a function of expiry time (1C)
T = np.linspace(1/252,10/252,10)

plt.figure()

j=0
for T in T:
    implied_vol_paypal = []
    for i in range(len(paypal_strike)):
        sigma_0 = np.sqrt(2*abs((np.log(S_0/paypal_strike[i])+r*T)/T))
        sigma = sigma_0
        n = 0
        while n < 10:
            d1 = (np.log(S_0/paypal_strike[i])+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
            d2 = (np.log(S_0/paypal_strike[i])+(r-0.5*sigma**2)*T)/(sigma*np.sqrt(T))
            Nd1 = norm.cdf(d1)
            Nd2 = norm.cdf(d2)
            vega = S_0*np.sqrt(T)*1/np.sqrt(2*np.pi)*np.exp(-0.5*d1**2)
            C = S_0*Nd1-paypal_strike[i]*np.exp(-r*T)*Nd2
            sigma = sigma-(C-paypal_prices[i])/vega
            n += 1
        implied_vol_paypal.append(sigma)
    plt.plot(paypal_strike,implied_vol_paypal,label='T = '+str(j+1)+' days')
    plt.scatter(paypal_strike,implied_vol_paypal,marker='D')
    j+=1
plt.legend(fontsize=6)
plt.xlabel('Exercise price')
plt.ylabel('Implied volatility')
plt.axvline(S_0,linestyle='--',linewidth=1,color='black')
plt.title('PYPL, 19 December 2023')
plt.savefig('1Ca.png')

E = 63
price = 0.94
implied_vol_paypal = []
t = np.linspace(1,10,10)
T = np.linspace(1/252,10/252,10)

for T in T:
    sigma_0 = np.sqrt(2*abs((np.log(S_0/E)+r*T)/T))
    sigma = sigma_0
    n = 0
    while n < 10:
        d1 = (np.log(S_0/E)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = (np.log(S_0/E)+(r-0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2)
        vega = S_0*np.sqrt(T)*1/np.sqrt(2*np.pi)*np.exp(-0.5*d1**2)
        C = S_0*Nd1-E*np.exp(-r*T)*Nd2
        sigma = sigma-(C-price)/vega
        n += 1
    implied_vol_paypal.append(sigma)

plt.figure()
plt.plot(t,implied_vol_paypal,color='black')
plt.scatter(t,implied_vol_paypal,color='black',marker='D')
plt.xlabel('Days to expiry')
plt.ylabel('Implied volatility')
plt.title('PYPL call option, 19 December 2023, E=63')
plt.savefig('1Cb.png')

#############BINOMIAL METHOD################
T = 3
n_points = 5000 #number of S points for European and American put options
deg = 7 #degree of fitted polynomial
r = 0.05
vol = 0.3
N = 3 #N-period binomial model for European and American options
N_bermudan = 72 #N-period binomial model for Bermudan options
n_bermudan = 1000 #number of S points for Bermudan option
E = 0.5 #Exercise price
p = 0.5
spacing = [2,6,12,24] #spacing of early-exercise dates for Bermudan option

S_lst0 = np.linspace(0.1,1.5,15) 
S_lst = np.linspace(0.1,1.5,n_points) 
Sb_lst = np.linspace(0.1,1.5,n_bermudan) 

def Payoff(S,E): #payoff function for put option
    if E > S:
        P = E - S
    else:
        P = 0
    return P

def binomial_model_european(T,S_lst,E,r,vol,N,p):
    dt = T/N
    u = np.exp(vol*np.sqrt(dt)+(r-0.5*vol**2)*dt)
    d = np.exp(-vol*np.sqrt(dt)+(r-0.5*vol**2)*dt)
    V_0 = []
    payoff_lst = []

    for S in S_lst0:  #points for payoff at expiry
        payoff = Payoff(S,E)
        payoff_lst.append(payoff)

    for S in S_lst: #points for time-zero option value
        n=0
        S_T = []
        S_new = []
        V_lst = []
        S_T.append(S)
        while n < N:
            for S in S_T:
                S_u = u*S
                S_d = d*S
                S_new.append(S_u)
                S_new.append(S_d)
            S_T = S_new
            S_new = []
            dup_lst = []
            for j in range(len(S_T)-1): #removing duplicate stock prices at each time-step
                if abs(S_T[j] - S_T[j+1]) < 0.01:
                    dup_lst.append(S_T[j])
            for j in dup_lst:
                S_T.remove(j)
            n+=1
        for S in S_T: #calculating for each S option value at expiry
            payoff = Payoff(S,E)
            V_lst.append(payoff) 

        V_new = []
        while len(V_lst) > 1: #discounting option value backwards in time
            for v in range(len(V_lst)-1):
                V = np.exp(-r*dt)*(p*V_lst[v]+(1-p)*V_lst[v+1])
                V_new.append(V)
            V_lst = V_new
            V_new = []


        V = V_lst[0]
        V_0.append(V)

    c = np.polyfit(S_lst,V_0,deg) #fitting polynomial to data points to obtain a smooth curve
    poly = np.poly1d(c)
    V_0 = poly(S_lst)

    plt.figure()
    plt.plot(S_lst,V_0,color='red',label='Option Value V_0')
    plt.plot(S_lst0,payoff_lst,color='black',label='Payoff at expiry')
    plt.title('European Put Option')
    plt.legend()
    plt.xlabel('S')
    plt.ylabel('V')
    plt.savefig('2A.png')

    return V_0

def binomial_model_american(T,S_lst,E,r,vol,N,p):
    dt = T/N
    u = np.exp(vol*np.sqrt(dt)+(r-0.5*vol**2)*dt)
    d = np.exp(-vol*np.sqrt(dt)+(r-0.5*vol**2)*dt)
    V_0 = []
    payoff_lst = []

    for S in S_lst0:
        payoff = Payoff(S,E)
        payoff_lst.append(payoff)

    for S in S_lst:
        n=0
        S_T = []
        S_new = []
        V_lst = []
        S_T.append(S)
        while n < N:
            for S in S_T:
                S_u = u*S
                S_d = d*S
                S_new.append(S_u)
                S_new.append(S_d)
            S_T = S_new
            S_new = []
            dup_lst = []
            for j in range(len(S_T)-1):
                if abs(S_T[j] - S_T[j+1]) < 0.01:
                    dup_lst.append(S_T[j])
            for j in dup_lst:
                S_T.remove(j)
            n+=1
        for S in S_T:
            payoff = Payoff(S,E)
            V_lst.append(payoff)
   
        V_new = []
        S_new = []

        while len(V_lst) > 1:
            for v in range(len(V_lst)-1):
                V = np.exp(-r*dt)*(p*V_lst[v]+(1-p)*V_lst[v+1])
                SS = S_T[v]/u #recovering expiry time stock prices for each time step backwards
                S_new.append(SS)
                SS = Payoff(SS,E) #calculating relevant payoff
                if V > SS: #payoff for American option
                    V_new.append(V)
                else:
                    V_new.append(SS)
            S_T = S_new
            S_new = []
            
            V_lst = V_new
            V_new = []

        V = V_lst[0]
        V_0.append(V)

    c = np.polyfit(S_lst,V_0,deg)
    poly = np.poly1d(c)
    V_0 = poly(S_lst)

    plt.figure()
    plt.plot(S_lst,V_0,color='red',label='Option Value V_0')
    plt.plot(S_lst0,payoff_lst,color='black',label='Payoff at expiry')
    plt.title('American Put Option')
    plt.legend()
    plt.xlabel('S')
    plt.ylabel('V')
    plt.savefig('2B.png')

    return V_0

def binomial_model_bermudan(T,S_lst,E,r,vol,N,p,spacing):
    plt.figure()
    dt = T/N
    u = np.exp(vol*np.sqrt(dt)+(r-0.5*vol**2)*dt)
    d = np.exp(-vol*np.sqrt(dt)+(r-0.5*vol**2)*dt)

    for s in spacing:
        V_0 = []
        payoff_lst = []

        for S in S_lst0:
            payoff = Payoff(S,E)
            payoff_lst.append(payoff)

        for S in S_lst:
            n=0
            S_T = []
            S_new = []
            V_lst = []
            S_T.append(S)
            while n < N:
                for S in S_T:
                    S_u = u*S
                    S_d = d*S
                    S_new.append(S_u)
                    S_new.append(S_d)
                S_T = S_new
                S_new = []
                dup_lst = []
                for j in range(len(S_T)-1):
                    if abs(S_T[j] - S_T[j+1]) < 0.0000001: #stock prices at expiry are much closer in value due to size of model, to remove actual duplicates difference must be much smaller.
                        dup_lst.append(S_T[j])
                for j in dup_lst:
                    S_T.remove(j)
                n+=1
            for S in S_T:
                payoff = Payoff(S,E)
                V_lst.append(payoff)
    
            V_new = []
            S_new = []

            while len(V_lst) > 1:
                for v in range(len(V_lst)-1):
                    V = np.exp(-r*dt)*(p*V_lst[v]+(1-p)*V_lst[v+1])
                    SS = S_T[v]/u
                    S_new.append(SS)
                    SS = Payoff(SS,E)
                    if (len(V_lst)-1)%s == 0: #determines whether we are at an early-exercise date, based on current time spacing of dates
                        if V > SS:
                            V_new.append(V)
                        else:
                            V_new.append(SS)
                    else:
                        V_new.append(V)
                S_T = S_new
                S_new = []
                
                V_lst = V_new
                V_new = []

            V = V_lst[0]
            V_0.append(V)

        c = np.polyfit(S_lst,V_0,deg)
        poly = np.poly1d(c)
        V_0 = poly(S_lst)

        plt.plot(S_lst,V_0,label='Option Value, '+str(N/s)+' early-exercise dates')
        print(f'spacing {s} done')
    plt.plot(S_lst0,payoff_lst,color='black',label='Payoff at expiry')
    plt.legend(fontsize=6)
    plt.title('Bermudan Put Option')
    plt.xlabel('S')
    plt.ylabel('V')
    plt.savefig('2C.png')
    return V_0

binomial_model_european(T,S_lst,E,r,vol,N,p)
binomial_model_american(T,S_lst,E,r,vol,N,p)
binomial_model_bermudan(T,Sb_lst,E,r,vol,N_bermudan,p,spacing) 

    
################MONTE CARLO################### Similar to Higham's matlab code, converted to python and slightly modified for option delta.
np.random.seed(100) 

S = 1
E = 1
sigma = 0.3
r = 0.05
h = 0.01
T = 1
nr_samples = [250,1000,2500,5000,7500,10000,12500,15000,17500,20000]
aM_lst = []
bM_lst = []
aManti_lst = []
bManti_lst = []
conf_lst = []
confanti_lst = []

for M in nr_samples:
    Dt = T/M
    N = int(T / Dt)
    V = np.zeros(M)
    Vh = np.zeros(M)
    Vhanti = np.zeros(M)
    Svals = np.zeros(M)
    delta1 = np.zeros(M)
    Vanti = np.zeros(M)
    delta2 = np.zeros(M)

    for i in range(M-1):
        samples = np.random.randn(N)
        Svals = S * np.cumprod(np.exp((r - 0.5 * sigma**2) * Dt + sigma * np.sqrt(Dt) * samples))
        Svalsh = (S + h) * np.cumprod(np.exp((r - 0.5 * sigma**2) * Dt + sigma * np.sqrt(Dt) * samples))
        Smax = np.max(Svals)
        Smaxh = np.max(Svalsh)
        V[i] = max(Smax - E, 0)
        Vh[i] = max(Smaxh - E, 0)

        Svals2 = S * np.cumprod(np.exp((r - 0.5 * sigma**2) * Dt - sigma * np.sqrt(Dt) * samples))
        Svalsh2 = (S + h) * np.cumprod(np.exp((r - 0.5 * sigma**2) * Dt - sigma * np.sqrt(Dt) * samples))
        Smax2 = np.max(Svals2)
        Smaxh2 = np.max(Svalsh2)
        V2 = max(Smax2 - E, 0)
        Vh2 = max(Smaxh2 - E, 0)
        Vanti[i] = 0.5 * (V[i] + V2)
        Vhanti[i] = 0.5 * (Vh[i] + Vh2)

    for i in range(M):
        delta1[i] = np.exp(-r * Dt) * (V[i] - Vh[i]) / h
        delta2[i] = np.exp(-r * Dt) * (Vanti[i] - Vhanti[i]) / h

    aM = np.mean(delta1)
    bM = np.std(delta1)

    aM_lst.append(aM)
    bM_lst.append(bM)

    conf = [aM - 1.96 * bM / np.sqrt(M), aM + 1.96 * bM / np.sqrt(M)]

    conf_lst.append(conf)

    aManti = np.mean(delta2)
    bManti = np.std(delta2)
    
    aManti_lst.append(aManti)
    bManti_lst.append(bManti)

    confanti = [aManti - 1.96 * bManti / np.sqrt(M), aManti + 1.96 * bManti / np.sqrt(M)]

    confanti_lst.append(confanti)

print(aM,conf,aManti,confanti)

plt.figure()
for aM in aM_lst:
    plt.errorbar(x=nr_samples[aM_lst.index(aM)], y=aM, yerr=np.array(abs(conf_lst[aM_lst.index(aM)]-aM)).reshape(2,1),ecolor='black')
    plt.scatter(nr_samples[aM_lst.index(aM)],aM,s=30,color='black',marker='x')
    plt.axhline(y=aM_lst[-1],linestyle='--',linewidth=1,color='black')
plt.xlabel('Number of Samples')
plt.ylabel('Delta approximation')
plt.title('Monte Carlo Simulation Results with Confidence Interval')
plt.savefig('3B.png')

################FINITE DIFFERENCE###################  
# L = np.pi
# Nx = 9
# dx = L / Nx
# T = 3
# Nt = 19
# dt = T / Nt
# nu = dt / dx**2
# sigma = 0.9

# # FTCS
# I = np.eye(Nx - 1)
# F = (1 - nu * sigma**2) * np.eye(Nx - 1) + 0.5 * nu * sigma**2 * np.diag(np.ones(Nx - 2), 1) + 0.5 * nu * sigma**2 * np.diag(np.ones(Nx - 2), -1)

# U = np.zeros((Nx - 1, Nt + 1))
# U[:, 0] = np.sin(np.arange(dx, L, dx))

# for i in range(Nt):
#     x = np.linalg.solve(I, F @ U[:, i])
#     U[:, i + 1] = x

# bc = np.zeros(Nt + 1)
# U = np.vstack([bc, U, bc])

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x_vals = np.linspace(0, L, Nx + 1)
# t_vals = np.linspace(0, T, Nt + 2)
# X, T = np.meshgrid(x_vals, t_vals[:-1])
# ax.plot_surface(X, T, U.T, cmap='viridis')
# ax.view_init(elev=20, azim=-150)

# ax.set_xlabel('x', fontsize=20)
# ax.set_ylabel('t', fontsize=20)
# ax.set_zlabel('U', fontsize=20)

# plt.title('FTCS')
# plt.savefig('FTCS.png')

# # BTCS
# B = (1 + nu * sigma**2) * np.eye(Nx - 1) - 0.5 * nu * sigma**2 * np.diag(np.ones(Nx - 2), 1) - 0.5 * nu * sigma**2 * np.diag(np.ones(Nx - 2), -1)

# U = np.zeros((Nx - 1, Nt + 1))

# U[:, 0] = np.sin(np.arange(dx, L, dx))

# for i in range(Nt):
#     x = np.linalg.solve(B, U[:, i])
#     U[:, i + 1] = x

# bc = np.zeros(Nt + 1)
# U = np.vstack([bc, U, bc])

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x_vals = np.linspace(0, L, Nx + 1)
# t_vals = np.linspace(0, T, Nt + 2)
# X, T = np.meshgrid(x_vals, t_vals[:-1])
# ax.plot_surface(X, T, U.T, cmap='viridis')
# ax.view_init(elev=20, azim=-150)

# ax.set_xlabel('x', fontsize=20)
# ax.set_ylabel('t', fontsize=20)
# ax.set_zlabel('U', fontsize=20)

# plt.title('BTCS')
# plt.savefig('BTCS.png')

# # Crank-Nicolson
# B = (1 + 0.5 * nu * sigma**2) * np.eye(Nx - 1) - 0.25 * nu * sigma**2 * np.diag(np.ones(Nx - 2), 1) - 0.25 * nu * sigma**2 * np.diag(np.ones(Nx - 2), -1)
# F = (1 - 0.5 * nu * sigma**2) * np.eye(Nx - 1) + 0.25 * nu * sigma**2 * np.diag(np.ones(Nx - 2), 1) + 0.25 * nu * sigma**2 * np.diag(np.ones(Nx - 2), -1)

# U = np.zeros((Nx - 1, Nt + 1))
# U[:, 0] = np.sin(np.arange(dx, L, dx))

# for i in range(Nt):
#     x = np.linalg.solve(B, F @ U[:, i])
#     U[:, i + 1] = x

# bc = np.zeros(Nt + 1)
# U = np.vstack([bc, U, bc])

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x_vals = np.linspace(0, L, Nx + 1)
# t_vals = np.linspace(0, T, Nt + 2)
# X, T = np.meshgrid(x_vals, t_vals[:-1])
# ax.plot_surface(X, T, U.T, cmap='viridis')
# ax.view_init(elev=20, azim=-150)

# ax.set_xlabel('x', fontsize=20)
# ax.set_ylabel('t', fontsize=20)
# ax.set_zlabel('U', fontsize=20)

# plt.title('Crank-Nicolson')
# plt.savefig('cranknicolson.png')