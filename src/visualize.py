#%% Inital
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from classes import *

import seaborn as sns
sns.set(style="whitegrid", palette="deep", color_codes=True)

# Options
save = True
show = False

# Colors
Scol = "#07D5E6"
Ecol = "#FFD700"
Icol = "#FB9214"
Rcol = "#3B7548"
Dcol = "#EA1313"

#%% SIR
print("Simulating SIR")
np.random.seed(1)

beta_list = [2,5,10,20]

recovery_process = ss.erlang(a = 5)

fig = plt.subplots(nrows=2, ncols=2,figsize = (14,8), sharex=True, sharey=True)
for i, beta in enumerate(beta_list):
    es = SIR(beta, recovery_process, population=100_000, init_exposed=10)
    history = [(time, S, I, R) for __, time, (S, I, R) in es]
    time, S, I, R = zip(*history)
    
    plt.sca(fig[0].axes[i])
    plt.plot(time, S, label = "Susceptible", color = Scol)
    plt.plot(time, I, label = "Infected", color = Icol)
    plt.plot(time, R, label = "Recovered", color = Rcol)
    
    plt.legend(loc = 'center right')
    plt.xlabel("Days", fontsize = 14)
    plt.ylabel("Individuals", fontsize = 14)
    plt.title(r'SIR-model with $\beta$={}'.format(beta), fontsize = 16)

plt.tight_layout()

if save:
    plt.savefig('../reports/figures/SIR-model.pdf', format='pdf')

if show: 
    plt.show()



#%% SIRS
print("Simulating SIRS")
np.random.seed(2)

beta_list = [1,3]
mutation_list = [8,30]

recovery_process = ss.erlang(a = 5)
mutation_process = ss.expon(scale=8)

fig = plt.subplots(nrows=2, ncols=2,figsize = (14,8), sharex=True, sharey=True)
i = 0
for beta in beta_list:
      for m in mutation_list:
        mutation_process = ss.expon(scale=m)
        es = SIRS(beta, recovery_process, mutation_process, population=100_000, init_exposed=10)
        history = [(time, S, I, R) for __, time, (S, I, R) in es.run_until(60) ]
        time, S, I, R = zip(*history)

        plt.sca(fig[0].axes[i])
        plt.plot(time, S, label = "Susceptible", color = Scol)
        plt.plot(time, I, label = "Infected", color = Icol)
        plt.plot(time, R, label = "Recovered", color = Rcol)

        plt.legend()
        plt.xlabel("Days", fontsize = 14)
        plt.ylabel("Individuals", fontsize = 14)
        plt.title(r'SIRS-model with $\beta$={} and m={}'.format(beta,m), fontsize = 16)

        i += 1
    
plt.tight_layout()    

if save:
    plt.savefig('../reports/figures/SIRS-model.pdf', format='pdf')

if show:
    plt.show()


#%% SIRD
print("Simulating SIRD")
np.random.seed(3)

beta_list = [4, 10]
prob_dead_list = [0.05,0.4]

recovery_process = ss.erlang(a = 5)
death_process = ss.erlang( a = 7)

fig = plt.subplots(nrows=2, ncols=2,figsize = (14,8), sharex=True, sharey=True)
i = 0
for beta in beta_list:
    for prob in prob_dead_list:
        es = SIRD(beta, recovery_process, death_process, population=100_000, init_exposed=10, prob_dead=prob)

        history = [(time, S, I, R, D) for __, time, (S, I, R, D) in es ]
        time, S, I, R, D = zip(*history)

        plt.sca(fig[0].axes[i])
        plt.plot(time, S, label = "Susceptible", color = Scol)
        plt.plot(time, I, label = "Infected", color = Icol)
        plt.plot(time, R, label = "Recovered", color = Rcol)
        plt.plot(time, D, label = "Deceased", color = Dcol)
        
        plt.legend()
        plt.xlabel("Days", fontsize = 14)
        plt.ylabel("Individuals", fontsize = 14)
        plt.title(r'SIRD-model with $\beta$={} and prob_dead={}'.format(beta,prob), fontsize = 16)
        
        i += 1
        

plt.tight_layout()  

if save: 
    plt.savefig('../reports/figures/SIRD-model.pdf', format='pdf')

if show:
    plt.show()



#%% SR-SIR
print("Simulating SR_SIR")
np.random.seed(4)

beta_list = [1, 4]
vaccine_start_list = [1, 4]


recovery_process = ss.erlang(a = 3)
fig = plt.subplots(nrows=2, ncols=2,figsize = (14, 8), sharex=True, sharey=True)
i = 0
for beta in beta_list:
    for vaccine_start in vaccine_start_list:
        es = SR_SIR(
            beta,
            recovery_process, 
            population=100_000, 
            init_exposed=10, 
            begin_vaccine=vaccine_start,
            vaccine_rate=lambda t: 10+1000*t
        )

        history = [(event, time, S, I, R) for event, time, (S, I, R) in es ]
        event, time, S, I, R = zip(*history)
        
        plt.sca(fig[0].axes[i])
        plt.plot(time, S, label = "Susceptible", color = Scol)
        plt.plot(time, I, label = "Infected", color = Icol)
        plt.plot(time, R, label = "Recovered", color = Rcol)
        plt.axvline(x=vaccine_start, c = "k", ls = "--", label = "Vaccine found")
        
        plt.legend(loc = 'center right')
        plt.xlabel("Days", fontsize = 14)
        plt.ylabel("Individuals", fontsize = 14)
        plt.title(r'SRSIR-model with $\beta$={} and vaccine_start={}'.format(beta,vaccine_start), fontsize = 16)
        
        i += 1

plt.tight_layout()  

if save:
    plt.savefig('../reports/figures/SRSIR-model.pdf', format='pdf')

if show:
    plt.show()



#%% SEIR
print("Simulating SEIR")
np.random.seed(5)

recovery_process = ss.erlang(a = 3)

beta_list = [2, 6]
loc_list = [2, 7]


fig = plt.subplots(nrows=2, ncols=2,figsize = (14,8), sharex=True, sharey=True)
i = 0
for beta in beta_list:
    for loc in loc_list:
        incubation_process = ss.beta(a=2,b=2,loc=loc,scale=3)
        
        es = SEIR(
            beta,
            incubation_process,
            recovery_process, 
            population=100_000, 
            init_exposed=3, 
        )

        history = [(event, time, S, E, I, R) for event, time, (S, E, I, R) in es ]
        event, time, S, E, I, R = zip(*history)

        plt.sca(fig[0].axes[i])
        plt.plot(time, S, label = "Susceptible", color = Scol)
        plt.plot(time, E, label = "Exposed", color = Ecol)
        plt.plot(time, I, label = "Infected", color = Icol)
        plt.plot(time, R, label = "Recovered", color = Rcol)

        plt.legend(loc = 'center right')
        plt.xlabel("Days", fontsize = 14)
        plt.ylabel("Individuals", fontsize = 14)
        plt.title(r'SEIR-model with $\beta$={} and expected incubation={}'.format(beta, 2/(2+2)+loc), fontsize = 16)

        i += 1
    
plt.tight_layout() 

if save:
    plt.savefig('../reports/figures/SEIR-model.pdf', format='pdf')

if show:
    plt.show()


#%% SR-SEIRSD
print("Simulating SR_SEIRSD")
np.random.seed(6)
incubation_process = ss.beta(a=2,b=2,loc=4,scale=3)
recovery_process = ss.erlang(5)
death_process = ss.erlang(7)
mutation_process = ss.expon(30)
        
es = SR_SEIRSD(
    1,
    incubation_process,
    recovery_process,
    death_process,
    mutation_process,
    population=100_000, 
    init_exposed=5,
    begin_vaccine=100,
    vaccine_rate=lambda t: np.min([2500,250*t]),
    prob_dead=0.05
)


history = [(time, S, E, I, R, D) for __, time, (S, E, I, R, D) in es.run_until(150)]
time, S, E, I, R, D = zip(*history)

fig = plt.figure(figsize = (14, 8))
plt.plot(time, S, label = "Susceptible", color = Scol)
plt.plot(time, E, label = "Exposed", color = Ecol)
plt.plot(time, I, label = "Infected", color = Icol)
plt.plot(time, R, label = "Recovered", color = Rcol)
plt.plot(time, D, label = "Deceased", color = Dcol)
plt.axvline(x=100, c = "k", ls = "--", label = "Vaccine found")
plt.legend(loc = 'center right')
plt.title("Example of the complete SR_SEIRSD model", fontsize = 16)
plt.xlabel("Days", fontsize = 14)
plt.ylabel("Individuals", fontsize = 14)

if save:
    plt.savefig('../reports/figures/SR_SEIRSD-model.pdf', format='pdf')

if show:
    plt.show()


print("Done!")