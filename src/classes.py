import collections
import heapq

import numpy as np
import pandas as pd
import scipy.stats as ss

class EpidemicModel(object):
    """
    Every of the epidemic models below inherits from this class, which defines
    the infrastructure of the event by event principle. The class supplies the
    exposure process dependent on the model states, and can be subclassed to
    use any number of stationary processes for defining events and state 
    changes. Events are defined as methods with an '_EVENT_' prefix. The base
    class supplies a basic SIR model without initialization, with events for
    exposure and recovery.
    """
    stationary_processes = 'recovery', 
    states = 'S', 'I', 'R'

    def __init__(self, beta, *process_rvs, population, init_exposed, **kwargs):

        for key, value in kwargs.items():
            setattr(self, f'{key}', value)

        for name, rv in zip(self.stationary_processes, process_rvs):
            setattr(self, f'_{name}_iter', self._rv_generator(rv))

        self.beta = beta
        self.population = population
        self.inv_population = 1/population
        self.init_exposed = init_exposed

        self.state_container = collections.namedtuple(
            'State', self.states
            )

        self.time = 0.0
        self.event_list = list()
        self._exposure_iter = self._rv_generator( ss.expon() )

        self._init_model()

    def __next__(self):

        try: 
            self.time, event = self._pop() 
        except IndexError:
            raise StopIteration

        method = getattr(self, event)
        method()
        self._insert_exposure_event()

        return event, self.time, self.state

    def __iter__(self):
        return self

    def run_until(self, t):
        """
        Returns a generator, iterating through events until some specified 
        time point.  
        """
        prev_time = 0
        it = iter(self)
        try:
            while True:
                event, time, state = next(it)
                if  time >= t:
                    break
                yield event, time, state
        except StopIteration:
            pass

    def _push(self, el):
        heapq.heappush( self.event_list, el )

    def _pop(self):
        return heapq.heappop(self.event_list)

    def _insert_exposure_event(self):
        """
        Inserts next exposure event is such an event take place before the 
        next event.
        """
        if self.I == 0 or self.S == 0:
            return

        dt = self.event_list[0][0] - self.time
        rate = self.beta * self.S * self.I * self.inv_population
        exposure_time = next(self._exposure_iter)/rate
        if exposure_time < dt:
            self._push( ( self.time + exposure_time, '_EVENT_exposure' ) )

    @property
    def state(self):
        return self.state_container(*[getattr(self, s) for s in self.states])

    def _rv_generator(self, rv):
        """
        Returns a generator of the scipy-stats random variable, where multiple
        numbers are generated at a time, since this is more efficient.
        """
        while True:
            yield from rv.rvs(size=20_000)

    def _EVENT_exposure(self):

        recovery_time = next( getattr(self, '_recovery_iter') )
        
        self.S -= 1
        self.I += 1

        self._push( (self.time + recovery_time, '_EVENT_recovery' ))
    
    def _EVENT_recovery(self):

        self.I -= 1
        self.R += 1

class SIR( EpidemicModel ):
        
    stationary_processes = 'recovery',
    states = 'S', 'I', 'R'
        
    def _init_model(self):

        self.S = self.population - self.init_exposed
        self.I = self.init_exposed
        self.R = 0

        self.event_list = list()
        # Push recoveries for inital onto queue
        for i in range(self.init_exposed):
            recovery_time = next( getattr(self, '_recovery_iter') )
            self._push((recovery_time, '_EVENT_recovery'))

        self._insert_exposure_event()
    
class SIRS(EpidemicModel):

    stationary_processes = 'recovery', 'mutation'
    states = 'S', 'I', 'R'
        
    def _init_model(self):

        self.S = self.population - self.init_exposed
        self.I = self.init_exposed
        self.R = 0

        self.event_list = list()
        # Push recoveries for inital onto queue
        for i in range(self.init_exposed):
            recovery_time = next( getattr(self, '_recovery_iter') )
            self._push((recovery_time, '_EVENT_recovery'))

        self._insert_exposure_event()

    def _EVENT_recovery(self):

        self.I -= 1
        self.R += 1
        mutation_time = next( getattr(self, '_mutation_iter') )
        self._push( (self.time + mutation_time, '_EVENT_mutation') )

    def _EVENT_mutation(self):

        self.R -= 1
        self.S += 1

class SIRD(EpidemicModel):
    
    stationary_processes = 'recovery', 'death'
    states = 'S', 'I', 'R', 'D'

    def _init_model(self):

        self.S = self.population - self.init_exposed
        self.I = self.init_exposed
        self.R = 0
        self.D = 0

        self.event_list = list()
        self._binom_iter = self._rv_generator( ss.bernoulli(p=self.prob_dead) )

        # Push  nital  recoveries or deaths onto queue
        for i in range(self.init_exposed):
            self._add_death_or_recovery()

        self._insert_exposure_event()
        
    def _EVENT_exposure(self):
        
        self.S -= 1
        self.I += 1
        
        self._add_death_or_recovery()
        
    def _EVENT_death(self):
        
        self.I -= 1
        self.D += 1

    def _add_death_or_recovery(self):

        does_die = next( self._binom_iter )
        if does_die:
            dead_time = next( getattr(self, '_death_iter') )
            self._push( (self.time + dead_time, '_EVENT_death' ))
        else:
            recovery_time = next( getattr(self, '_recovery_iter') )
            self._push((self.time + recovery_time, '_EVENT_recovery'))
        
class SR_SIR(EpidemicModel):

    stationary_processes = 'recovery',
    states = 'S', 'I', 'R'

    def _init_model(self):

        self.S = self.population - self.init_exposed
        self.I = self.init_exposed
        self.R = 0

        self.event_list = list()
        
        # Push recoveries for inital onto queue
        for i in range(self.init_exposed):
            recovery_time = next( getattr(self, '_recovery_iter') )
            self._push((recovery_time, '_EVENT_recovery'))

        # Push vaccine event into queue
        self._push((self.begin_vaccine, '_EVENT_determine_vaccines'))

        self._insert_exposure_event()

    def _EVENT_determine_vaccines(self):

        if not self.I >= 1:
            return

        vaccines_per_day = int(self.vaccine_rate(self.time))
        vaccine_times = np.linspace(0, 1, num=vaccines_per_day + 1,
            endpoint=False)

        for time in vaccine_times[1:]:
            self._push(((self.time + time, '_EVENT_vaccine')))

        self._push(((self.time + 1, '_EVENT_determine_vaccines')))

    def _EVENT_vaccine(self):

        if self.S <= 0:
            return

        self.S -= 1
        self.R += 1
    

class SEIR(EpidemicModel):

    stationary_processes = 'incubation', 'recovery'
    states = 'S', 'E', 'I', 'R'
    
    def _init_model(self):
        
        self.S = self.population - self.init_exposed 
        self.E = self.init_exposed
        self.I = 0
        self.R = 0
        
        self.event_list = list()
        # Push incubations for inital onto queue
        for i in range(self.init_exposed):
            incubation_time = next( getattr(self, '_incubation_iter') )
            self._push((incubation_time, '_EVENT_incubation'))

    def _EVENT_exposure(self):

        self.S -= 1
        self.E += 1
        
        incubation_time =  next( getattr(self, '_incubation_iter') )
        self._push( (self.time + incubation_time, '_EVENT_incubation'))
    
    def _EVENT_incubation(self):
        
        self.E -= 1
        self.I += 1

        recovery_time =  next( getattr(self, '_recovery_iter') )
        self._push( (self.time + recovery_time, '_EVENT_recovery' ))
        
    def _EVENT_recovery(self):

        self.I -= 1
        self.R += 1

class SR_SEIRSD(EpidemicModel):

    stationary_processes = 'incubation', 'recovery',  'death', 'mutation' 
    states = 'S', 'E', 'I', 'R', 'D'

    def _init_model(self):

        self.S = self.population - self.init_exposed
        self.E = self.init_exposed
        self.I = 0
        self.R = 0
        self.D = 0

        self.event_list = list()
        self._binom_iter = self._rv_generator(ss.bernoulli(p=self.prob_dead))
        

        # Push incubations for inital onto queue
        for i in range(self.init_exposed):
            incubation_time = next( getattr(self, '_incubation_iter') )
            self._push((incubation_time, '_EVENT_incubation'))

        # Push vaccine event into queue
        self._push((self.begin_vaccine, '_EVENT_determine_vaccines'))

    def _EVENT_exposure(self):

        self.S -= 1
        self.E += 1
        
        incubation_time = next( getattr(self, '_incubation_iter') )
        self._push( (self.time + incubation_time, '_EVENT_incubation'))

    def _EVENT_incubation(self):

        self.E -= 1
        self.I += 1
        
        self._add_death_or_recovery()
            
    def _EVENT_recovery(self):

        self.I -= 1
        self.R += 1
        
        mutation_time = next( getattr(self, '_mutation_iter') )
        self._push((self.time + mutation_time, '_EVENT_mutation'))
        
    def _EVENT_death(self):
        
        self.I -= 1
        self.D += 1

    def _EVENT_determine_vaccines(self):

        if not self.I >= 1:
            return

        vaccines_per_day = int(self.vaccine_rate(self.time))
        vaccine_times = np.linspace(0, 1, num=vaccines_per_day + 1,
            endpoint=False)

        for time in vaccine_times[1:]:
            self._push(((self.time + time, '_EVENT_vaccine')))

        self._push(((self.time + 1, '_EVENT_determine_vaccines')))

    def _EVENT_vaccine(self):

        if self.S <= 0:
            return

        mutation_time = next( getattr(self, '_mutation_iter') )
        self._push((self.time + mutation_time, '_EVENT_mutation'))
        
        self.S -= 1
        self.R += 1
        
    def _EVENT_mutation(self):

        self.R -= 1
        self.S += 1

    def _add_death_or_recovery(self):

        does_die = next(self._binom_iter)
        
        if does_die:
            dead_time = next( getattr(self, '_death_iter') )
            self._push( (self.time + dead_time, '_EVENT_death' ))
        else:
            recovery_time =next( getattr(self, '_recovery_iter') )
            self._push( (self.time + recovery_time, '_EVENT_recovery' ))


class Ebola_SEIRSD(EpidemicModel):

    stationary_processes = 'incubation', 'recovery',  'death', 'mutation' 
    states = 'S', 'E', 'I', 'R', 'D', 'C'

    def _init_model(self):

        self.S = self.population - self.init_exposed
        self.E = self.init_exposed
        self.I = 0
        self.R = 0
        self.D = 0

        # Total cases
        self.C = 0

        self.event_list = list()
        self._binom_iter = self._rv_generator(ss.bernoulli(p=self.prob_dead))
        

        # Push incubations for inital onto queue
        for i in range(self.init_exposed):
            incubation_time = next( getattr(self, '_incubation_iter') )
            self._push((incubation_time, '_EVENT_incubation'))

        # Push time of beta change into queue
        self._push((self.beta_change, '_EVENT_beta_change'))

    def _EVENT_exposure(self):

        self.S -= 1
        self.E += 1
        
        incubation_time = next( getattr(self, '_incubation_iter') )
        self._push( (self.time + incubation_time, '_EVENT_incubation'))

    def _EVENT_incubation(self):

        self.E -= 1
        self.I += 1
        self.C += 1
        
        self._add_death_or_recovery()
            
    def _EVENT_recovery(self):

        self.I -= 1
        self.R += 1
        
        mutation_time = next( getattr(self, '_mutation_iter') )
        self._push((self.time + mutation_time, '_EVENT_mutation'))
        
    def _EVENT_death(self):

         self.I -= 1
         self.D += 1

    def _EVENT_mutation(self):

        self.R -= 1
        self.S += 1

    def _add_death_or_recovery(self):

        does_die = next(self._binom_iter)

        if does_die:
            dead_time = next( getattr(self, '_death_iter') )
            self._push( (self.time + dead_time, '_EVENT_death' ))
        else:
            recovery_time =next( getattr(self, '_recovery_iter') )
            self._push( (self.time + recovery_time, '_EVENT_recovery' ))

    def _EVENT_beta_change(self):

        self.beta=self.new_beta 
        
class Covid_SEIRD(EpidemicModel):

    stationary_processes = 'incubation', 'recovery', 'death'
    states = 'S', 'E', 'I', 'R', 'D'
    
    def _init_model(self):
        
        self.S = self.population - self.init_exposed 
        self.E = self.init_exposed
        self.I = 0
        self.R = 0
        self.D = 0
        
        self.event_list = list()
        self._binom_iter = self._rv_generator(ss.bernoulli(p=self.prob_dead))
        
        # Push incubations for inital onto queue
        for i in range(self.init_exposed):
            incubation_time = next( getattr(self, '_incubation_iter') )
            self._push((incubation_time, '_EVENT_incubation'))
            
        self._push((self.beta_change, '_EVENT_beta_change'))

    def _EVENT_exposure(self):

        self.S -= 1
        self.E += 1
        
        incubation_time =  next( getattr(self, '_incubation_iter') )
        self._push( (self.time + incubation_time, '_EVENT_incubation'))
    
    def _EVENT_incubation(self):
        
        self.E -= 1
        self.I += 1
        
        self._add_death_or_recovery()
        
        #recovery_time =  next( getattr(self, '_recovery_iter') )
        #self._push( (self.time + recovery_time, '_EVENT_recovery' ))
        
    def _EVENT_recovery(self):

        self.I -= 1
        self.R += 1
        
    def _EVENT_death(self):
        
        self.I -= 1
        self.D += 1

    def _add_death_or_recovery(self):

        does_die = next( self._binom_iter )
        if does_die:
            dead_time = next( getattr(self, '_death_iter') )
            self._push( (self.time + dead_time, '_EVENT_death' ))
        else:
            recovery_time = next( getattr(self, '_recovery_iter') )
            self._push((self.time + recovery_time, '_EVENT_recovery'))
            
            
    def _EVENT_beta_change(self):

        self.beta=self.new_beta


class Plague_SEIRD(EpidemicModel):

    stationary_processes = 'incubation', 'recovery',  'death'
    states = 'S', 'E', 'I', 'R', 'D', 'C'

    def _init_model(self):

        self.S = self.population - self.init_exposed
        self.E = self.init_exposed
        self.I = 0
        self.R = 0
        self.D = 0

        # Total cases
        self.C = 0

        self.event_list = list()
        self._binom_iter = self._rv_generator(ss.bernoulli(p=self.prob_dead))
        
        # Push incubations for inital onto queue
        for i in range(self.init_exposed):
            incubation_time = next( getattr(self, '_incubation_iter') )
            self._push((incubation_time, '_EVENT_incubation'))

        # Push time of beta change into queue
        self._push((self.beta_change, '_EVENT_beta_change'))

    def _EVENT_exposure(self):

        self.S -= 1
        self.E += 1
        
        incubation_time = next( getattr(self, '_incubation_iter') )
        self._push( (self.time + incubation_time, '_EVENT_incubation'))

    def _EVENT_incubation(self):

        self.E -= 1
        self.I += 1
        self.C += 1
        
        self._add_death_or_recovery()
            
    def _EVENT_recovery(self):

        self.I -= 1
        self.R += 1
        
    def _EVENT_death(self):
        
        self.I -= 1
        self.D += 1
        
    def _EVENT_beta_change(self):

        self.beta=self.new_beta
        
    def _add_death_or_recovery(self):

        does_die = next(self._binom_iter)
        
        if does_die:
            dead_time = next( getattr(self, '_death_iter') )
            self._push( (self.time + dead_time, '_EVENT_death' ))
        else:
            recovery_time =next( getattr(self, '_recovery_iter') )
            self._push( (self.time + recovery_time, '_EVENT_recovery' ))
    