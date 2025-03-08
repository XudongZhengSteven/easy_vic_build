# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

from deap import creator, base, tools, algorithms
import random
import pickle
import os
from tqdm import *
from ..decoractors import clock_decorator

class NSGAII_Base:

    def __init__(self, algParams={"popSize": 40, "maxGen": 250, "cxProb": 0.7, "mutateProb": 0.2}, save_path="checkpoint.pkl"):
        # set algorithm params
        self.popSize = algParams["popSize"]
        self.maxGen = algParams["maxGen"]
        self.toolbox = base.Toolbox()
        self.set_algorithm_params(**algParams)
        
        # create
        self.createFitness()
        self.createInd()
        
        # register
        self.registerInd()
        self.registerPop()
        self.registerEvaluate()
        self.registerOperators()
        
        # set initial variables
        self.history = []
        self.current_generation = 0
        self.initial_population = None
        
        # set save path
        self.save_path = save_path
        
        # try to load state (if exist)
        self.load_state()
    
    #* Design for your own situation
    def get_obs(self):
        self.obs = 0
    
    def get_sim(self):
        self.sim = 0
    
    def set_algorithm_params(self, popSize=None, maxGen=None, cxProb=None, mutateProb=None):
        self.toolbox.popSize = 40 if not popSize else popSize
        self.toolbox.maxGen = 250 if not maxGen else maxGen
        self.toolbox.cxProb = 0.7 if not cxProb else cxProb
        self.toolbox.mutateProb = 0.2 if not mutateProb else mutateProb
    
    #* Design for your own situation
    def createFitness(self):
        creator.create("Fitness", base.Fitness, weights=(-1.0,))
    
    def createInd(self):
        creator.create("Individual", list, fitness=creator.Fitness)
    
    #* Design for your own situation
    def samplingInd(self):
        # example: generate 5 elements/params in each Ind
        ind_elements = [random.uniform(-10, 10) for _ in range(5)]
        return creator.Individual(ind_elements)
    
    def registerInd(self):
        self.toolbox.register("individual", self.samplingInd)
    
    def registerPop(self):
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
    
    #* Design for your own situation
    def evaluate(self, ind):
        x, y = ind
        return (x**2 + y**2,)
        
    def registerEvaluate(self):
        self.toolbox.register("evaluate", self.evaluate)
    
    def evaluatePop(self, population):
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
    
    #* Design for your own situation
    @staticmethod
    def operatorMate(parent1, parent2):
        # parent is ind
        kwargs = {}
        return tools.cxTwoPoint(parent1, parent2, **kwargs)
    
    #* Design for your own situation
    @staticmethod
    def operatorMutate(ind):
        kwargs = {}
        return tools.mutFlipBit(ind, kwargs)

    #* Design for your own situation
    @staticmethod
    def operatorSelect(population):
        kwargs = {}
        return tools.selTournament(population, **kwargs)
    
    def registerOperators(self):
        self.toolbox.register("mate", self.operatorMate)
        self.toolbox.register("mutate", self.operatorMutate)
        self.toolbox.register("select", self.operatorSelect)
    
    def apply_genetic_operators(self, offspring):
        # it can be implemented by algorithms.varAnd
        # crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self.toolbox.cxProb:
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # mutate
        for mutant in offspring:
            if random.random() < self.toolbox.mutateProb:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values
    
    def select_next_generation(self, combined):
        fronts = tools.sortNondominated(combined, len(combined), first_front_only=False)
        next_generation = []
        for front in fronts:
            if len(next_generation) + len(front) <= self.popSize:
                next_generation.extend(front)
            else:
                # cal crowding
                tools.emo.assignCrowdingDist(front)
                front.sort(key=lambda ind: ind.fitness.crowding_dist, reverse=True)
                next_generation.extend(front[:self.popSize - len(next_generation)])
                break
        
        return next_generation
    
    def print_results(self, population):
        best_ind = tools.selBest(population, k=1)[0]
        print("best_ind:", best_ind)
        print("fitness:", best_ind.fitness.values)
        
    def load_state(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, "rb") as f:
                state = pickle.load(f)
                self.current_generation = state["current_generation"]
                self.initial_population = state["initial_population"]
                self.population = state["population"]
                self.history = state["history"]
                
        else:
            self.population = self.toolbox.population(n=self.popSize)
            self.initial_population = self.population[:]
            
    def save_state(self):
        state = {'current_generation': self.current_generation,
                'population': self.population,
                'initial_population': self.initial_population,
                'history': self.history
                }
        
        with open(self.save_path, 'wb') as f:
            pickle.dump(state, f)
    
    @clock_decorator(print_arg_ret=False)
    def run(self):
        # print gen
        print(f"current generation: {self.current_generation}")
        
        # evaluate population
        print("============== evaluating initial pop ==============")
        self.evaluatePop(self.population)

        # loop for generations
        start_gen = self.current_generation
        print("============== NSGAII generating ==============")
        for gen in tqdm(range(start_gen, self.maxGen), desc="loop for NSGAII generation", colour="green"):
            # current generation
            self.current_generation = gen
            
            # generate offspring
            offspring = self.toolbox.select(self.population, self.popSize)
            offspring = list(map(self.toolbox.clone, offspring))

            # apply_genetic_operators and evaluate it
            self.apply_genetic_operators(offspring)
            self.evaluatePop(offspring)

            # combine population and offspring
            combined = self.population + offspring
            
            # sortNondominated to get fronts and front
            front = tools.sortNondominated(combined, len(combined), first_front_only=True)
            
            # cal crowding
            for f in front:
                tools.emo.assignCrowdingDist(f)
            
            # save history (population and front)
            self.history.append((self.population, front))
            
            # save state at the end of each gen
            self.save_state()
            
            # update population: select next generation
            self.population[:] = self.select_next_generation(combined)
        
        print("============== Results ==============")
        self.print_results(self.population)
