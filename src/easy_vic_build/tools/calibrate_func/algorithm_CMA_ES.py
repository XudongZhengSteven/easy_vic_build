# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

from deap import base, creator, tools, cma
from tqdm import *
from copy import deepcopy

from ... import logger
from ..decoractors import clock_decorator


class CMA_ES_Base:

    def __init__(
        self,
        algParams={"dim": None, "popSize": 20, "maxGen": 250, "sigma":0.5},
        save_path="checkpoint.pkl",
    ):
        # set algorithm params
        self.dim = algParams["dim"]
        self.popSize = algParams["popSize"]
        self.maxGen = algParams["maxGen"]
        self.sigma = algParams["sigma"]
        self.algParams = algParams
        self.toolbox = base.Toolbox()

        # create base types
        self.createFitness()
        self.createInd()

        # internal variables
        self.history = []
        self.current_generation = 0
        self.population = None

        self.save_path = save_path

        # load or create population
        self.load_state()
        
        # register DEAP functions
        self.registerEvaluate()
        self.registerGenerate()
        self.registerUpdate()
        
        # init pop
        if self.population is None:
            self.population = self.toolbox.generate()
            self.initial_population = self.population[:]

    # -----------------------------
    #  set algorithm parameters
    # -----------------------------
    def init_cma_strategy(self, dim, popSize, sigma, **kwargs):
        strategy = cma.Strategy(
            centroid=[0.0]*dim,
            sigma=sigma,
            lambda_=popSize,
            **kwargs
        )
        return strategy
    
    # -----------------------------
    #  User should define these
    # -----------------------------
    def get_obs(self):
        self.obs = None

    def get_sim(self):
        self.sim = None

    def createFitness(self):
        """ For single objective minimization """
        creator.create("Fitness", base.Fitness, weights=(-1.0,))

    def createInd(self):
        creator.create("Individual", list, fitness=creator.Fitness)

    def evaluate(self, ind):
        """User-defined fitness"""
        x, y = ind
        return (x**2 + y**2,)

    # -----------------------------
    #  Registering DEAP components
    # -----------------------------
    def registerEvaluate(self):
        self.toolbox.register("evaluate", self.evaluate)
        
    def registerGenerate(self):
        self.toolbox.register("generate", self.strategy.generate, creator.Individual)
    
    def registerUpdate(self):
        self.toolbox.register("update", self.strategy.update)

    # -----------------------------
    #  Generation in CMA_ES
    # -----------------------------
    def evaluatePop(self, population):
        """
        Evaluates the fitness of the entire population.

        Parameters
        ----------
        population : list of Individual
            The population to evaluate.
        """
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
    
    def updatePop(self, offspring):
        population = offspring
        return population
        
    # -----------------------------
    #  Save & Load
    # -----------------------------
    def load_state(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, "rb") as f:
                state = pickle.load(f)
                self.current_generation = state["current_generation"]
                self.initial_population = state["initial_population"]
                self.population = state["population"]
                self.history = state["history"]
                self.strategy = state["strategy"]
        else:
            self.strategy = self.init_cma_strategy(self.dim, self.popSize, self.sigma)
            self.population = None
            self.initial_population = None

    def save_state(self):
        state = {
            "current_generation": self.current_generation,
            "population": deepcopy(self.population),
            "history": deepcopy(self.history),
            "initial_population": deepcopy(self.initial_population),
            "strategy": deepcopy(self.strategy)
        }
        with open(self.save_path, "wb") as f:
            pickle.dump(state, f)

    # -----------------------------
    #  Print and Plot
    # -----------------------------
    def print_results(self, population):
        """
        Prints the best individual in current population.
        Suitable for CMA-ES (single-objective).
        """
        best_ind = tools.selBest(population, k=1)[0]
        logger.info(f"Best individual: {best_ind}")
        logger.info(f"Fitness: {best_ind.fitness.values[0]}")
        
    def plot_progress(self, plot_dir="cmaes_progress", ylim=None):
        # check plot_dir
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        # Extract history
        all_pop_fitness = []
        all_best_ind_fitness = []

        for h in self.history:
            pop = h["population"]
            best_ind = h["best_ind"]
            
            pop_fitness = [ind.fitness.values[0] for ind in pop]
            best_ind_fitness = best_ind.fitness.values[0]

            all_pop_fitness.append(pop_fitness)
            all_best_ind_fitness.append(best_ind_fitness)

        generations = np.arange(len(all_best_ind_fitness))
        
        # plot
        plt.figure(figsize=(8, 5))
        for gen, vals in enumerate(all_pop_fitness):
            gens = np.full(len(vals), gen)
            plt.scatter(gens, vals, s=12, c="gray", alpha=0.4, zorder=2)
        
        plt.scatter(generations, all_best_ind_fitness, c="red", s=25, zorder=5)
        plt.plot(generations, all_best_ind_fitness, c="blue", linewidth=1.5, zorder=4, label="Best fitness curve")
        
        if ylim is not None:
            if isinstance(ylim, (int, float)):  
                # only bottom is given
                plt.ylim(bottom=ylim)
            elif isinstance(ylim, (list, tuple)) and len(ylim) == 2:
                plt.ylim(ylim[0], ylim[1])
            else:
                raise ValueError("ylim must be a number or a tuple/list of two numbers")
            
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("CMA-ES Fitness Convergence Curve", fontsize=14, weight='bold')
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(plot_dir, "CMAES_process.png"))
        plt.close()
        
    # -----------------------------
    #  Run
    # -----------------------------
    @clock_decorator(print_arg_ret=False)
    def run(
        self,
        plot_progress=False,
        plot_dir="cmaes_progress",
        plot_ylim=None,
    ):
        # evaluate initial
        self.evaluatePop(self.population)

        start_gen = self.current_generation

        for gen in tqdm(
            range(start_gen, self.maxGen),
            desc="eaMuPlusLambda generation",
            colour="blue"
        ):
            self.current_generation = gen

            # offspring
            offspring = self.toolbox.generate()
            self.evaluatePop(offspring)
            
            # update
            self.toolbox.update(offspring)
            self.population = self.updatePop(offspring)
            
            # get best
            best_ind = tools.selBest(self.population, 1)[0]
            
            # save history
            self.history.append({
                "population": deepcopy(self.population),
                "best_ind": deepcopy(best_ind)
            })
            
            # save state
            self.save_state()
            
            # plot
            if plot_progress:
                self.plot_progress(
                    plot_dir=plot_dir,
                    ylim=plot_ylim,
                )

        self.print_results(self.population)
        
        return self.population