# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: algorithm_NSGAII

This module implements the NSGA-II (Non-dominated Sorting Genetic Algorithm II)
for multi-objective optimization. The algorithm is designed to evolve a population
of individuals through selection, crossover, and mutation, in order to converge towards
an optimal Pareto front. It also includes the capability to save and load the state
of the algorithm, enabling checkpointing for long-running optimization tasks.

Class:
--------
    - NSGAII_Base: A class representing the core logic of the NSGA-II algorithm.

Class Methods:
---------------
    - __init__: Initializes the NSGAII algorithm with the given parameters.
    - get_obs: A placeholder method to get observed values (to be defined for specific use cases).
    - get_sim: A placeholder method to get simulated values (to be defined for specific use cases).
    - set_algorithm_params: Configures the algorithm parameters like population size,
      number of generations, crossover and mutation probabilities.
    - createFitness: Creates a fitness class with minimization as the objective.
    - createInd: Creates an individual with a fitness attribute.
    - samplingInd: Samples a new individual with randomly initialized parameters.
    - registerInd: Registers the individual creation function to the toolbox.
    - registerPop: Registers the population creation function to the toolbox.
    - evaluate: A placeholder method for evaluating the fitness of individuals (to be defined).
    - registerEvaluate: Registers the evaluation function to the toolbox.
    - evaluatePop: Evaluates the fitness of the entire population.
    - operatorMate: A crossover operator for mating two parents.
    - operatorMutate: A mutation operator that flips a bit in an individual.
    - operatorSelect: A selection operator based on tournament selection.
    - registerOperators: Registers the genetic operators (mate, mutate, select) to the toolbox.
    - apply_genetic_operators: Applies crossover and mutation operators to the offspring.
    - select_next_generation: Selects the next generation of individuals using non-dominated sorting
      and crowding distance.
    - print_results: Prints the best individual and its fitness.
    - load_state: Loads the algorithm's state from a checkpoint file.
    - save_state: Saves the algorithm's state to a checkpoint file.
    - run: Executes the NSGA-II algorithm over multiple generations, applying genetic operators
      and selecting the next generation, while saving the state after each generation.

Dependencies:
-------------
    - deap: Provides the core functionality for evolutionary algorithms.
    - random: Used for generating random values for crossover, mutation, and initial population.
    - pickle: Used for saving and loading the algorithm's state.
    - os: Used for checking and managing file paths.
    - tqdm: Provides a progress bar for the generation loop.
    - ..decorators: Contains the `clock_decorator` for measuring execution time.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
"""

import os
import pickle
import random

from deap import algorithms, base, creator, tools
from tqdm import *

from ... import logger
from ..decoractors import clock_decorator


class NSGAII_Base:
    """
    A class that implements the NSGA-II (Non-dominated Sorting Genetic Algorithm II) for multi-objective optimization.
    This class provides methods for setting up the genetic algorithm, evaluating individuals,
    applying genetic operators (crossover, mutation), selecting the next generation, and saving/loading the algorithm state.

    Attributes
    ----------
    popSize : int
        The population size for each generation.
    maxGen : int
        The maximum number of generations to run the algorithm.
    save_path : str
        The path to save and load the algorithm state.
    history : list
        A history of the population and non-dominated fronts at each generation.
    current_generation : int
        The current generation number.
    initial_population : list
        The initial population before any genetic operators are applied.
    population : list
        The current population of individuals.

    Methods
    -------
    __init__(algParams, save_path):
        Initializes the NSGA-II algorithm with the given parameters and attempts to load the state.

    get_obs():
        Placeholder function to get the observed values (to be designed for specific use cases).

    get_sim():
        Placeholder function to get the simulated values (to be designed for specific use cases).

    set_algorithm_params(popSize, maxGen, cxProb, mutateProb):
        Sets the parameters for the genetic algorithm (population size, max generations, crossover probability, and mutation probability).

    createFitness():
        Creates the fitness function for individuals.

    createInd():
        Creates an individual representation (a list).

    samplingInd():
        Samples a new individual by generating random values for its elements.

    registerInd():
        Registers the individual sampling function with the toolbox.

    registerPop():
        Registers the population initialization function with the toolbox.

    evaluate(ind):
        A placeholder function for evaluating an individual's fitness (to be customized for specific use cases).

    registerEvaluate():
        Registers the evaluation function with the toolbox.

    evaluatePop(population):
        Evaluates the fitness of the entire population.

    operatorMate(parent1, parent2):
        Defines the crossover operation for mating two individuals (using a two-point crossover).

    operatorMutate(ind):
        Defines the mutation operation for an individual (using a bit-flip mutation).

    operatorSelect(population):
        Defines the selection operation (using tournament selection).

    registerOperators():
        Registers the genetic operators (mate, mutate, and select) with the toolbox.

    apply_genetic_operators(offspring):
        Applies the genetic operators (crossover and mutation) to the offspring.

    select_next_generation(combined):
        Selects the next generation by sorting individuals based on Pareto dominance and applying crowding distance.

    print_results(population):
        Prints the results of the best individual from the final population.

    load_state():
        Loads the algorithm state from the specified save path (if a saved state exists).

    save_state():
        Saves the current algorithm state (current generation, population, and history) to the specified save path.

    run():
        Runs the NSGA-II algorithm for the specified number of generations, applying genetic operators and selecting the next generation.
    """

    def __init__(
        self,
        algParams={"popSize": 40, "maxGen": 250, "cxProb": 0.7, "mutateProb": 0.2},
        save_path="checkpoint.pkl",
    ):
        """
        Initializes the NSGA-II algorithm with the given parameters.

        Parameters
        ----------
        algParams : dict
            Dictionary containing the algorithm parameters:
            - popSize: The population size for each generation.
            - maxGen: The maximum number of generations to run the algorithm.
            - cxProb: The crossover probability.
            - mutateProb: The mutation probability.

        save_path : str, optional
            The path to save and load the algorithm state (default is "checkpoint.pkl").
        """
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

    # * Design for your own situation
    def get_obs(self):
        """
        Placeholder function to get observed values (to be designed for specific use cases).

        Returns
        -------
        None
        """
        self.obs = 0

    def get_sim(self):
        """
        Placeholder function to get simulated values (to be designed for specific use cases).

        Returns
        -------
        None
        """
        self.sim = 0

    def set_algorithm_params(
        self, popSize=None, maxGen=None, cxProb=None, mutateProb=None
    ):
        """
        Sets the parameters for the genetic algorithm.

        Parameters
        ----------
        popSize : int, optional
            The population size for each generation (default is 40).
        maxGen : int, optional
            The maximum number of generations to run the algorithm (default is 250).
        cxProb : float, optional
            The crossover probability (default is 0.7).
        mutateProb : float, optional
            The mutation probability (default is 0.2).
        """
        self.toolbox.popSize = 40 if not popSize else popSize
        self.toolbox.maxGen = 250 if not maxGen else maxGen
        self.toolbox.cxProb = 0.7 if not cxProb else cxProb
        self.toolbox.mutateProb = 0.2 if not mutateProb else mutateProb

    # * Design for your own situation
    def createFitness(self):
        """Creates the fitness function for individuals."""
        creator.create("Fitness", base.Fitness, weights=(-1.0,))

    def createInd(self):
        """Creates an individual representation (a list)."""
        creator.create("Individual", list, fitness=creator.Fitness)

    # * Design for your own situation
    def samplingInd(self):
        """
        Samples a new individual by generating random values for its elements.

        Returns
        -------
        Individual
            A new individual sampled with random elements.
        """
        # example: generate 5 elements/params in each Ind
        ind_elements = [random.uniform(-10, 10) for _ in range(5)]
        return creator.Individual(ind_elements)

    def registerInd(self):
        """Registers the individual sampling function with the toolbox."""
        self.toolbox.register("individual", self.samplingInd)

    def registerPop(self):
        """Registers the population initialization function with the toolbox."""
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

    # * Design for your own situation
    def evaluate(self, ind):
        """
        A placeholder function for evaluating an individual's fitness.

        Parameters
        ----------
        ind : Individual
            The individual to evaluate.

        Returns
        -------
        tuple
            A tuple containing the fitness values.
        """
        x, y = ind
        return (x**2 + y**2,)

    def registerEvaluate(self):
        """Registers the evaluation function with the toolbox."""
        self.toolbox.register("evaluate", self.evaluate)

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

    # * Design for your own situation
    @staticmethod
    def operatorMate(parent1, parent2):
        """
        Defines the crossover operation for mating two individuals.

        Parameters
        ----------
        parent1 : Individual
            The first parent individual.
        parent2 : Individual
            The second parent individual.

        Returns
        -------
        tuple
            A tuple containing the offspring resulting from the crossover.
        """
        # parent is ind
        kwargs = {}
        return tools.cxTwoPoint(parent1, parent2, **kwargs)

    # * Design for your own situation
    @staticmethod
    def operatorMutate(ind):
        """
        Defines the mutation operation for an individual.

        Parameters
        ----------
        ind : Individual
            The individual to mutate.

        Returns
        -------
        tuple
            A tuple containing the mutated individual.
        """
        kwargs = {}
        return tools.mutFlipBit(ind, kwargs)

    # * Design for your own situation
    @staticmethod
    def operatorSelect(population):
        """
        Defines the selection operation for choosing individuals from the population.

        Parameters
        ----------
        population : list of Individual
            The population from which to select individuals.

        Returns
        -------
        list
            A list of selected individuals.
        """
        kwargs = {}
        return tools.selTournament(population, **kwargs)

    def registerOperators(self):
        """Registers the genetic operators (mate, mutate, and select) with the toolbox."""
        self.toolbox.register("mate", self.operatorMate)
        self.toolbox.register("mutate", self.operatorMutate)
        self.toolbox.register("select", self.operatorSelect)

    def apply_genetic_operators(self, offspring):
        """
        Applies the genetic operators (crossover and mutation) to the offspring.

        Parameters
        ----------
        offspring : list of Individual
            The offspring to apply the genetic operators to.
        """
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
        """
        Selects the next generation from the combined population and offspring.

        Parameters
        ----------
        combined : list of Individual
            The combined population of parents and offspring.

        Returns
        -------
        list
            The selected next generation.
        """
        fronts = tools.sortNondominated(combined, len(combined), first_front_only=False)
        next_generation = []
        for front in fronts:
            if len(next_generation) + len(front) <= self.popSize:
                next_generation.extend(front)
            else:
                # cal crowding
                tools.emo.assignCrowdingDist(front)
                front.sort(key=lambda ind: ind.fitness.crowding_dist, reverse=True)
                next_generation.extend(front[: self.popSize - len(next_generation)])
                break

        return next_generation

    def print_results(self, population):
        """
        Prints the results of the best individual from the final population.

        Parameters
        ----------
        population : list of Individual
            The final population.
        """
        best_ind = tools.selBest(population, k=1)[0]
        logger.info("best_ind:", best_ind)
        logger.info("fitness:", best_ind.fitness.values)

    def load_state(self):
        """
        Loads the algorithm state from the specified save path if a saved state exists.
        """
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
        """
        Saves the current algorithm state (current generation, population, and history) to the specified save path.
        """
        state = {
            "current_generation": self.current_generation,
            "population": self.population,
            "initial_population": self.initial_population,
            "history": self.history,
        }

        with open(self.save_path, "wb") as f:
            pickle.dump(state, f)

    @clock_decorator(print_arg_ret=False)
    def run(self):
        """
        Runs the NSGA-II algorithm for the specified number of generations.

        This method applies genetic operators, evaluates individuals,
        selects the next generation, and stores the results.

        Returns
        -------
        list
            The final population after all generations.
        """
        # evaluate population
        self.evaluatePop(self.population)

        # loop for generations
        start_gen = self.current_generation
        for gen in tqdm(
            range(start_gen, self.maxGen),
            desc="loop for NSGAII generation",
            colour="green",
        ):
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
            front = tools.sortNondominated(
                combined, len(combined), first_front_only=True
            )

            # cal crowding
            for f in front:
                tools.emo.assignCrowdingDist(f)

            # save history (population and front)
            self.history.append((self.population, front))

            # save state at the end of each gen
            self.save_state()

            # update population: select next generation
            self.population[:] = self.select_next_generation(combined)

        self.print_results(self.population)

        return self.population
