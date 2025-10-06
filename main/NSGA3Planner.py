import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination
from util import vecPredictProba


class NSGA3Planner:
    def __init__(self, reqClassifiers, targetConfidence, controllableFeatureIndices, controllableFeatureDomains,
                 optimizationDirections, successScoreFunction, optimizationScoreFunction):
        self.reqClassifiers = reqClassifiers
        self.targetConfidence = targetConfidence
        self.successScoreFunction = successScoreFunction
        self.optimizationScoreFunction = optimizationScoreFunction

        # create the reference directions to be used for the optimization
        ref_dirs = get_reference_directions("das-dennis", len(controllableFeatureIndices), n_partitions=12)

        # create the algorithm object
        self.algorithm = NSGA3(ref_dirs=ref_dirs)

        self.termination = get_termination("n_gen", 100)
        self.externalFeatures = np.zeros((1, len(controllableFeatureIndices)))

        # create problem instance
        self.problem = Adaptation(reqClassifiers, targetConfidence, self.algorithm.pop_size, controllableFeatureIndices,
                                  controllableFeatureDomains, optimizationDirections)

    def findAdaptation(self, externalFeatures):
        # set problem
        self.problem.externalFeatures = externalFeatures

        # execute the optimization
        res = minimize(self.problem,
                       self.algorithm,
                       seed=1,
                       termination=self.termination)

        if res.X is not None:
            adaptations = res.X
        else:
            adaptations = np.array([individual.X for individual in res.pop])

        adaptations = np.append(adaptations, np.repeat([externalFeatures], adaptations.shape[0], axis=0), axis=1)
        optimizationScores = [self.optimizationScoreFunction(a) for a in adaptations]

        if res.X is not None:
            adaptationIndex = np.argmax(optimizationScores)
        else:
            successScores = [self.successScoreFunction(a, self.reqClassifiers, self.targetConfidence) for a in adaptations]
            adaptationIndex = np.argmax(successScores)

        adaptation = adaptations[adaptationIndex]
        confidence = vecPredictProba(self.reqClassifiers, [adaptation])[0]
        score = self.optimizationScoreFunction(adaptation)

        return adaptation, confidence, score


class Adaptation(Problem):
    @property
    def externalFeatures(self):
        return self._externalFeatures

    @externalFeatures.setter
    def externalFeatures(self, externalFeatures):
        # Store the single external features vector for later use
        self._externalFeatures = np.array([externalFeatures])

    def __init__(self, models, targetConfidence, popSize, controllableFeatureIndices, controllableFeatureDomains,
                 optimizationDirections):
        super().__init__(n_var=len(controllableFeatureIndices), n_obj=len(controllableFeatureIndices),
                         n_constr=len(models), xl=controllableFeatureDomains[:, 0], xu=controllableFeatureDomains[:, 1])
        self.models = models
        self.targetConfidence = np.repeat([targetConfidence], popSize, axis=0)
        self.controllableFeatureIndices = controllableFeatureIndices
        self.optimizationDirections = optimizationDirections
        self.popSize = popSize
        self.externalFeatures = []

    def _evaluate(self, x, out, *args, **kwargs):
        # Use actual batch size instead of popSize
        actual_batch_size = x.shape[0]
        
        # Create external features for the actual batch size
        externalFeatures_batch = np.repeat([self.externalFeatures[0]], actual_batch_size, axis=0)
        
        xFull = np.empty((actual_batch_size, self.n_var + externalFeatures_batch.shape[1]))
        xFull[:, self.controllableFeatureIndices] = x
        externalFeatureIndices = np.delete(np.arange(xFull.shape[1]), self.controllableFeatureIndices)
        xFull[:, externalFeatureIndices] = externalFeatures_batch

        out["F"] = [-self.optimizationDirections[i] * x[:, i] for i in range(x.shape[1])]
        
        # Adjust target confidence for actual batch size
        targetConfidence_batch = np.repeat([self.targetConfidence[0]], actual_batch_size, axis=0)
        constraints = targetConfidence_batch - vecPredictProba(self.models, xFull)
        out["G"] = constraints.T if len(self.models) > 1 else constraints.flatten()
