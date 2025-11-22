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
        if len(externalFeatures) > 0:
            self._externalFeatures = np.array([externalFeatures])
        else:
            self._externalFeatures = np.array([[]])

    def __init__(self, models, targetConfidence, popSize, controllableFeatureIndices, controllableFeatureDomains,
                 optimizationDirections):
        super().__init__(n_var=len(controllableFeatureIndices), n_obj=len(controllableFeatureIndices),
                         n_constr=len(models), xl=controllableFeatureDomains[:, 0], xu=controllableFeatureDomains[:, 1])
        self.models = models
        self.targetConfidence = targetConfidence
        self.controllableFeatureIndices = controllableFeatureIndices
        self.optimizationDirections = optimizationDirections
        self.popSize = popSize
        self._externalFeatures = np.array([[]])  # Initialize as empty

    def _evaluate(self, x, out, *args, **kwargs):
        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        n_samples = x.shape[0]
        
        # Check if we have external features properly set
        if self.externalFeatures.size == 0:
            raise ValueError("External features not set properly")
        
        n_external_features = self.externalFeatures.shape[1]
        total_features = self.n_var + n_external_features
        
        # Create full feature vectors
        xFull = np.zeros((n_samples, total_features))
        
        # Set controllable features
        xFull[:, self.controllableFeatureIndices] = x
        
        # Set external features
        externalFeatureIndices = np.delete(np.arange(total_features), self.controllableFeatureIndices)
        
        # Broadcast external features to all samples
        external_features_broadcast = np.repeat(self.externalFeatures, n_samples, axis=0)
        xFull[:, externalFeatureIndices] = external_features_broadcast

        # Objectives: minimize/maximize each controllable feature according to optimization directions
        objectives = np.zeros((n_samples, self.n_obj))
        for i in range(self.n_obj):
            objectives[:, i] = -self.optimizationDirections[i] * x[:, i]
        out["F"] = objectives

        # Constraints: ensure confidence is above target
        confidence_predictions = vecPredictProba(self.models, xFull)
        
        # Constraint: g(x) <= 0, so we formulate as (target - confidence) <= 0
        # which means confidence >= target
        if len(self.models) == 1:
            # Single requirement case
            target_val = self.targetConfidence if np.isscalar(self.targetConfidence) else self.targetConfidence[0]
            constraints = target_val - confidence_predictions.flatten()
            out["G"] = constraints.reshape(-1, 1)
        else:
            # Multiple requirements case
            target_val = self.targetConfidence if len(self.targetConfidence.shape) == 1 else self.targetConfidence[0]
            constraints = np.zeros((n_samples, len(self.models)))
            for i in range(len(self.models)):
                constraints[:, i] = target_val[i] - confidence_predictions[:, i]
            out["G"] = constraints
