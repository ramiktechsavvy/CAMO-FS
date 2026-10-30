# CAMO-FS optimization methodology

This implementation uses 261 predictors in fixed metadata order and the saved train/test split. EARLY (July 5), TRANSITION (July 14 minus July 5), and LATE (July 14) are operational temporal representations, not verified biological stages.

## Chromosome and feasibility

Each candidate is a 261-bit vector, where 1 selects a feature. Empty vectors are rejected/repaired; indices are zero-based into the canonical feature order. The target label is excluded from the chromosome.

## SHAP-guided initialization and variation

Half of the initial population is sampled independently using probabilities `clip(0.02 + 0.65*pi_j, 0.02, 0.67)`. The other half uses independently sampled chromosome densities between 0.05 and 0.95. This mixture prevents identical initialization and retains broad subset-size diversity. Uniform crossover is applied to a parent pair with probability 0.9. Mutation uses the requested per-bit probabilities: selected bits leave at `pm*(1-pi_j)` and unselected bits enter at `pm*pi_j`, with `pm=0.03`.

## Objectives and ECI

The objectives are minimized: `f1=1-Macro-F1`, `f2=selected_count/261`, and `f3=1-ECI_subset`. ECI uses the five existing SHAP feature-rank files and averages the 10 pairwise Kendall tau-b coefficients before mapping to [0,1]. For a candidate, each fold ranking is restricted to its selected features before Kendall tau is calculated. A subset with fewer than two features is assigned ECI=1 by convention because its ranking agreement is not estimable.

Full-feature ECI: 0.996387; representation ECI: {"EARLY": 0.9952953755680298, "TRANSITION": 0.9966319165998396, "LATE": 0.9960438385458433}.

## NSGA-II and archive

The implementation uses pairwise dominance sorting, rank/crowding tournament selection, uniform crossover, guided mutation, and elitist environmental selection from parent-plus-offspring populations. A nondominated archive is updated at every generation, de-duplicates chromosomes, removes dominated points, and is capped at 500 using crowding distance to preserve objective-space diversity. Population size is 60 for 40 generations with seed 42.

## Search fitness and computational compromise

A timing probe with 8,000 training rows, 100 features, and 100 boosting rounds took about 0.82 seconds for one fit. The requested naive workload can approach 2,400 unique candidates × 5 fits = 12,000 full-data fits; extrapolating the existing full baseline's roughly 129-second fit suggests many days of computation. Therefore evolutionary candidate fitness uses a fixed, stratified 10,000-row sample drawn only from the saved training partition, split 80/20 into 8,000 fit and 2,000 validation rows, and 30 boosting rounds. All other baseline XGBoost settings are unchanged. Candidate scores are approximate; caching avoids repeated chromosome fits. The chosen final solution is then evaluated with the full baseline's 300 rounds and stratified 5-fold CV over all 260,667 saved training rows. The test set is not used.

The search holdout's rare-class support is limited, and 30 rounds can underfit relative to the 300-round baseline; these surrogate scores can alter the approximate Pareto front and weighted selection. The final CV result is the reported model estimate, but the selected subset is conditional on the surrogate search.

## Pareto indicators

Hypervolume is exact in three dimensions with reference point (1,1,1). This is the worst feasible bound for each normalized minimization objective: classification error, subset ratio, and explanation instability each lie in [0,1]. Hypervolume contributions are leave-one-solution-out differences. Spread is reported as the coefficient of variation of each point's nearest-neighbor Euclidean distance in objective space (lower means more even spacing).

## Final selection

The archive objectives are min-max normalized. The selected point minimizes weighted distance to ideal (0,0,0), using `sqrt(0.5*x1^2 + 0.2*x2^2 + 0.3*x3^2)`. The paper calls its selection a knee point, but this code performs weighted distance-to-ideal selection; it does not detect a geometric knee. The selected subset has 5 features ({"EARLY": 0, "TRANSITION": 2, "LATE": 3}), subset ratio 0.019157, pooled full-CV Macro-F1 0.969073, and ECI 1.000000.

## Limitations

The optimizer uses a single saved SHAP prior and fold-ranking set. Fold ECI ranks do not include refitting the model per fold; they measure consistency among previously generated SHAP rankings. Candidate search fitness is an explicitly approximate holdout surrogate. Random row splits do not establish spatially independent generalization. Representation labels remain operational dates/differences, not phenological stages.

Run from the repository root with `python src/camo_fs/optimization/run_camo_fs.py`. Results and selected feature lists are saved under `results/camo_fs/`; full ECI calculations are under `results/eci/`.
