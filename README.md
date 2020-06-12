## Feature Extraction by Grammatical Evolution for One-Class Time Series Classification


Please use the file **feature_extraction.py** to run the algorithm, and so to extract features from your time series.

The file **run_a_feature_extractor.py** shows you how to transform your time series in a feature-based representation according to a given feature-extractor. Then, the file shows you how to evaluate classification performance in a one-class scenario.

Other important files:
1. GRAMMAR: /grammars/tsc_grammar.bnf
2. EVOLUTIONARY PARAMETERS: /parameters/tsc_parameters.txt
3. FITNESS FUNCTION: /src/fitness/tsc_fitness.py
4. FUNCTIONS USED IN THE GRAMMAR (PRIMITIVES): /src/fitness/math_functions.py

Please refer to the paper for further information. Get in touch with me if necessary.


**Note:** This repository is basically a copy of [PonyGE2](https://github.com/PonyGE/PonyGE2) [2] with some adjustments required to perform feature-extraction from time series as described in the related paper.


**Contact:** mauceri.stefano@gmail.com


**References:**

<a id="1">[1]</a> ...

<a id="2">[2]</a> Fenton, Michael, James McDermott, David Fagan, Stefan Forstenlechner, Erik Hemberg, and Michael O'Neill. "Ponyge2: Grammatical evolution in python." In Proceedings of the Genetic and Evolutionary Computation Conference Companion, pp. 1194-1201. 2017.
