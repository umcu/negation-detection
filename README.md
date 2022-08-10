# Negation detection
Code for negation detection in Dutch clinical texts. 

TODO: Add URL to paper.

## Folder structure
```
negation-detection
└───bilstm             : biLSTM method code 
└───data               : input data (not git tracked)
└───models             : model output files (not git tracked)
└───results            : result files
└───robbert            : RobBERT method code
└───rule-based_context : Rule-based method code
└───utils              : Code for general (pre-)processing
```

- Notebooks to reproduce results for assessed methods are in their respective folders.
- Notebooks for general evaluation, error analysis and error comparisons are in the root directory.

If you find any of the code published in this repository useful please cite as:

@software{bram_van_es_2022_6980076,
  author       = {Bram van Es and
                  Leon C. Reteig and
                  Sander C. Tan and
                  Marijn Schraagen and
                  Myrthe M. Hemker and
                  Sebastiaan R.S. Arends and
                  Miguel Rios and
                  Saskia Haitjema},
  title        = {{Negation detection using various rule- 
                   based/machine learning methods}},
  month        = aug,
  year         = 2022,
  publisher    = {Zenodo},
  version      = 1,
  doi          = {10.5281/zenodo.6980076},
  url          = {https://doi.org/10.5281/zenodo.6980076}
}


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6980076.svg)](https://doi.org/10.5281/zenodo.6980076)
