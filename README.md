# Negation detection
Code for negation detection in Dutch clinical texts. 

Publication: https://doi.org/10.48550/arxiv.2209.00470

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

We place the finetuned BERT-models on the Huggingface [modelhub](https://huggingface.co/UMCU).

If you find any of the code published in this repository useful please cite as:

```
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
```

and the associated publication

```
@misc{https://doi.org/10.48550/arxiv.2209.00470,
  doi = {10.48550/ARXIV.2209.00470},
  
  url = {https://arxiv.org/abs/2209.00470},
  
  author = {van Es, Bram and Reteig, Leon C. and Tan, Sander C. and Schraagen, Marijn and Hemker, Myrthe M. and Arends, Sebastiaan R. S. and Rios, Miguel A. R. and Haitjema, Saskia},
  
  keywords = {Computation and Language (cs.CL), Information Retrieval (cs.IR), Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences, I.2.7; J.3; H.3.3, 68T50, 68P20},
  
  title = {Negation detection in Dutch clinical texts: an evaluation of rule-based and machine learning methods},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6980076.svg)](https://doi.org/10.5281/zenodo.6980076)
