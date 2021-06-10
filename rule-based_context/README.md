# Negation detection with ConText

Python implementation of [ContextD](https://doi.org/10.1186/s12859-014-0373-3) with [medspacy](https://github.com/medspacy/medspacy)

## Running the code

1. Clone this repo, e.g.:
    ```zsh
    git clone https://github.com/umcu/negation-detection.git
    cd negation-detection
    ```
2. Make and activate a virtual environment, e.g.:
    ```zsh
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3. Install the required packages, e.g.:
     ```zsh
    python -m pip install -r requirements_context.txt
    ```
4. Open JupyterLab and run `rule-based_context/notebooks/context.ipynb`

## References

- **Original paper**: Harkema H, Dowling JN, Thornblade T, Chapman WW: [ConText: an algorithm for determining negation, experiencer, and temporal status from clinical reports](https://doi.org/10.1016/j.jbi.2009.05.002). J Biomaed Inform. 2009, 42: 839-851.
    - code: https://code.google.com/archive/p/negex/downloads

- **Dutch version**: Afzal Z, Pons E, Kang N, Sturkenboom MC, Schuemie MJ, Kors JA. [ContextD: an algorithm to identify contextual properties of medical terms in a Dutch clinical corpus. BMC Bioinformatics](https://doi.org/10.1186/s12859-014-0373-3). 2014 Nov 29;15(1):373.