# Thesis (TODO: Find a better name)
This thesis is about detecting what/whether a TCR Cell would react to a specific epitope, while keeping the missing alpha part of the TCR cell into account.

## ESP description
The [esp](https://esp.uantwerpen.be/) contains possible thesis subjects (only accessible using UAntwerp VPN). [This](https://esp.uantwerpen.be/project-page?project_id=218) is the description of my project:

Together with the Vaccine and Infectious Disease Institute (VAXINFECTIO) and the Center for Medical Genetics, the Adrem data lab studies the human immune system. This natural defence system of our body relies on the generation of a large and diverse set of biological protein sequences (T cell receptors) that are rwsgi_esponsible for recognising and eliminating any pathological threats. Thanks to high throughput experimental techniques, we are now able to map this diverse set of sequences with unprecedenting detail, typically resulting in the characterisation of millions of unique sequences per individual. the goal of this research is to gain new insights into vaccines, infectious diseases, cancer and auto-immune disorders.

With our research group we are tackling several challenges that will be appropriate for a research project/thesis:
- creation of a novel labeling algorithm to assign immuen cells to ttheir targets
- implement high performant classifiers to diagnose disease
- develop novel feature reduction methods that can group similar immune cells
- generation of novel visualisation methods for high density information

The tools will preferably be implemented in Python.

## Notebook order
In case you want to sequentially read the notebooks, this is the recommended order:
- data.ipynb
- baseline.ipynb

## Data
You can find the datasets in the `data` folder. 

- `data/GILGFVFTL_data.tsv` is a positive dataset (contains TCR cells known to react to the epitope `GILGFVFTL`). Sometimes you have the full TCR cell (alpha+beta part), sometimes you only have the beta part.
  - GeneA: Dit is altijd TRA als er een alpha is
  - CDR3_alfa: Dit is de CDR3 amino zuur sequentie (soms ook junction_aa genoemd) van de alpha
  - TRAV: V gen van de alpha
  - TRAJ: J gen van de alpha
  - MHC A_alfa: Dit is het MHC type van het epitoop (best negeren, altijd hetzelfde hier)
  - Epitope: epitoop naam (altijd hetzelfde hier)
  - Score_alfa: betrouwbaarheids score voor de alpha (mag je negeren)
  - GeneB: Altijd TRB als er een beta is
  - CDR3_beta: Dit is de CDR3 amino zuur sequentie (soms ook junction_aa genoemd) van de beta
  - TRBV: V gen van de beta
  - TRBJ: J gen vande beta
  - MHC A_beta: idem MHC alpha
  - Epitope: Weer een epitoop kolom (dit is omdat er twee files gemerged zijn)
  - Score_beta: Nog een betrouwbaarheids score die je mag negeren
- `background.tsv` contains samples of alpha and beta combination without epitope 
  - Can be used as negative samples, be sure to check alpha and beta is not in the positive dataset (else remove them)
  - Also need to make sure I have an equal distribution of missing alpha and missing beta in both the negative as positive dataset
  - Columns
    - CDR3_alfa
    - TRAV
    - TRAJ
    - CDR3_beta
    - TRBV
    - TRBJ