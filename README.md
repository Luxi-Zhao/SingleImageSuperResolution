# Single Image Super Resolution Techniques: Ablation and Integration Studies of Residual Dense Network (RDN) and Densely Residual Laplacian Network (DRLN)

---
### Ao Tang
### Luxi Zhao

---

This repository includes the implementation of RDN and DRLN, as well as the four testing datasets: BSD100, Set14, Set5, Urban100. 
Code is adapted from [here](https://github.com/saeed-anwar/DRLN).

You can read the project report [here](Report.pdf).

We provided all the necessary scripts to reproduce our experiments:

- [Ablation Scripts](ablation_script) to train the ablation models for RDN and DRLN.
- [Integration Scripts](integration_script) to train the integration models for RDN and DRLN.
- [Test Scripts](test_script) to run test the specified models on the four testing datasets
- [Plot Scripts](plot_script) to plot the data

All the python notebook can be executed in the Google Colab environment.
## Major Components:
RDN:
- LRL: Local Residual Learning
- CM: Contiguous Memory
- GFF: Global Feature Fusion

DRLN:
- SSC: Short Skip Connection
- LSC: Long Skip Connection
- LA: Laplacian Attention

## Ablation Studies
The baseline versions for RDN and DRLN can be run with the master branch.

We have different branches for running different ablation studies:

RDN:
- rdn-ablate-lrl-gff: Ablate LRL, GFF
- rdn-ablate-cm-gff: Ablate CM, GFF
- rdn-ablate-cm-lrl: Ablate CM, LRL
- rdn-ablate-gff: Ablate GFF
- rdn-ablate-lrl: Ablate LRL
- rdn-ablate-cm: Ablate CM
- rdn-ablate-cm-lrl-gff: Ablate LRL, GFF, CM

DLRN:
- drln-no-medium-no-laplacian: Ablate SSC, LA
- drln-no-medium-skip: Ablate SKC
- drln-no-medium-no-long: Ablate SSC, LSC
- drln-no-long-skip: Ablate LSC
- drln-no-long-no-laplacian: Ablate LSC, LA
- drln-no-laplacian: Ablate LA
- drln-no-everything: Ablate LA, LSC, SSC

## Integration Studies
We have different branches for running integration studies:

RDN:
- rdn-shortskipconn: Integrate with SSC
- rdn-laplacian: Integrate with LA
- rdn-residual-block: Integrate with RB
- rdn-cascading-block: Integrate with CB

DRLN
- drln-gff: Integrate with GFF