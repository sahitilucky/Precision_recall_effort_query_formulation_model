# Precision_recall_effort_query_formulation_model

This is the code repository for the PRE model - Precision-Recall-Effort query optimization framework. 

**Description:** Precision-Recall-Effort query framework is a query formulation and reformulation using Precision-Recall-Effort framework by maximizing the estimated recall and precision of the retrieval results and minimizing the effort for making the query. 


### Paper

Labhishetty et al., In Proceedings of the 2022 ACM SIGIR ICTIR, _PRE: A Precision-Recall-Effort Optimization Framework for Query Simulation_, https://dl.acm.org/doi/10.1145/3539813.3545136


### Contents

`Code/` : to simulate queries for TREC session track tasks and to simulate session through query reformulations using PRE, to evaluate simulated queries/sessions using TREC user queries and sessions. 

`Code_CCQF_model/` : to simulate queries using cognitive and communication query formulation framework - first version of PRE, optimizing precision-recall together and minimizing effort.

`figures/` : sensitivity of effort, recall and precision parameters.

`recall_precision_hists/` : Example histograms of recall and precision scores for generated queries of topics.

### Research Usage

If you use our work in your research please cite:

```
@inproceedings{10.1145/3539813.3545136,
author = {Labhishetty, Sahiti and Zhai, ChengXiang},
title = {PRE: A Precision-Recall-Effort Optimization Framework for Query Simulation},
year = {2022},
isbn = {9781450394123},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3539813.3545136},
doi = {10.1145/3539813.3545136},
booktitle = {Proceedings of the 2022 ACM SIGIR International Conference on Theory of Information Retrieval},
pages = {51–60},
numpages = {10},
keywords = {knowledge state, query simulation, formal interpretable framework},
location = {Madrid, Spain},
series = {ICTIR '22}
}
```

### License

By using this source code you agree to the license described in https://github.com/sahitilucky/Precision_recall_effort_query_formulation_model/blob/master/LICENSE.md.




