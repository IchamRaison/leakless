# Experimental audio examples

Three unchanged, one-second WAV recordings from the **train** fold of the frozen
`split_v2.csv`. These are demonstration inputs, not held-out evaluation examples.

Source: [Zenodo 18631450](https://zenodo.org/records/18631450),
DOI [10.5281/zenodo.18631450](https://doi.org/10.5281/zenodo.18631450).

Authors: Qi Wang, Zhongyi Mei, Fan Zhan, Jiongxi Chen, Guangdong University of Technology.

Dataset title: *Acoustic data for the manuscript entitled 'Self-supervised acoustic
leakage detection for water distribution systems: A real-time diagnosis framework
under data scarcity'*. License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The WAV bytes have not been modified. Their filenames use opaque dataset clip IDs.
Attribution, input/file hashes, dependency clusters, split identity and display
categories are recorded in `frontend/src/demo/recordings.json`.

Building coordinates and measurement-point associations are illustrative demo
choices. They do not come from the dataset and are not a leak-localization result.
The example labels are never sent to inference; the prediction contract is binary.
