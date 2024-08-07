Stored in this repository are the most relevant scripts used for the results obtained in my master thesis report.

The characterization of the shape of the central sulcus is performed using an Isomap dimensionality reduction and a direct geometrical measurement of each sulcus volume.

In Isomap calculations, the computation follows this sequence of script run:
1. calc_dissimilarity.py
2. get_dissim_matrix.py
3. isomap_choose_k.ipynb
4. calc_isomap_and_neutral_pos.py
5. isomap_wind_visualization.py


In the direct geometrical measurements, the order of computation follows:
1. SulcusParameterization2022.py (under BrainVisa processes)
2. build_param_df_visualiz.ipynb


The statistical analysis is performed mostly on these scripts:
1. analysis_parameterization.ipynb
2. analysis_isomap.ipynb
