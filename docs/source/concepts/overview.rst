Overview
========

.. seealso::
   Read the associated book `Identification, Characterization, and Manipulation of Neuronal Ensembles <https://link.springer.com/book/10.1007/978-1-0716-4208-5>`_ 
   for a better understanding of neuronal ensembles.

Neuronal ensembles are groups of neurons with coordinated activity that are related to a specific brain function. We present a unified graphical user interface to identify and compare neuronal ensembles (ENCORE) from neuronal activity data obtained through two-photon microscopy or electrophysiological recordings, using five different algorithms. Behavioral and stimulation data can be used to select the algorithm that best represents the functional relevance of each neuronal ensemble. 

Recent advances on simultaneous recordings of hundreds of neurons have allowed the development of complex and accurate models of brain functions. Some of these functions can be explained by groups of neurons with coordinated activity within a defined time window, known as neuronal ensembles. However, a standardized method to identify these neuronal ensembles across different research groups is still lacking. Various approaches have been proposed to find neuronal ensembles, each based on different principles and tested in different animal models and brain regions. Among the different methods are:

Identification by Singular Value Decomposition (SVD) in two-photon calcium recordings of the mouse visual cortex, described by (Carrillo-Reid et al., 2016; Velazquez-Contreras & Carrillo-Reid, 2025). Identification by Independent Component Analysis (ICA), proposed by (Lopes-dos-Santos et al., 2013) in the rat hippocampus using single-unit recordings. Identification using Xsembles2P, proposed by (Pérez-Ortega et al., 2024) in volumetric two-photon recordings of the mouse visual cortex. Identification based in Similarity Graph Clustering (SGC), proposed by (Avitan et al., 2017) and implemented in calcium imaging data from the zebrafish optic tectum. Identification based in Principal Component Analysis (PCA), proposed by (Herzog et al., 2021) using spike train data from multi-electrode array recordings of ganglion cells. 

Although these methods are useful in different scenarios, they may be difficult and time consuming to implement individually. The specific implementation of each method requires different formatting for the input data and produces different outputs, that complicates the comparison of results to select the best approach. A unified method for the identification of neuronal ensembles would increase the availability of these techniques to more researchers. If such standardization is implemented through a graphical user interface (GUI) that includes multiple visualizations, its accessibility would further increase, reaching even non-programmer users. ENCORE (Ensembles Comparison and Recognition) paves the way towards a more standardized approach for identifying neuronal ensembles and understanding their relevance in behavior and cognition.

Included algorithms
-------------------

- SVD based method: Velazquez-Contreras, R., Carrillo-Reid, L. (2025). Identification of Neuronal Ensembles from Similarity Maps Using Singular Value Decomposition. In: Carrillo-Reid, L. (eds) Identification, Characterization, and Manipulation of Neuronal Ensembles. Neuromethods, vol 215. Humana, New York, NY. https://doi.org/10.1007/978-1-0716-4208-5_5
- PCA based method: Herzog et al. (2021) "Scalable and accurate automated method for neuronal ensemble detection in spiking neural networks. https://pubmed.ncbi.nlm.nih.gov/34329314/ Rubén Herzog Dec 2021
- ICA based method: Lopes-dos-Santos V, Ribeiro S, Tort AB (2013) Detecting cell assemblies in large neuronal populations. J Neurosci Methods 220(2):149-66. 10.1016/j.jneumeth.2013.04.010
- Xsembles2P method: Pérez-Ortega, J., Akrouh, A. & Yuste, R. (2024). Stimulus encoding by specific inactivation of cortical neurons. Nat Commun 15, 3192. doi: 10.1038/s41467-024-47515-x
- Similarity Graph Clustering method: L. Avitan et al. "Spontaneous Activity in the Zebrafish Tectum Reorganizes over Development and Is Influenced by Visual Experience". Curr. Biol. 27 (2017). DOI: 10.1016/j.cub.2017.06.056

