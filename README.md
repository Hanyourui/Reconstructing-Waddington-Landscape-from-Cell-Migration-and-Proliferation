# Reconstructing-Waddington-Landscape-from-Cell-Migration-and-Proliferation
The Waddington landscape was initially proposed to depict the process of cell differentiation, and have been extended to explain phenomena such as reprogramming. The landscape serves as a concrete representation of cellular differentiation potential, yet the precise representation of this potential remains an unsolved problem, posing significant challenges to reconstructing the Waddington landscape. Consequently, it was once regarded merely as a vivid metaphor. The characterization of cellular differentiation potential relies on transcriptomic signatures of known markers typically. Subsequently, numerous computational models based on various energy indicators, such as Shannon entropy, signaling entropy, and Hopfield energy, have been proposed. While these models can effectively characterize cellular differentiation potential, most of them lack corresponding dynamical interpretations, which are crucial for enhancing our understanding of cell fate transitions.
To define an energy indicator based on continuous cellular dynamics, the cell differentiation potential was dissected into the potential of cell migration and proliferation, and a feasible computational framework was developed to reconstruct Waddington landscape based on sparse autoencoder and reaction diffusion advection equation. Firstly, a feature selection approach was employed for the cell type-specific discrete data with pseudo time series, which was based on the identification of highly variable genes. Secondly, a deep learning method based on sparse autoencoder and reaction diffusion advection equation was introduced to reconstruct state-continuous cellular dynamics from the perspective of cell migration and proliferation, which is inspired by the TIGON. Subsequently, the energy indicator $U_{m}$ was defined to characterize the potential of cell migration inherent in the cell type-specific gene regulatory network learned by the GRN sparse autoencoder. And the energy indicator $U_{p}$ was defined to characterize the potential of cell proliferation inherent in the cell type-specific growth function learned by the BRD sparse autoencoder. Finally, the differentiation potential $U$ of different cell types is the sum of their cell migration potential and cell proliferation potential, and the landscape elevation can be defined by the differentiation potentials of the corresponding cell state in both processes.
