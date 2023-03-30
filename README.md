# GeNN_Autoparam


GeNN Hyperparameter Automation by Genetic Algorithm


---


```
###################################################################################
#                                                                                 # 
# The BSD Zero Clause License (0BSD)                                              #
#                                                                                 #
# Copyright (c) 2023 Equosile.                                                    #
#                                                                                 #
# Permission to use, copy, modify, and/or distribute this software for any        #
# purpose with or without fee is hereby granted.                                  #
#                                                                                 #
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH   #
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY     #
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,    #
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM     #
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR   #
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR          #
# PERFORMANCE OF THIS SOFTWARE.                                                   #
#                                                                                 #
###################################################################################
```


---


## Introduction


This individual project is inspired by the need of conveying simple, easy and convenient ways of generating neural architectures for GeNN. GeNN (GPU Enhanced Neural Network) is a software architecture of simulating Spiking Neural Networks. The details of GeNN ought to be presented by the official sites.


- https://genn-team.github.io/


On the other hand, the project intoduced on this occasion does not include sophisticated explanations about GeNN. All enquiries such as its installation procedures and more instructions would be resolved within the official documents above (e.g. https://genn-team.github.io/genn/documentation/4/html/index.html).


Aside from the GeNN manuals, the paper for this project is provided by the equosile web page.


- http://www.equosile.uk/data/819G5-ProjectReport-260108.pdf


This README.md is a rough abbreviation from the paper above.


In order to try the features, the main Python script would be executed within an appropriate Python 3 development environment along with decent development preparations of GeNN and PyGeNN.


- e.g.) $> python genn-autoparam.py


In MS Windows environments, this can be from an Anaconda sandbox that is being operated by Visual Studio x64 Native Tools Command Prompt. Then, the outcome would be like the following demo video.


- http://www.equosile.uk/data/AL-260108-demo-rec.mp4


The goal of this project is to demonstrate some promising potentials that can automatically produce meaningful cybernetic initiations for GeNN. The prototype brain models from the Genetic Algorithm of this project can be being adapted throughout the GeNN simulations, changing the neural architecture of the brain, but more specifically focusing on the better adjustment of the GeNN hyperparameters.


---


## Method


The GeNN technology (Nowotny et al., 2009) prepares initial agent populations. Each genotype of each agent in a certain population is evolved by the Genetic Algorithm (Harvey, 1996). In terms of the Genetic Algorithmic approaches, the fixed chance of uniform crossover (50 percent) is addjusted for the tournaments between winners-versus-losers strategies and then one random gene is mutated. All the random productions of hyperparameters, however, are fulfilled within biological plausibility which lets the variables come up with reasonable coverages.


One simulation cycle takes two individual agents from one population, contesting their classification learning performances, examining their fitness scores as for the outcomes of the comparisons. Winner agent genes override the genes of the loser, overwriting half of them in a manner of uniform crossover. After that the loser agent from that tournament will have one mutation of its genes, going back to the original population with the winner agent. This cycle repeats as many times as possible until the entire population gets composed of many decent agents.


As for the evaluations for fitness scores of Genetic Algorithmic approaches, the purpose is to measure how much the prototype brain of each agent can have proper learning capabilities for certain machine learning classification tasks. The given models of this project provide only simple datasets that can evaluate very simple classification tasks. In the given dataset, there can be three types of distractions and one correct information. Even though there are such 4 kinds of classes, the actual notation would be still a part of simple binary classification about whether a testing image is a distraction or not. Several criteria are set up in the main source codes. Each criterium offers certain amount of fitness score as a compensation of better learning capabilities. The reason that the system checks this ability is to produce decent prototype brain models afterwards as a final goal of this project. Then, the brain outcome can undertake future tasks as well as it used to well deal with the production lines.


---


## Project Aim


The project goal is to quickly produce initial brain sets that can be compatible within GeNN regimes and their future research.


---


## Intermediate Report


- Normally the local optima (around 220) can be reached after 300 natural selection cycles.


- The global optimum is technically 242 (the maximum fitness score).


- It still requires decent CPUs to faster productions.


- For instance, i7-7700 takes 3 to 4 hours to reach the optima.


- If the randomly mutated genes cannot perform any meaningful brain activities, the entire simulation may shut down.


- Since this issue is not common, the project does not offer a sort of hotfixes yet.


---


## Reference


de Belle J.S., Heisenberg, M. (1994). Associative odor learning in Drosophila abolished by chemical


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ablation of mushroom bodies. Science 263: 692–695.


<br>


Dubnau, J., Grady, L., Kitamoto, T., Tully, T. (2001). Disruption of neurotransmission in Drosophila


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; mushroom body blocks retrieval but not acquisition of memory. Nature 411: 476–480.


<br>


Fahrbach, S.E. (1997). Regulation of Age Polyethism in Bees and Wasps by Juvenile Hormone.


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0065-3454/97.


<br>


Hammer, M., Menzel, R. (1998). Multiple sites of associative odor learning as revealed by local brain


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; microinjections of octopamine in honeybees. Learn. Mem. 5: 146–156.


<br>


Harvey, I. (1996). The Microbial Genetic Algorithm. Unpublished report.


<br>


Heisenberg, M. (2003). Mushroom body memoir: From maps to models. Nat. Rev. Neurosci.


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4: 266–275.


<br>


Huerta, R., Nowotny, T. (2009). Fast and Robust Learning by Reinforcement Signals: Explorations in


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; the Insect Brain. Neural Computation 21, 2123–2151.


<br>


Huerta, R., Nowotny, T., Garcia-Sanchez M, Abarbanel HDI, Rabinovich MI (2004). Learning


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; classification in the olfactory system of insects. Neural Comput 16:1601–1640.


<br>


Knight, J.C., Komissarov, A., Nowotny, T. (2021). PyGeNN: A Python Library for GPU-Enhanced Neural


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Networks. Frontiers in Neuroinformatics 15: 10. doi: 10.3389/fninf.2021.659005.


<br>


Knight, J.C., Nowotny, T. (2018). GPUs Outperform Current HPC and Neuromorphic Solutions in


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; terms of Speed and Energy When Simulating a Highly-Connected Cortical Model.


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; doi: 10.3389/fnins.2018.00941.


<br>


Lake, B.M., Salakhutdinov, R., Tenenbaum, J.B. (2019). The Omniglot challenge: a 3-year progress


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; report. doi: 10.1016/j.cobeha.2019.04.007.


<br>


Masuda-Nakagawa, L.M., Ito, K., Awasaki, T., O’Kane, C.J. (2014). A single GABAergic neuron


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; mediates feedback of odor-evoked signals in the mushroom body of larval Drosophila.


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; doi: 10.3389/fncil.2014.00035.


<br>


Nowotny, T., Huerta, R., Abarbanel, H.D.I., Rabinovich, M.I. (2005). Self-organization in the olfactory


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; system: one shot odor recognition in insects. Biol Cybern (2005) 93: 436–446 DOI


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 10.1007/s00422-005-0019-7.


<br>


Nowotny, T., Rabinovich, M.I., Huerta, R. (2003). Decoding Temporal Information Through Slow


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Lateral Excitation in the Olfactory System of Insects. Journal of Computational Neuroscience


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 15, 271–281, 2003.


<br>


Papadopoulou, M., Cassenaer, S., Nowotny, T., Laurent, G. (2011). Normalization for Sparse Encoding


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; of Odors by a Wide-Field Interneuron. Science 332, 721 (2011) DOI:


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 10.1126/science.1201835.


<br>


Montague, P.R., Dayan, P., Person, C., Sejnowski, T.J. (1995). Bee foraging in uncertain environments


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; using predictive Hebbian learning. Nature, 377(6551), 725–728.


<br>


Wang, Y., Wright, J.D., Guo, H.-F., Zuoping, X., Svoboda, K., Malinow, R., Smith, D.P., & Zhong, Y.


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2001). Genetic manipulation of the odor-evoked distributed neural activity in the Drosophila


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; mushroom body. Neuron, 29, 267–276.


<br>


Yavuz, E., Turner, J., Nowotny, T. (2015). GeNN: a code generation framework for accelerated brain


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Simulations. 6:18854 DOI: 10.1038/srep18854.


---
