# COVID OUT!
This repository contains a implementation of graph informational spreading model
which simulates an illness propagation. Data is being generated in silulation of 
 interactions between people. Internal states of people are being described by 
 significant symptoms of CoVid19 ilness. This model has been developed after
 medical consultations and it is an entry for machine learning pipeline (after
 feeding with real data).
 
DOCUMENTATION IS [HERE](https://anty-filidor.github.io/covidout/#)

### Structure of repository

├── data_generation  
│   ├── databases.example.json  
│   ├── fill_database.py  
│   ├── gen_database.py  
│   ├── gen_graphs.py  
│   ├── graphs  
│   ├── names  
│   ├── simulation.py  
│   └── visualize_movement_simulation.py  
├── networks  
│   ├── edg2.csv  
│   ├── experiment.gif   
│   └── nod2.csv  
├── readme.md  
└── spreading_model  
    ├── ioops.py  
    ├── spreading.py  
    ├── src.ipynb  
    └── visualisations.py  


## How to run this code  

### Data generation
Michał - uzupełnij proszę!

### Spreading model
Please use `src.ipynb` notebook to perform an experiment. It contains 
* read of network
* experiment
* saving results as json of internal states of nodes or gif: 

![gif](networks/experiment.gif)  

Model bases on two probabilities:
- internal probability of being ill of each node. This function is a sigmoid of
 13 different medical symptoms (see `ioops.py_comp_internal_weights`)
- external probability of illness transfer  by interpersonal contact. It is
stored in edges of graph model.
From hat coefficients a resultant probability is being computed by using
conditional probability laws.
