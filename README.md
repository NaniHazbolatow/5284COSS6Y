# **Complex System Simulation Project**  
**Simulating a Coupled Stock Market to Analyze the Effect of Information Delay on Bubble Formation**  

### **Model Structure**  

The model is implemented in Python. The base model is sourced from the following article:  
[Stochastic Cellular Automata Model for Stock Market Dynamics](https://www.researchgate.net/publication/8537791_Stochastic_Cellular_Automata_Model_for_Stock_Market_Dynamics).  

The **base model** elements can be found in the following directory:  
```
CAStockPriceModel/model/base_model
```  

We developed the coupling component of the self-organizing model to analyze our research topic. Details can be found in the notes, presentation, and function descriptions within the module:  
```
CAStockPriceModel/model/trading_strat/stochastic_dyn.py
```  

### **Running the Model**  

The model results can be observed by running the different modules in the following directory:  
```
CAStockPriceModel/model/analysis
```  

### **Authors**  

- Marcell Szegedi  
- Ernani Hazbolatow  
- Michelangelo Grasso  
- Alexis Ledru  
