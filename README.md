# Capstone_HKSA

A Hybrid Keshtel Simulated Annealing and Game Model approch for designing close-loop supply chain design for reusable and recycle dairy packaging, with considerations of both cost-effectiveness and CO2 Emission. 

Abstract: A case study from a dairy manufacturer is invested in developing a supply chain network design for both products and packaging. The packaging includes carton boxes, glass bottles, and plastic bottles, in which carton boxes and plastic bottles can be remanufactured, and qualified glass will be cleaned and reused. The goal of the study is to find the most suitable locations for allocating distribution and collection centers which will minimize all transporation, purchasing and manufacturing cost. The project applied Hybrid Keshtel and Simulated Annealing algorithm, along with intensive usage of game model to balance the benefits of both Company and suppliers, and Fuzzy Compromise method to find a solution which both cost-effective and minimized-CO2 emssion.
The results demonstrate the practicality and efficiency of our proposed model, which create a suitable network for the CLSC of packaging for dairy products.

PPT: [https://drive.google.com/drive/folders/1aZxom3_2tS2jTI1yyAp9YTJvg-cg26my](url)

Repository usage:
- Folder HKSA: Python code for near-optimal solution, large scale problem
1. Run the Cost_game file -> Record the Minimized Cost and Maximized CO2 emission
2. Run the Emission file -> Record the Minimized Emission and Maximized Cost
3. Input above result into Compromise file -> Get the optimal network

The Cost_Nogame is used for validation.

- Folder MILP: CPLEX Code for optimal solution, small scale problem
/The running procedure is the same as HKSA.
