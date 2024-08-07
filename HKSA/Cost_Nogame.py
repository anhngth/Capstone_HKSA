import numpy as np
import pandas as pd
import random, math
from itertools import product
from mip import *
import xlwings as xw
import time 

#SET UP
# Case
K = 3

P = 1
D = 6
C = 9
W = 9
T = 7
I = 18
F = 100
Nkeshtel = 30

Product_in_SCTN    = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} 
Product_in_Plastic = {10, 11, 12, 13, 14} 
Product_in_Glass   = {15, 16, 17}

#Algorithm
GBest = 10**30
Max_main_iter = 100

SA_best_solution     = Nkeshtel
SA_current_solution  = Nkeshtel+1
SA_neighbor_solution = Nkeshtel+2

p_N1 = 0.2
p_N2 = 0.3
N1 = round(Nkeshtel*p_N1)
N2 = round(Nkeshtel*p_N2)
N3 = Nkeshtel - (N1 + N2)

#FUNCTIONS
Direct   = Product_in_Glass #Ship directly from Plant to customer due to short life cycle time
Indirect = Product_in_SCTN.union(Product_in_Plastic) #Ship either directly or indirectly to customer due to long life cycle time

Recycled_packaging = {0, 1}
Reused_packaging = {2}

def load(defined_name):
    wb = xw.Book(r"D:\D\University\4th year\1. Capstone\Capstone - Code\After submit\Largescale_11.xlsx")
    named_range = wb.names[defined_name]
    data = named_range.refers_to_range.value
    df = pd.DataFrame(data)
    return df

print("cost")
# 1.Cost
Fc_d_small          = np.squeeze(load("Fc_d_small").values)
Fc_d_medium         = np.squeeze(load("Fc_d_medium").values)
Fc_d_large          = np.squeeze(load("Fc_d_large").values)

size_d_small        = np.squeeze(load("size_d_small").values)
size_d_medium       = np.squeeze(load("size_d_medium").values)

Fc_c_small          = np.squeeze(load("Fc_c_small").values)
Fc_c_medium         = np.squeeze(load("Fc_c_medium").values)
Fc_c_large          = np.squeeze(load("Fc_c_large").values)

size_c_small        = np.squeeze(load("size_c_small").values)
size_c_medium       = np.squeeze(load("size_c_medium").values)

vb_p                = np.reshape(load("vb_p").values, (P, I))
vb_d                = np.reshape(load("vb_d").values, (D, I))
vb_c                = np.reshape(load("vb_c").values, (C, K))
vb_w                = np.reshape(load("vb_w").values, (W, K))
Pur_RM              = np.squeeze(load("Pur_RM").values)  
Pur_PCK             = np.squeeze(load("Pur_PCK").values)   
Required_percentage = np.squeeze(load("Required_percentage").values)                 
vr                  = np.squeeze(load("vr").values)
Cost_pd             = np.squeeze(load("Cost_pd").values)
Cost_pf             = np.squeeze(load("Cost_pf").values)
Cost_df             = np.squeeze(load("Cost_df").values)
Cost_dc             = np.squeeze(load("Cost_dc").values)
Cost_fc             = np.squeeze(load("Cost_fc").values)
Cost_cw             = np.squeeze(load("Cost_cw").values)
Cost_cp             = np.squeeze(load("Cost_cp").values)
distance_pd         = np.reshape(load("distance_pd").values, (P,D))
distance_pf         = np.reshape(load("distance_pf").values, (P,F))
distance_df         = np.reshape(load("distance_df").values, (D,F))
distance_dc         = np.reshape(load("distance_dc").values, (D,C))
distance_fc         = np.reshape(load("distance_fc").values, (F,C))
distance_cw         = np.reshape(load("distance_cw").values, (C,W))
distance_cp         = np.reshape(load("distance_cp").values, (C,P))

Tc_pd               = np.zeros((P, D, I))
Tc_pf               = np.zeros((P, F, I))
Tc_df               = np.zeros((D, F, I))
Tc_dc               = np.zeros((D, C, K))
Tc_fc               = np.zeros((F, C, K))
Tc_cw               = np.zeros((C, W, K))
Tc_cp               = np.zeros((C, P, K))
Oc                  = np.zeros((F, I))

for p, d, f, c, w, i, k in product(range(P), range(D), range(F), range(C), range(W), range(I), range(K)):
    Tc_pd[p][d][i]  = Cost_pd[i] * distance_pd[p][d]
    Tc_pf[p][f][i]  = Cost_pf[i] * distance_pf[p][f]
    Tc_df[d][f][i]  = Cost_df[i] * distance_df[d][f]
    Tc_dc[d][c][k]  = Cost_dc[k] * distance_dc[d][c]
    Tc_fc[f][c][k]  = Cost_fc[k] * distance_fc[f][c]
    Tc_cw[c][w][k]  = Cost_cw[k] * distance_cw[c][w]
    Tc_cp[c][p][k]  = Cost_cp[k] * distance_cp[c][p]
    Oc[f][i]        = (vb_p[0][i] + Tc_pf[0][f][i])*1.15

# 2. CO2 Emission
print("emission")
Es_p                = np.reshape(load("Es_p").values, (P, I))
Es_d                = np.reshape(load("Es_d").values, (D, I))
Es_c                = np.reshape(load("Es_c").values, (C, K))
Es_w                = np.reshape(load("Es_w").values, (W, K))
Esr                 = np.squeeze(load("Esr").values)
Esm                 = np.squeeze(load("Esm").values)

Emission_pd         = np.squeeze(load("Emission_pd").values)
Emission_pf         = np.squeeze(load("Emission_pf").values)
Emission_df         = np.squeeze(load("Emission_df").values)
Emission_dc         = np.squeeze(load("Emission_dc").values)
Emission_fc         = np.squeeze(load("Emission_fc").values)
Emission_cw         = np.squeeze(load("Emission_cw").values)
Emission_cp         = np.squeeze(load("Emission_cp").values)

Ets_pd              = np.zeros((P, D, I))
Ets_pf              = np.zeros((P, F, I))
Ets_df              = np.zeros((D, F, I))
Ets_dc              = np.zeros((D, C, K))
Ets_fc              = np.zeros((F, C, K))
Ets_cw              = np.zeros((C, W, K))
Ets_cp              = np.zeros((C, P, K))
Eos                 = np.zeros((F, I))

for p, d, f, c, w, i, k in product(range(P), range(D), range(F), range(C), range(W), range(I), range(K)):
    Ets_pd[p][d][i] = Emission_pd[i] * distance_pd[p][d]
    Ets_pf[p][f][i] = Emission_pf[i] * distance_pf[p][f]
    Ets_df[d][f][i] = Emission_df[i] * distance_df[d][f]
    Ets_dc[d][c][k] = Emission_dc[k] * distance_dc[d][c]
    Ets_fc[f][c][k] = Emission_fc[k] * distance_fc[f][c]
    Ets_cw[c][w][k] = Emission_cw[k] * distance_cw[c][w]
    Ets_cp[c][p][k] = Emission_cp[k] * distance_cp[c][p]
    Eos[f][i]       = (Es_p[0][i] + Ets_pf[0][f][i])*0.8

# Other parameter
Dmd                 = np.reshape(load("Dmd").values, (I, T, F))
nonzeros_Dmd        = np.nonzero(Dmd)
theta               = np.squeeze(load("theta").values)
phi                 = np.squeeze(load("phi").values) 
gamma               = np.squeeze(load("gamma").values)
Def                 = np.squeeze(load("Def").values)
Qal                 = np.squeeze(load("Qal").values)
omega               = np.squeeze(load("omega").values)
beta                = np.squeeze(load("beta").values)
Pck_f               = np.zeros((F, K, T))

CP_p                = 12000000
CP_d                = np.squeeze(load("CP_d").values)
CP_c                = np.squeeze(load("CP_c").values)
CP_w                = np.squeeze(load("CP_w").values)

discount_list       = [[0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20], 
                       [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]]

#delete to free up memory
del distance_pd, distance_cp, 
Cost_pd, Cost_pf, Cost_df, Cost_dc, Cost_fc, Cost_cw, Cost_cp,
Emission_pd, Emission_pf, Emission_df, Emission_dc, Emission_fc, Emission_cw, Emission_cp

#Variables
Bd                  = np.zeros((D, Nkeshtel+3))
Bc                  = np.zeros((C, Nkeshtel+3))
Qp                  = np.zeros((P, I, T, Nkeshtel+3))
Qd                  = np.zeros((D, I, T, Nkeshtel+3))
Qc                  = np.zeros((C, K, T, Nkeshtel+3))
Qw                  = np.zeros((W, K, T, Nkeshtel+3))
RM                  = np.zeros((K, T, Nkeshtel+3)) 
NB                  = np.zeros((P, T, Nkeshtel+3))
Qof                 = np.zeros((F, I, T, Nkeshtel+3))
Qt_pd               = np.zeros((P, D, I, T, Nkeshtel+3))
Qt_pf               = np.zeros((P, F, I, T, Nkeshtel+3))
Qt_df               = np.zeros((D, F, I, T, Nkeshtel+3))
Qt_dc               = np.zeros((D, C, K, T, Nkeshtel+3)) 
Qt_fc               = np.zeros((F, C, K, T, Nkeshtel+3)) 
Qt_cw               = np.zeros((C, W, K, T, Nkeshtel+3)) 
Qt_cp               = np.zeros((C, P, K, T, Nkeshtel+3)) 

QP                  = np.zeros((P, K, T, Nkeshtel+3))
QP_nextperiod       = np.zeros((P, K, Nkeshtel+3))
Pck_d               = np.zeros((D, K, T, Nkeshtel+3))

print("checkk")
#proportion
proportion_Qt_df    = np.zeros([D, F, I, T, Nkeshtel+3])
proportion_Qt_pf    = np.zeros([P, F, I, T, Nkeshtel+3])
proportion_Qt_dc    = np.zeros([D, C, K, T, Nkeshtel+3])
proportion_Qt_fc    = np.zeros([F, C, K, T, Nkeshtel+3])
proportion_Qt_cw    = np.zeros([C, W, K, T, Nkeshtel+3])

#fitness (cost)
fitness             = np.zeros((Nkeshtel+3))
company_cost        = np.zeros((Nkeshtel+3))
cost_at_facilities  = np.zeros((Nkeshtel+3))
packaging_cost      = np.zeros((Nkeshtel+3))
outsource_cost      = np.zeros((Nkeshtel+3))
transport_cost      = np.zeros((Nkeshtel+3))
cleaning_cost       = np.zeros((Nkeshtel+3))
supplier_cost       = np.zeros((2, Nkeshtel+3))
raw_material_cost   = np.zeros((2, Nkeshtel+3))
recycle_cost        = np.zeros((2, Nkeshtel+3))
revenue_loss        = np.zeros((2, Nkeshtel+3))
discount_rate       = np.zeros((2, Nkeshtel+3))
original_Qrecycling = np.zeros((T))

required_capacity_D = np.zeros((T))
required_capacity_C = np.zeros((T))

for t in range(T):
    required_capacity_D[t] = sum(Dmd[i][t][f] for f,i in product(range(F), Indirect))
    for f in range(F):
            #packaging required to collect from customer (carton, glass, plastic)
            Pck_f[f][0][t] = (sum(Dmd[i][t][f]*theta[0]*phi[i]   for i in Product_in_SCTN)
                               + sum(Dmd[i][t][f]*theta[0]*gamma[i] for i in Product_in_SCTN) 
                               + sum(Dmd[i][t][f]*theta[1]*gamma[i] for i in Product_in_Plastic) 
                               + sum(Dmd[i][t][f]*theta[2]*gamma[i] for i in Product_in_Glass))
            Pck_f[f][1][t] =  sum(Dmd[i][t][f]*theta[1]*phi[i] for i in Product_in_Plastic)
            Pck_f[f][2][t] =  sum(Dmd[i][t][f]*theta[2]*phi[i] for i in Product_in_Glass)
    
    required_capacity_C[t] = sum(Pck_f[f][k][t] for f, k in product(range(F), range(K))) + sum(Dmd[i][t][f] / (1 - Def[i]) for f, i in product(range(F), Indirect))

def round_up(number):
    if number > 0:
        multiplier = 10 ** 3
        return math.ceil(number * multiplier) / multiplier
    else:
        return number

def random_binary(n):
    sum_capacity_D = 0
    while all(sum_capacity_D < required_capacity_D[t] for t in range(T)):
        for d in range(D):
            Bd[d][n] = random.randint(0,1)
        sum_capacity_D = sum(CP_d[d] for d in range(D) if Bd[d][n] == 1)

    sum_capacity_C = 0
    while all(sum_capacity_C < required_capacity_C[t] for t in range(T)):
        for c in range(C):
            Bc[c][n] = random.randint(0,1)
        sum_capacity_C = sum(CP_c[c] for c in range(C) if Bc[c][n] == 1)

    return

def random_proportion (n): 
    #clear old proportion
    proportion_Qt_pf[..., n].fill(0)
    proportion_Qt_df[..., n].fill(0)
    proportion_Qt_dc[..., n].fill(0)
    proportion_Qt_fc[..., n].fill(0)
    proportion_Qt_cw[..., n].fill(0)
    Qof[..., n].fill(0)

    openD = [d for d in range(D) if Bd[d][n] == 1]
    openC = [c for c in range(C) if Bc[c][n] == 1]
    # Forward (DF & PF)
    for i, t, f in zip(nonzeros_Dmd[0], nonzeros_Dmd[1], nonzeros_Dmd[2]):
        if i in Direct:
            proportion_Qt_pf[p][f][i][t][n] = 1
        if i in Indirect:
            for d in openD:
                proportion_Qt_df[d][f][i][t][n] = (np.random.uniform(0, 1))
                    
            sum_proportion_Qt_df = sum([proportion_Qt_df[d][f][i][t][n] for d in openD])
            for d in openD:
                proportion_Qt_df[d][f][i][t][n] = round_up(proportion_Qt_df[d][f][i][t][n]/(sum_proportion_Qt_df))

            sum_proportion_Qt_df = sum(proportion_Qt_df[d][f][i][t][n] for d in openD)
            if sum_proportion_Qt_df != 1:
                for d in openD:
                    if proportion_Qt_df[d][f][i][t][n] >= 0:
                        proportion_Qt_df[d][f][i][t][n] += (1 - sum_proportion_Qt_df)
                        break

    for t in range(T):
        # For DC to CC
        for d, k in product(openD, range(K)):
            for c in openC:
                proportion_Qt_dc[d][c][k][t][n] = np.round(np.random.uniform(0, 1),2)

            sum_proportion_Qt_dc = sum(proportion_Qt_dc[d][c][k][t][n] for c in range(C))
            for c in openC:
                if sum_proportion_Qt_dc != 0:
                    proportion_Qt_dc[d][c][k][t][n] = round_up(proportion_Qt_dc[d][c][k][t][n] / (sum_proportion_Qt_dc)) 
            
            sum_proportion_Qt_dc = sum(proportion_Qt_dc[d][c][k][t][n] for c in range(C))
            if sum_proportion_Qt_dc != 1:
                for c in openC:
                    if proportion_Qt_dc[d][c][k][t][n] >= 0:
                        proportion_Qt_dc[d][c][k][t][n] += (1 - sum_proportion_Qt_dc) 
                        break

        # For Customer to CC
        for f, k in product(range(F), range(K)):
            for c in openC:
                proportion_Qt_fc[f][c][k][t][n] = np.round(np.random.uniform(0, 1),2)

            sum_proportion_Qt_fc = sum(proportion_Qt_fc[f][c][k][t][n] for c in range(C))
            for c in openC:
                if sum_proportion_Qt_fc != 0:
                    proportion_Qt_fc[f][c][k][t][n] = round_up(proportion_Qt_fc[f][c][k][t][n] / (sum_proportion_Qt_fc)) 
            
            sum_proportion_Qt_fc = sum(proportion_Qt_fc[f][c][k][t][n] for c in range(C))
            if sum_proportion_Qt_fc != 1:
                for c in openC:
                    if proportion_Qt_fc[f][c][k][t][n] >= 0:
                        proportion_Qt_fc[f][c][k][t][n] += (1 - sum_proportion_Qt_fc)
                        break

        # For CC to Disposal
        for c, k in product(openC, range(K)):
            for w in range(W):
                proportion_Qt_cw[c][w][k][t][n] = np.round(np.random.uniform(0, 1),2)

            sum_proportion_Qt_cw = sum(proportion_Qt_cw[c][w][k][t][n] for w in range(W))
            for w in range(W):
                if sum_proportion_Qt_fc != 0:
                    proportion_Qt_cw[c][w][k][t][n] = round_up(proportion_Qt_cw[c][w][k][t][n] / (sum_proportion_Qt_cw))
            
            sum_proportion_Qt_cw = sum(proportion_Qt_cw[c][w][k][t][n] for w in range(W))
            if sum_proportion_Qt_cw != 1:
                for w in range(W):
                    if proportion_Qt_cw[c][w][k][t][n] >= 0:
                        proportion_Qt_cw[c][w][k][t][n] += (1 - sum_proportion_Qt_cw)
                        break
    return

def clear_variables (n):
    Qp              [..., n].fill(0)
    Qd              [..., n].fill(0)
    Qc              [..., n].fill(0)
    Qw              [..., n].fill(0)
    Qt_pd           [..., n].fill(0)
    Qt_pf           [..., n].fill(0)
    Qt_df           [..., n].fill(0)
    Qt_dc           [..., n].fill(0)
    Qt_fc           [..., n].fill(0)
    Qt_cp           [..., n].fill(0)
    Qt_cw           [..., n].fill(0)
    QP              [..., n].fill(0)
    QP_nextperiod   [..., n].fill(0)
    Pck_d           [..., n].fill(0)
    RM              [..., n].fill(0)
    NB              [..., n].fill(0)
    discount_rate   [..., n].fill(0)
    return

def segment_P (n): 
    gap_p = np.zeros((P))
    openD = [d for d in range(D) if Bd[d][n] == 1]
    for t in range(T):
        #calculate Qp based on proportion
        for p, i in product(range(P), range(I)):
            Qp[p][i][t][n] = sum(proportion_Qt_pf[p][f][i][t][n]*Dmd[i][t][f] for f in range(F)) + sum(proportion_Qt_df[d][f][i][t][n]*Dmd[i][t][f] for d, f in product(openD, range(F)))/ (1 - Def[i])
        
        #correction 
        for p in range(P): 
            gap_p[p] = sum(Qp[p][i][t][n] for i in range(I)) - CP_p
        OverCapacitated_p = [p for p in range(P) if gap_p[p] > 0]

        for p in OverCapacitated_p:
            dummy1 = proportion_Qt_df[..., t, n].copy()
            dummy2 = proportion_Qt_pf[p, :, :, t, n].copy()
            proportion_Qt_df[..., t, n] = 0
            proportion_Qt_pf[p, :, :, t, n] = 0
            current_sum = 0

            sorted_F = sorted(range(F), key=lambda x: distance_pf[0][x])

            for f in sorted_F:
                Dmd_f = Dmd[:, t, f].copy()
                corresponding_i = np.where(Dmd_f != 0)[0]
                for i in corresponding_i:
                    if i in Indirect:
                        for d in openD:
                            A = dummy1[d][f][i]*Dmd[i][t][f]/ (1 - Def[i])
                            if current_sum + A < CP_p:
                                proportion_Qt_df[d][f][i][t][n] = dummy1[d][f][i].copy()
                                current_sum += A
                            else:
                                Qof[f][i][t][n] += dummy1[d][f][i]*Dmd[i][t][f]
                                
                    if i in Direct:
                        if current_sum + dummy2[f][i]*Dmd[i][t][f] < CP_p:
                            proportion_Qt_pf[p][f][i][t][n] = dummy2[f][i].copy()
                            current_sum += dummy2[f][i]*Dmd[i][t][f]
                        else:
                            Qof[f][i][t][n] += dummy2[f][i]*Dmd[i][t][f]
    return 

def segment_D (n):
    openD = [d for d in range(D) if Bd[d][n] == 1]
    gap_d = np.zeros((D))
    for t in range(T):
        #calculate Qd based on proportion
        for d, i in product(openD, Indirect):
            Qd[d][i][t][n] = sum(proportion_Qt_df[d][f][i][t][n] * Dmd[i][t][f] for f in range(F))/ (1 - Def[i])
    
        #correction proportion
        for d in openD:
            gap_d[d] = sum(Qd[d][i][t][n] for i in Indirect) - CP_d[d]
        
        sort_list = [(d, gap_d[d]) for d in openD]

        sorted_gap_d = sorted(sort_list, key = lambda x: x[1], reverse= True)
        OverCapacitated_d  = [d for d, gap in sorted_gap_d if gap > 0]
        UnderCapacitated_d = [d for d, gap in sorted_gap_d if gap <= 0]

        for d_over in OverCapacitated_d:
            dummy = proportion_Qt_df[d_over, :, :, t, n].copy()
            proportion_Qt_df[d_over, :, :, t, n] = 0

            current_sum = 0
            sorted_F = sorted(range(F), key=lambda x: distance_df[d_over][x])

            for f in sorted_F:
                sorted_D = sorted(UnderCapacitated_d, key=lambda x: distance_df[x][f])
                Dmd_f = Dmd[:, t, f].copy()
                corresponding_i = np.where(Dmd_f != 0)[0]
                for i in corresponding_i:
                    if i in Indirect:
                        A = dummy[f][i]*Dmd[i][t][f]/(1 - Def[i]).copy()
                        if current_sum + A < CP_d[d_over]:
                            proportion_Qt_df[d_over][f][i][t][n] = dummy[f][i]
                            current_sum += A
                        else:
                            Found_facility = False
                            for d_under in sorted_D:
                                if A <= -gap_d[d_under]: #since gap under is negative value
                                    proportion_Qt_df[d_under][f][i][t][n] += dummy[f][i]
                                    gap_d[d_under] += A
                                    Found_facility = True
                                    break
                            if Found_facility == False: #break down shipment to scatter in many facilities
                                dummy_dummy =  dummy[f][i].copy()
                                if (-gap_d)[d_over] > 0:
                                    proportion_Qt_df[d_over][f][i][t][n] = (-gap_d[d_over])/Dmd[i][t][f]*(1-Def[i])
                                    dummy_dummy -= (-gap_d[d_over])/Dmd[i][t][f]*(1-Def[i])
                                    gap_d[d_over] = 0
                                for d_under in sorted_D:
                                    if (-gap_d)[d_under] > 0:
                                        if dummy_dummy*Dmd[i][t][f]/(1 - Def[i]) >= (-gap_d[d_under]):
                                            proportion_Qt_df[d_under][f][i][t][n] += (-gap_d[d_under])/Dmd[i][t][f]*(1-Def[i])
                                            dummy_dummy -= (-gap_d[d_under])/Dmd[i][t][f]*(1-Def[i])
                                            gap_d[d_under] = 0
                                        else:
                                            proportion_Qt_df[d_under][f][i][t][n] += dummy_dummy
                                            gap_d[d_under] += dummy_dummy*Dmd[i][t][f]/(1 - Def[i])
                                            dummy_dummy = 0
                                    if dummy_dummy == 0:
                                        break
                                            
    for d in openD:
        for i, t, f in zip(nonzeros_Dmd[0], nonzeros_Dmd[1], nonzeros_Dmd[2]):
            if i in Indirect:
                Qt_df[d][f][i][t][n] = proportion_Qt_df[d][f][i][t][n] * Dmd[i][t][f]
    for d, i, t in product(openD, Indirect, range(T)):
        Qd[d][i][t][n] = sum(Qt_df[d][f][i][t][n] for f in range(F)) / (1-Def[i])
        for p in range(P):
            Qt_pd[p][d][i][t][n] = Qd[d][i][t][n].copy()

    for i, t, f in zip(nonzeros_Dmd[0], nonzeros_Dmd[1], nonzeros_Dmd[2]):
        Qt_pf[0][f][i][t][n] = proportion_Qt_pf[0][f][i][t][n] * Dmd[i][t][f]
    for p, i, t in product(range(P), range(I), range(T)):
        Qp[p][i][t][n] = (sum(Qt_pd[p][d][i][t][n] for d in range(D)) + sum(Qt_pf[p][f][i][t][n] for f in range(F)))

    return

def prepare_for_reverse (n):
    for t in range(T):
        for p in range(P):
            #packaging needed for the production (carton, plastic, glass)
            QP[p][0][t][n]    = sum(Qp[p][i][t][n]*phi[i] for i in Product_in_SCTN)    + sum(Qp[p][i][t][n]*gamma[i] for i in range(I))
            QP[p][1][t][n]    = sum(Qp[p][i][t][n]*phi[i] for i in Product_in_Plastic)
            QP[p][2][t][n]    = sum(Qp[p][i][t][n]*phi[i] for i in Product_in_Glass)
        for d in range(D):
            #packaging required to collect from DC (carton, glass, plastic)
            Pck_d[d][0][t][n] = sum(Qd[d][i][t][n] * Def[i] *phi[i] for i in Product_in_SCTN)    + sum(Qd[d][i][t][n] * Def[i] * gamma[i] for i in range(I)) 
            Pck_d[d][1][t][n] = sum(Qd[d][i][t][n] * Def[i] *phi[i] for i in Product_in_Plastic)
            Pck_d[d][2][t][n] = sum(Qd[d][i][t][n] * Def[i] *phi[i] for i in Product_in_Glass)    
    for p, k in product(range(P), range(K)):
        #FORECAST packaging needed for period T+1
        QP_nextperiod[p][k][n] = np.mean(QP[p, k, :, n]).copy()
        
    return

def segment_C (n): 
    openD = [d for d in range(D) if Bd[d][n] == 1]
    openC = [c for c in range(C) if Bc[c][n] == 1]
    gap_c = np.zeros((C))

    for t in range(T):      
        #calculate Qc based on proportion
        for c, k  in product(openC, range(K)):
            Qc[c][k][t][n] = sum(proportion_Qt_dc[d][c][k][t][n] * Pck_d[d][k][t][n] for d in openD) + sum(proportion_Qt_fc[f][c][k][t][n] * Pck_f[f][k][t] for f in range(F))

        #correction proportion
        for c in openC:
            gap_c[c] = sum(Qc[c][k][t][n] for k in range(K)) - CP_c[c]

        sort_list = [(c, gap_c[c]) for c in openC]
        sorted_gap_c = sorted(sort_list, key = lambda x: x[1], reverse= True)
        OverCapacitated_c  = [c for c, gap in sorted_gap_c if gap > 0]
        UnderCapacitated_c = [c for c, gap in sorted_gap_c if gap <= 0]

        for c_over in OverCapacitated_c:
            dummy1 = proportion_Qt_dc[:, c_over, :, t, n].copy()
            dummy2 = proportion_Qt_fc[:, c_over, :, t, n].copy()
            proportion_Qt_dc[:, c_over, :, t, n] = 0
            proportion_Qt_fc[:, c_over, :, t, n] = 0

            current_sum = 0
            for d in openD:
                sorted_C = sorted(UnderCapacitated_c, key=lambda x: distance_dc[d][x])
                for k in range(K):                    
                    if current_sum + dummy1[d][k] * Pck_d[d][k][t][n] < CP_c[c_over]:
                        proportion_Qt_dc[d][c_over][k][t][n] = dummy1[d][k].copy()
                        current_sum += dummy1[d][k] * Pck_d[d][k][t][n]
                    else:
                        for c_under in sorted_C:
                            if dummy1[d][k] * Pck_d[d][k][t][n] <= -gap_c[c_under]: #since gap under is negative value
                                proportion_Qt_dc[d][c_under][k][t][n] += dummy1[d][k]
                                gap_c[c_under] += dummy1[d][k] * Pck_d[d][k][t][n]
                                break

            sorted_F = sorted(range(F), key=lambda x: distance_fc[x][c_over])
            for f in sorted_F:
                sorted_C = sorted(UnderCapacitated_c, key=lambda x: distance_fc[f][x])
                for k in range(K):                    
                    if current_sum + dummy2[f][k] * Pck_f[f][k][t] < CP_c[c_over]:
                        proportion_Qt_fc[f][c_over][k][t][n] = dummy2[f][k].copy()
                        current_sum += dummy2[f][k] * Pck_f[f][k][t]
                    else:
                        Found_facility = False
                        for c_under in sorted_C:
                            if dummy2[f][k] * Pck_f[f][k][t] <= -gap_c[c_under]: #since gap under is negative value
                                proportion_Qt_fc[f][c_under][k][t][n] += dummy2[f][k]
                                gap_c[c_under] += dummy2[f][k] * Pck_f[f][k][t]
                                Found_facility = True
                                break
                        if Found_facility == False:
                            dummy_dummy =  dummy2[f][k].copy()
                            if (-gap_c)[c_over] > 0:
                                proportion_Qt_fc[f][c_over][k][t][n] = (-gap_c[c_over])/Pck_f[f][k][t]
                                dummy_dummy -= (-gap_c[c_over])/Pck_f[f][k][t]
                                gap_c[c_over] = 0
                            for d_under in sorted_C:
                                if (-gap_c)[c_under] > 0:
                                    if dummy_dummy*Pck_f[f][k][t] >= (-gap_c[d_under]):
                                        proportion_Qt_fc[f][c_under][k][t][n] += (-gap_c[d_under])/Pck_f[f][k][t]
                                        dummy_dummy -= (-gap_c[c_under])/Pck_f[f][k][t]
                                        gap_c[c_under] = 0
                                    else:
                                        proportion_Qt_fc[f][c_under][k][t][n] += dummy_dummy
                                        gap_c[d_under] += dummy_dummy*Pck_f[f][k][t]
                                        dummy_dummy = 0
                                if dummy_dummy == 0:
                                    break
            
        for c, k in product(openC, range(K)):
            for d in openD:
                Qt_dc[d][c][k][t][n] = proportion_Qt_dc[d][c][k][t][n] * Pck_d[d][k][t][n]
            for f in range(F):
                Qt_fc[f][c][k][t][n] = proportion_Qt_fc[f][c][k][t][n] * Pck_f[f][k][t]

            Qc[c][k][t][n] = sum(Qt_dc[d][c][k][t][n] for d in openD) + sum(Qt_fc[f][c][k][t][n] for f in range(F))
    return

def segment_W (n):
    openC = [c for c in range(C) if Bc[c][n] == 1]
    gap_w = np.zeros(W)
    for t in range(T):
        for w, k in product(range(W), range(K)):
        #calulate Qw based on proportion
            Qw[w][k][t][n] = sum(proportion_Qt_cw[c][w][k][t][n]*Qc[c][k][t][n] for c in openC)
    
        #correction proportion
        for w in range(W):
            gap_w[w] = sum(Qw[w][k][t][n] for k in range(K)) - CP_w[w]

        sorted_gap_w = sorted(enumerate(gap_w), key = lambda x: x[1], reverse= True)
        OverCapacitated_w = []
        UnderCapacitated_w = []

        for w, value in sorted_gap_w:
            if value > 0:   OverCapacitated_w.append(w)
            else:           UnderCapacitated_w.append(w)

        for w_over in OverCapacitated_w:
            dummy = proportion_Qt_cw[:, w_over, :, t, n].copy()
            proportion_Qt_cw[:, w_over, :, t, n] = 0

            current_sum = 0
            sorted_C = sorted(openC, key=lambda x: distance_cw[x][w_over])

            for c in sorted_C:
                sorted_W = sorted(UnderCapacitated_w, key=lambda x: distance_cw[c][x])
                for k in range(K):
                    if current_sum + dummy[c][k] * Qc[c][k][t][n] < CP_w[w_over]:
                        proportion_Qt_cw[c][w_over][k][t][n] = dummy[c][k].copy()
                        current_sum += dummy[c][k] * Qc[c][k][t][n]
                    else:
                        for w_under in sorted_W:
                            if dummy[c][k] * Qc[c][k][t][n] < -gap_w[w_under]: #since gap under is negative value
                                proportion_Qt_cw[c][w_under][k][t][n] += dummy[c][k].copy()
                                gap_w[w_under] += dummy[c][k] * Qc[c][k][t][n]
                                break

        for w, k in product(range(W), range(K)):
            for c in openC:
                Qt_cw[c][w][k][t][n] = (1-Qal[k])*Qc[c][k][t][n] * proportion_Qt_cw[c][w][k][t][n]
                Qt_cp[c][0][k][t][n] = Qal[k] * Qc[c][k][t][n]
                
            Qw[w][k][t][n] = sum(Qt_cw[c][w][k][t][n] for c in openC)

    for p, t in product(range(P), range(T)):
        if t != T-1:
            for k in Recycled_packaging:
                RM[k][t][n] = max(QP[p][k][t+1][n]/omega[k] - beta[k]*sum(Qt_cp[c][p][k][t][n] for c in range(C)), 0)
            for k in Reused_packaging:
                NB[p][t][n] = max(QP[p][k][t+1][n] - beta[k]*sum(Qt_cp[c][p][k][t][n] for c in range(C)), 0)  
        if t == T-1:
            for k in Recycled_packaging:
                RM[k][t][n] = max(QP_nextperiod[p][k][n]/omega[k] - beta[k]*sum(Qt_cp[c][p][k][t][n] for c in range(C)), 0)
            for k in Reused_packaging:
                NB[p][t][n] = max(QP_nextperiod[p][k][n] - beta[k]*sum(Qt_cp[c][p][k][t][n] for c in range(C)), 0)  
    return

def correction(n):
    clear_variables(n)
    segment_P(n)
    segment_D(n)
    prepare_for_reverse(n)
    segment_C(n)
    segment_W(n)
    return

def cost(n):
    openD  = [d for d in range(D) if Bd[d][n] == 1]
    openC  = [c for c in range(C) if Bc[c][n] == 1]
    sum_d  = np.zeros((D, T))
    sum_c  = np.zeros((C, T))
    Fc_d   = np.zeros((D))
    Fc_c   = np.zeros((C))

    for k in Recycled_packaging:
        discount_rate[k][n] = random.choice(discount_list[k])

    for t in range(T):
        for d in openD:
            sum_d[d][t] = sum(Qd[d][i][t][n] for i in range(I))
        for c in openC:
            sum_c[c][t] = sum(Qc[c][k][t][n] for k in range(K))
    
    for d in openD:
        small_size  = all(sum_d[d][t] <= size_d_small [d] for t in range(T))
        medium_size = all(sum_d[d][t] <= size_d_medium[d] for t in range(T))
        if small_size:      Fc_d[d] = Fc_d_small[d].copy()
        elif medium_size:   Fc_d[d] = Fc_d_medium[d].copy()
        else:               Fc_d[d] = Fc_d_large[d].copy()

    for c in openC:
        small_size  = all(sum_c[c][t] <= size_c_small [c] for t in range(T))
        medium_size = all(sum_c[c][t] <= size_c_medium[c] for t in range(T))
        if small_size:      Fc_c[c] = Fc_c_small[c].copy()
        elif medium_size:   Fc_c[c] = Fc_c_medium[c].copy()
        else:               Fc_c[c] = Fc_c_large[c].copy()
    
    #company   
    cost_at_facilities[n] = (
          sum(Fc_d[d] * Bd[d][n] * (T) for d in range(D)) + sum(vb_d[d][i] * Qd[d][i][t][n] for d, i, t in product(range(D), range(I), range(T)))
        + sum(Fc_c[c] * Bc[c][n] * (T) for c in range(C)) + sum(vb_c[c][k] * Qc[c][k][t][n] for c, k, t in product(range(C), range(K), range(T))) 
        + sum(vb_p[p][i] * Qp[p][i][t][n] for p, i, t in product(range(P), range(I), range(T))) 
        + sum(vb_w[w][k] * Qw[w][k][t][n] for w, k, t in product(range(W), range(K), range(T))))
    
    packaging_cost[n] = (
          sum(Pur_PCK[k] * QP[p][k][t][n] * (1 - discount_rate[k][n]) for p, k, t in product(range(P), Recycled_packaging, range(T)))
        + sum(Pur_PCK[k] * NB[p][t][n] for k, p, t in product(Reused_packaging, range(P), range(T))))

    outsource_cost[n]  = sum(Oc[f][i] * Qof[f][i][t][n] for i, t, f in zip(nonzeros_Dmd[0], nonzeros_Dmd[1], nonzeros_Dmd[2]))

    transport_cost[n] = (
          sum(Tc_pd[p][d][i] * Qt_pd[p][d][i][t][n] for p, d, i, t in product(range(P), range(D), range(I), range(T))) 
        + sum(Tc_pf[p][f][i] * Qt_pf[p][f][i][t][n] for p, f, i, t in product(range(P), range(F), range(I), range(T)))
        + sum(Tc_df[d][f][i] * Qt_df[d][f][i][t][n] for d, f, i, t in product(range(D), range(F), range(I), range(T)))  
        + sum(Tc_dc[d][c][k] * Qt_dc[d][c][k][t][n] for d, c, k, t in product(range(D), range(C), range(K), range(T)))
        + sum(Tc_fc[f][c][k] * Qt_fc[f][c][k][t][n] for f, c, k, t in product(range(F), range(C), range(K), range(T)))
        + sum(Tc_cp[c][p][k] * Qt_cp[c][p][k][t][n] for c, p, k, t in product(range(C), range(P), range(K), range(T)))
        + sum(Tc_cw[c][w][k] * Qt_cw[c][w][k][t][n] for c, w, k, t in product(range(C), range(W), range(K), range(T))))

    cleaning_cost[n] = sum(vr[k] * Qt_cp[c][p][k][t][n] for k, c, p, t in product(Reused_packaging, range(C), range(P), range(T)))

    company_cost[n] = cost_at_facilities[n] + packaging_cost[n] + outsource_cost[n] + transport_cost[n] + cleaning_cost[n]

    #supplier
    for k in Recycled_packaging:
        raw_material_cost[k][n]  = sum(Pur_RM[k] * RM[k][t][n] for t in range(T)) 
        recycle_cost[k][n]       = sum(vr[k] * Qt_cp[c][p][k][t][n] for c, p, t in product(range(C), range(P), range(T)))
        revenue_loss[k][n]       = sum(Pur_PCK[k] * QP[p][k][t][n] * discount_rate[k][n] for p, t in product(range(P), range(T)))

        supplier_cost[k][n] = raw_material_cost[k][n] + recycle_cost[k][n] + revenue_loss[k][n]

    #supply chain cost
    fitness[n] = company_cost[n] + sum(supplier_cost[k][n] for k in Recycled_packaging)

    return 

def emission(n):
    Z2_max = (sum(Es_p[p][i] * Qp[p][i][t][n] for p, i, t in product(range(P), range(I), range(T))) 
    + sum(Es_d[d][i] * Qd[d][i][t][n] for d, i, t in product(range(D), range(I), range(T)))
    + sum(Es_c[c][k] * Qc[c][k][t][n] for c, k, t in product(range(C), range(K), range(T)))
    + sum(Es_w[w][k] * Qw[w][k][t][n] for w, k, t in product(range(W), range(K), range(T)))
    + sum(Esr[k] * Qt_cp[c][p][k][t][n] for c, p, k, t in product(range(C), range(P), range(K), range(T)))
    + sum(Esm[k] * RM[k][t][n] for k, t    in product(Recycled_packaging, range(T)))
    + sum(Esm[k] * NB[p][t][n] for p, k, t in product(range(P), Reused_packaging, range(T)))
    + sum(Ets_pd[p][d][i] * Qt_pd[p][d][i][t][n] for p, d, i, t in product(range(P), range(D), range(I), range(T)))
    + sum(Ets_pf[p][f][i] * Qt_pf[p][f][i][t][n] for p, f, i, t in product(range(P), range(F), range(I), range(T)))
    + sum(Ets_df[d][f][i] * Qt_df[d][f][i][t][n] for d, f, i, t in product(range(D), range(F), range(I), range(T)))
    + sum(Ets_dc[d][c][k] * Qt_dc[d][c][k][t][n] for d, c, k, t in product(range(D), range(C), range(K), range(T)))
    + sum(Ets_fc[f][c][k] * Qt_fc[f][c][k][t][n] for f, c, k, t in product(range(F), range(C), range(K), range(T)))
    + sum(Ets_cw[c][w][k] * Qt_cw[c][w][k][t][n] for c, w, k, t in product(range(C), range(W), range(K), range(T)))
    + sum(Ets_cp[c][p][k] * Qt_cp[c][p][k][t][n] for c, p, k, t in product(range(C), range(P), range(K), range(T)))
    + sum(Eos[f][i]   * Qof [f][i][t][n] for i, t, f in zip(nonzeros_Dmd[0], nonzeros_Dmd[1], nonzeros_Dmd[2])))
    return Z2_max

def reverse_cost(n):
    openC  = [c for c in range(C) if Bc[c][n] == 1]
    sum_c  = np.zeros((C, T))
    Fc_c   = np.zeros((C))

    for c in openC:
        small_size  = all(sum_c[c][t] <= size_c_small [c] for t in range(T))
        medium_size = all(sum_c[c][t] <= size_c_medium[c] for t in range(T))
        if small_size:      Fc_c[c] = Fc_c_small[c].copy()
        elif medium_size:   Fc_c[c] = Fc_c_medium[c].copy()
        else:               Fc_c[c] = Fc_c_large[c].copy()

    reverse = (sum(Fc_c[c] * Bc[c][n] * (T) for c in range(C)) + sum(vb_c[c][k] * Qc[c][k][t][n] for c, k, t in product(range(C), range(K), range(T))) 
        + sum(vb_w[w][k] * Qw[w][k][t][n] for w, k, t in product(range(W), range(K), range(T)))
        + sum(Pur_PCK[k] * QP[p][k][t][n] * (1 - discount_rate[k][n]) for p, k, t in product(range(P), Recycled_packaging, range(T)))
        + sum(Pur_PCK[k] * NB[p][t][n] for k, p, t in product(Reused_packaging, range(P), range(T)))
        + sum(Tc_dc[d][c][k] * Qt_dc[d][c][k][t][n] for d, c, k, t in product(range(D), range(C), range(K), range(T)))
        + sum(Tc_fc[f][c][k] * Qt_fc[f][c][k][t][n] for f, c, k, t in product(range(F), range(C), range(K), range(T)))
        + sum(Tc_cp[c][p][k] * Qt_cp[c][p][k][t][n] for c, p, k, t in product(range(C), range(P), range(K), range(T)))
        + sum(Tc_cw[c][w][k] * Qt_cw[c][w][k][t][n] for c, w, k, t in product(range(C), range(W), range(K), range(T)))
        + sum(vr[k] * Qt_cp[c][p][k][t][n] for k, c, p, t in product(Reused_packaging, range(C), range(P), range(T)))
        + sum(Pur_RM[k] * RM[k][t][n] for k, t in product(Recycled_packaging, range(T)))
        + sum(vr[k] * Qt_cp[c][p][k][t][n] for c, p, k, t in product(range(C), range(P), Recycled_packaging, range(T)))
        + sum(Pur_PCK[k] * QP[p][k][t][n] * discount_rate[k][n] for p, k, t in product(range(P), Recycled_packaging, range(T))))
    return reverse

def mutation (n, mutation_rate):
    proportion_Qt_df_mutated = proportion_Qt_df[..., n].copy()
    proportion_Qt_dc_mutated = proportion_Qt_dc[..., n].copy()
    proportion_Qt_fc_mutated = proportion_Qt_fc[..., n].copy()
    proportion_Qt_cw_mutated = proportion_Qt_cw[..., n].copy()
    openD = [d for d in range(D) if Bd[d][n] == 1]
    openC = [c for c in range(C) if Bc[c][n] == 1]
    if len(openD) >= 2:
        #proportion_Qt_df
        for i, t, f in zip(nonzeros_Dmd[0], nonzeros_Dmd[1], nonzeros_Dmd[2]):
            if i in Indirect:
                # Mutation scramble
                indices_to_shuffle = [d for d in openD if random.random() < mutation_rate]
                if indices_to_shuffle:
                    shuffled_inner_list = [proportion_Qt_df_mutated[pos][f][i][t] for pos in indices_to_shuffle]
                    random.shuffle(indices_to_shuffle)
                    for x, pos in enumerate(indices_to_shuffle):
                        proportion_Qt_df_mutated[pos][f][i][t] = shuffled_inner_list[x]
                #random resetting
                if random.random() <= mutation_rate:
                    cell1  = random.choice(openD)
                    remain_indices = [x for x in openD if x!= cell1]
                    cell2     = random.choice(remain_indices)
                    sum_cell = proportion_Qt_df_mutated[cell1][f][i][t] + proportion_Qt_df_mutated[cell2][f][i][t]    
                    proportion_Qt_df_mutated[cell1][f][i][t] = random.choice([0, 0, np.round(random.uniform(0, sum_cell), 2)])
                    proportion_Qt_df_mutated[cell2][f][i][t] = sum_cell - proportion_Qt_df_mutated[cell1][f][i][t]
    for t in range(T):
        for k in range(K):
            if len(openC) >= 2:
                #proportion_Qt_dc
                for d in openD:
                    #mutation scramble
                    indices_to_shuffle = [c for c in openC if random.random() < mutation_rate]
                    if indices_to_shuffle:
                        shuffled_inner_list = [proportion_Qt_dc_mutated[d][pos][k][t] for pos in indices_to_shuffle]
                        random.shuffle(indices_to_shuffle)
                        for c, pos in enumerate(indices_to_shuffle):
                            proportion_Qt_dc_mutated[d][pos][k][t] = shuffled_inner_list[c]
                    #random resetting
                    if random.random() <= mutation_rate:
                        cell1  = random.choice(openC)
                        remain_indices = [x for x in openC if x!= cell1]
                        cell2     = random.choice(remain_indices)
                        sum_cell = proportion_Qt_dc_mutated[d][cell1][k][t] + proportion_Qt_dc_mutated[d][cell2][k][t]    
                        proportion_Qt_dc_mutated[d][cell1][k][t] = random.choice([0, 0, np.round(random.uniform(0, sum_cell), 2)])
                        proportion_Qt_dc_mutated[d][cell2][k][t] = sum_cell - proportion_Qt_dc_mutated[d][cell1][k][t]

                #proportion_Qt_fc
                for f in range(F):
                    #mutation scramble
                    indices_to_shuffle = [c for c in openC if random.random() < mutation_rate]
                    if indices_to_shuffle:
                        shuffled_inner_list = [proportion_Qt_fc_mutated[f][pos][k][t] for pos in indices_to_shuffle]
                        random.shuffle(indices_to_shuffle)
                        for c, pos in enumerate(indices_to_shuffle):
                            proportion_Qt_fc_mutated[f][pos][k][t] = shuffled_inner_list[c]
                    #random resetting
                    if random.random() <= mutation_rate:
                        cell1  = random.choice(openC)
                        remain_indices = [x for x in openC if x!= cell1]
                        cell2     = random.choice(remain_indices)
                        sum_cell = proportion_Qt_fc_mutated[f][cell1][k][t] + proportion_Qt_fc_mutated[f][cell2][k][t]    
                        proportion_Qt_fc_mutated[f][cell1][k][t] = random.choice([0, 0, np.round(random.uniform(0, sum_cell), 2)])
                        proportion_Qt_fc_mutated[f][cell2][k][t] = sum_cell - proportion_Qt_fc_mutated[f][cell1][k][t]

            #proportion_Qt_cw
            for c in openC:
                #Mutation scramble
                indices_to_shuffle = [w for w in range(W) if random.random() < mutation_rate]
                if indices_to_shuffle:
                    shuffled_inner_list = [proportion_Qt_cw_mutated[c][pos][k][t] for pos in indices_to_shuffle]
                    random.shuffle(indices_to_shuffle)
                    for w, pos in enumerate(indices_to_shuffle):
                        proportion_Qt_cw_mutated[c][pos][k][t] = shuffled_inner_list[w]
                #random resetting
                if random.random() <= mutation_rate:
                    cell1  = random.choice(range(W))
                    remain_indices = [x for x in range(W) if x!= cell1]
                    cell2     = random.choice(remain_indices)
                    sum_cell = proportion_Qt_cw_mutated[c][cell1][k][t] + proportion_Qt_cw_mutated[c][cell2][k][t]    
                    proportion_Qt_cw_mutated[c][cell1][k][t] = random.choice([0, 0, np.round(random.uniform(0, sum_cell), 2)])
                    proportion_Qt_cw_mutated[c][cell2][k][t] = sum_cell - proportion_Qt_cw_mutated[c][cell1][k][t]    
    
    proportion_Qt_df[..., SA_neighbor_solution] = proportion_Qt_df_mutated.copy()
    proportion_Qt_dc[..., SA_neighbor_solution] = proportion_Qt_dc_mutated.copy()
    proportion_Qt_fc[..., SA_neighbor_solution] = proportion_Qt_fc_mutated.copy()
    proportion_Qt_cw[..., SA_neighbor_solution] = proportion_Qt_cw_mutated.copy()
    
    return

def update_solution(current, new):
    Bd                   [..., current]      = Bd                   [..., new].copy()
    Bc                   [..., current]      = Bc                   [..., new].copy()
    Qp                   [..., current]      = Qp                   [..., new].copy()
    Qd                   [..., current]      = Qd                   [..., new].copy()
    Qc                   [..., current]      = Qc                   [..., new].copy()
    Qw                   [..., current]      = Qw                   [..., new].copy()
    RM                   [..., current]      = RM                   [..., new].copy()
    NB                   [..., current]      = NB                   [..., new].copy()
    Qof                  [..., current]      = Qof                  [..., new].copy()
    Qt_pd                [..., current]      = Qt_pd                [..., new].copy()
    Qt_pf                [..., current]      = Qt_pf                [..., new].copy()
    Qt_df                [..., current]      = Qt_df                [..., new].copy()
    Qt_dc                [..., current]      = Qt_dc                [..., new].copy()
    Qt_fc                [..., current]      = Qt_fc                [..., new].copy()
    Qt_cw                [..., current]      = Qt_cw                [..., new].copy()
    Qt_cp                [..., current]      = Qt_cp                [..., new].copy()
    QP                   [..., current]      = QP                   [..., new].copy()
    QP_nextperiod        [..., current]      = QP_nextperiod        [..., new].copy()
    Pck_d                [..., current]      = Pck_d                [..., new].copy()
    proportion_Qt_pf     [..., current]      = proportion_Qt_pf     [..., new].copy()
    proportion_Qt_df     [..., current]      = proportion_Qt_df     [..., new].copy()
    proportion_Qt_dc     [..., current]      = proportion_Qt_dc     [..., new].copy()
    proportion_Qt_fc     [..., current]      = proportion_Qt_fc     [..., new].copy()
    proportion_Qt_cw     [..., current]      = proportion_Qt_cw     [..., new].copy()
    fitness              [current]           = fitness              [new].copy()
    company_cost         [current]           = company_cost         [new].copy()
    cost_at_facilities   [current]           = cost_at_facilities   [new].copy()
    packaging_cost       [current]           = packaging_cost       [new].copy()
    outsource_cost       [current]           = outsource_cost       [new].copy()
    transport_cost       [current]           = transport_cost       [new].copy()
    cleaning_cost        [current]           = cleaning_cost        [new].copy()
    supplier_cost        [..., current]      = supplier_cost        [..., new].copy()
    raw_material_cost    [..., current]      = raw_material_cost    [..., new].copy()
    recycle_cost         [..., current]      = recycle_cost         [..., new].copy()
    revenue_loss         [..., current]      = revenue_loss         [..., new].copy()   
    discount_rate        [..., current]      = discount_rate        [..., new].copy()

    return

def create_dummy():
    global dummy_Bd, dummy_Bc, dummy_Qp, dummy_Qd, dummy_Qc, dummy_Qw, dummy_RM, dummy_NB, dummy_Qof
    global dummy_Qt_pd, dummy_Qt_pf, dummy_Qt_df, dummy_Qt_dc, dummy_Qt_fc, dummy_Qt_cw, dummy_Qt_cp
    global dummy_QP, dummy_QP_nextperiod, dummy_Pck_d
    global dummy_proportion_Qt_df, dummy_proportion_Qt_pf, dummy_proportion_Qt_dc, dummy_proportion_Qt_fc, dummy_proportion_Qt_cw
    global dummy_company_cost, dummy_cost_at_facilities, dummy_packaging_cost, dummy_outsource_cost, dummy_transport_cost, dummy_cleaning_cost
    global dummy_supplier_cost, dummy_raw_material_cost, dummy_recycle_cost, dummy_revenue_loss, dummy_discount_rate
    
    dummy_Bd                    = Bd.copy()
    dummy_Bc                    = Bc.copy()
    dummy_Qp                    = Qp.copy()
    dummy_Qd                    = Qd.copy()
    dummy_Qc                    = Qc.copy()
    dummy_Qw                    = Qw.copy()
    dummy_RM                    = RM.copy()
    dummy_NB                    = NB.copy()
    dummy_Qof                   = Qof.copy()
    dummy_Qt_pd                 = Qt_pd.copy()
    dummy_Qt_pf                 = Qt_pf.copy()
    dummy_Qt_df                 = Qt_df.copy()
    dummy_Qt_dc                 = Qt_dc.copy()
    dummy_Qt_fc                 = Qt_fc.copy()
    dummy_Qt_cw                 = Qt_cw.copy()
    dummy_Qt_cp                 = Qt_cp.copy()
    dummy_QP                    = QP.copy()
    dummy_QP_nextperiod         = QP_nextperiod.copy()
    dummy_Pck_d                 = Pck_d.copy()
    dummy_proportion_Qt_df      = proportion_Qt_df.copy()
    dummy_proportion_Qt_pf      = proportion_Qt_pf.copy()
    dummy_proportion_Qt_dc      = proportion_Qt_dc.copy()
    dummy_proportion_Qt_fc      = proportion_Qt_fc.copy()
    dummy_proportion_Qt_cw      = proportion_Qt_cw.copy()
    dummy_company_cost          = company_cost.copy()
    dummy_cost_at_facilities    = cost_at_facilities.copy()
    dummy_packaging_cost        = packaging_cost.copy()
    dummy_outsource_cost        = outsource_cost.copy()
    dummy_transport_cost        = transport_cost.copy()
    dummy_cleaning_cost         = cleaning_cost.copy()
    dummy_supplier_cost         = supplier_cost.copy()
    dummy_raw_material_cost     = raw_material_cost.copy()
    dummy_recycle_cost          = recycle_cost.copy()
    dummy_revenue_loss          = revenue_loss.copy()  
    dummy_discount_rate         = discount_rate.copy()

    return

def delete_dummy():   
    global dummy_Bd, dummy_Bc, dummy_Qp, dummy_Qd, dummy_Qc, dummy_Qw, dummy_RM, dummy_NB, dummy_Qof
    global dummy_Qt_pd, dummy_Qt_pf, dummy_Qt_df, dummy_Qt_dc, dummy_Qt_fc, dummy_Qt_cw, dummy_Qt_cp
    global dummy_QP, dummy_QP_nextperiod, dummy_Pck_d
    global dummy_proportion_Qt_df, dummy_proportion_Qt_pf, dummy_proportion_Qt_dc, dummy_proportion_Qt_fc, dummy_proportion_Qt_cw
    global dummy_company_cost, dummy_cost_at_facilities, dummy_packaging_cost, dummy_outsource_cost, dummy_transport_cost, dummy_cleaning_cost
    global dummy_supplier_cost, dummy_raw_material_cost, dummy_recycle_cost, dummy_revenue_loss, dummy_discount_rate

    del dummy_Bd, dummy_Bc, dummy_Qp, dummy_Qd, dummy_Qc, dummy_Qw, dummy_RM, dummy_NB, dummy_Qof
    del dummy_Qt_pd, dummy_Qt_pf, dummy_Qt_df, dummy_Qt_dc, dummy_Qt_fc, dummy_Qt_cw, dummy_Qt_cp
    del dummy_QP, dummy_QP_nextperiod, dummy_Pck_d
    del dummy_proportion_Qt_df, dummy_proportion_Qt_pf, dummy_proportion_Qt_dc, dummy_proportion_Qt_fc, dummy_proportion_Qt_cw
    del dummy_company_cost, dummy_cost_at_facilities, dummy_packaging_cost, dummy_outsource_cost, dummy_transport_cost, dummy_cleaning_cost
    del dummy_supplier_cost, dummy_raw_material_cost, dummy_recycle_cost, dummy_revenue_loss, dummy_discount_rate
    return

def update_indices(current_idx, new_idx):
    # Update based on the last index
    Bd               [..., current_idx] = dummy_Bd               [..., new_idx].copy()
    Bc               [..., current_idx] = dummy_Bc               [..., new_idx].copy()
    Qp               [..., current_idx] = dummy_Qp               [..., new_idx].copy()
    Qd               [..., current_idx] = dummy_Qd               [..., new_idx].copy()
    Qc               [..., current_idx] = dummy_Qc               [..., new_idx].copy()
    Qw               [..., current_idx] = dummy_Qw               [..., new_idx].copy()
    RM               [..., current_idx] = dummy_RM               [..., new_idx].copy()
    NB               [..., current_idx] = dummy_NB               [..., new_idx].copy()
    Qof              [..., current_idx] = dummy_Qof              [..., new_idx].copy()
    Qt_pd            [..., current_idx] = dummy_Qt_pd            [..., new_idx].copy()
    Qt_pf            [..., current_idx] = dummy_Qt_pf            [..., new_idx].copy()
    Qt_df            [..., current_idx] = dummy_Qt_df            [..., new_idx].copy()
    Qt_dc            [..., current_idx] = dummy_Qt_dc            [..., new_idx].copy()
    Qt_fc            [..., current_idx] = dummy_Qt_fc            [..., new_idx].copy()
    Qt_cw            [..., current_idx] = dummy_Qt_cw            [..., new_idx].copy()
    Qt_cp            [..., current_idx] = dummy_Qt_cp            [..., new_idx].copy()  
    QP               [..., current_idx] = dummy_QP               [..., new_idx].copy()
    QP_nextperiod    [..., current_idx] = dummy_QP_nextperiod    [..., new_idx].copy()
    Pck_d            [..., current_idx] = dummy_Pck_d            [..., new_idx].copy()
    proportion_Qt_df [..., current_idx] = dummy_proportion_Qt_df [..., new_idx].copy()
    proportion_Qt_pf [..., current_idx] = dummy_proportion_Qt_pf [..., new_idx].copy()
    proportion_Qt_dc [..., current_idx] = dummy_proportion_Qt_dc [..., new_idx].copy()
    proportion_Qt_fc [..., current_idx] = dummy_proportion_Qt_fc [..., new_idx].copy()
    proportion_Qt_cw [..., current_idx] = dummy_proportion_Qt_cw [..., new_idx].copy()
    company_cost          [current_idx] = dummy_company_cost          [new_idx].copy()
    cost_at_facilities    [current_idx] = dummy_cost_at_facilities    [new_idx].copy()
    packaging_cost        [current_idx] = dummy_packaging_cost        [new_idx].copy()
    outsource_cost        [current_idx] = dummy_outsource_cost        [new_idx].copy()
    transport_cost        [current_idx] = dummy_transport_cost        [new_idx].copy()
    cleaning_cost         [current_idx] = dummy_cleaning_cost         [new_idx].copy()
    supplier_cost    [..., current_idx] = dummy_supplier_cost    [..., new_idx].copy()
    raw_material_cost[..., current_idx] = dummy_raw_material_cost[..., new_idx].copy()
    recycle_cost     [..., current_idx] = dummy_recycle_cost     [..., new_idx].copy()
    revenue_loss     [..., current_idx] = dummy_revenue_loss     [..., new_idx].copy()
    discount_rate    [..., current_idx] = dummy_discount_rate    [..., new_idx].copy()

    return

def simulated_annealing(n, SA_max_iter, mutation_rate):
    update_solution(SA_best_solution, n)
    update_solution(SA_current_solution, n)
    Bd              [:, SA_neighbor_solution]   = Bd              [:, SA_current_solution].copy()
    Bc              [:, SA_neighbor_solution]   = Bc              [:, SA_current_solution].copy()    
    proportion_Qt_pf[..., SA_neighbor_solution] = proportion_Qt_pf[..., n].copy()

    SA_iter = 1
    SA_Temperature = 2000
    SA_Alpha = 0.8
    
    while SA_iter <= SA_max_iter:
        mutation    (SA_current_solution, mutation_rate) 
        correction  (SA_neighbor_solution)
        cost        (SA_neighbor_solution)

        #comparison and update (if any)
        delta_fitness =  (fitness[SA_neighbor_solution] - fitness[SA_current_solution])/fitness[SA_best_solution]
        if delta_fitness < 0 or random.uniform(0,1) < np.exp(-delta_fitness/SA_Temperature):
            update_solution(SA_current_solution, SA_neighbor_solution)
        if fitness[SA_current_solution] < fitness[SA_best_solution]:
            update_solution(SA_best_solution, SA_current_solution)
        
        SA_Temperature *= SA_Alpha
        SA_iter += 1

    if fitness[SA_best_solution] < fitness[n]:
        update_solution(n, SA_best_solution)
    
    return

def moving (n):
    sum_capacity_D = 0
    sum_capacity_C = 0
    Not_satisfy = True
    while Not_satisfy:
        Lamda = random.random()
        Alpha = random.random()
        P1 = np.random.choice(range(N1))
        P2 = np.random.choice(range(N1 + N2, Nkeshtel))
        for d in range(D):
            Bd_P1 = (Bd[d][P1])*Lamda
            Bd_P2 = (Bd[d][P2])*(1-Lamda)

            Bd[d][n] = Bd[d][n]*Alpha + (Bd_P1 + Bd_P2)*(1-Alpha)
            if Bd[d][n] >= 0.5: Bd[d][n] = 1
            else: Bd[d][n] = 0
        sum_capacity_D = sum(CP_d[d] for d in range(D) if Bd[d][n] == 1)
    
        for c in range(C):
            Bc_P1 = (Bc[c][P1])*Lamda
            Bc_P2 = (Bc[c][P2])*(1-Lamda)
            
            Bc[c][n] = Bc[c][n]*Alpha + (Bc_P1 + Bc_P2)*(1-Alpha)
            if Bc[c][n] >= 0.5: Bc[c][n] = 1
            else: Bc[c][n] = 0
        sum_capacity_C = sum(CP_c[c] for c in range(C) if Bc[c][n] == 1)
        
        if all(sum_capacity_D >= required_capacity_D[t] for t in range(T)) and all(sum_capacity_C >= required_capacity_C[t] for t in range(T)):
            Not_satisfy = False    
    return

#MAIN ALGORITHM
start_time = time.time()

print('initialize')
#initialize
for n in range(0, Nkeshtel):    
    random_binary(n)
    random_proportion(n)  
    correction(n)
    cost(n)

#sort and rank
sort_list = [fitness[n] for n in range(Nkeshtel)]
sorted_indices = np.argsort(sort_list)
fitness = np.sort(sort_list)
create_dummy()
for n, i in enumerate(sorted_indices):
    update_indices(n, i)
delete_dummy()
GBest = fitness[0]
#Resize of fitness
zeros = np.zeros((3,))
fitness = np.concatenate((fitness, zeros), axis = 0)

#Iteration 1: ...
iteration = 1
times_of_no_update = 0
mutation_rate = 0.5
SA_iteration = 30

while iteration <= Max_main_iter and times_of_no_update <= 7:
    print(iteration)
    #N1: EXPLOIT by Simulated Annealing 
    for n in range(0, N1):
        simulated_annealing(n, SA_iteration, mutation_rate)  
        print('N1', n, fitness[n])

    #N2: MOVING
    for n in range(N1, N1+N2):
        moving(n)
        random_proportion(n)
        correction(n)
        cost(n)
        print('N2', n, fitness[n])

    #N3: RANDOM
    for n in range(N1+N2, Nkeshtel):
        random_binary(n)
        random_proportion(n)   
        correction(n)
        cost(n)               
        print('N3', n, fitness[n])

    #sort and rank
    sort_list = [fitness[n] for n in range(Nkeshtel)]
    sorted_indices = np.argsort(sort_list)
    fitness = np.sort(sort_list)
    create_dummy()
    for n, i in enumerate(sorted_indices):
        update_indices(n, i)
    delete_dummy()
    #Resize of fitness
    zeros = np.zeros((3,))
    fitness = np.concatenate((fitness, zeros), axis = 0)

    if fitness[0] < GBest:
        GBest = fitness[0].copy()
        times_of_no_update = 0
        find_time = time.time()
    else: times_of_no_update += 1
    
    print('GBest', iteration, fitness[0])
    print('Bd', Bd[..., 0])
    print('Bc', Bc[..., 0])

    iteration += 1

end_time = time.time()

Z2_max = emission(0)
reverse = reverse_cost(0)

print('Bd', Bd[:, 0])
print('Bc', Bc[:, 0])

print('Z1_min', GBest)
print('Z2_max', Z2_max)

print("Total cost", fitness[0])
print("CreamWork" , company_cost[0])
for k in Recycled_packaging:
    print('supplier ', k, supplier_cost[k, 0])

Record_time = end_time - start_time 
Converged_time = find_time - start_time
print("Record time "      , Record_time )
print("Converged time"    , Converged_time)

print('Reverse cost',       reverse)
print('Transporation'     , transport_cost[0])

for k in Recycled_packaging:
    print('supplier ', k)
    print('reprocess'     , k, recycle_cost[k, 0])
    print('revenue loss'  , revenue_loss[k, 0])
    print('raw', raw_material_cost[k, 0])
    print('discount rate' , discount_rate[k][0])

