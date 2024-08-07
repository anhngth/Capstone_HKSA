int numP = ...; 
int numD = ...; 
int numF = ...; 
int numC = ...; 
int numW = ...; 
int numK = ...; 
int numI = ...; 
int numT = ...; 

range P = 1..numP; 
range D = 1..numD;
range F = 1..numF; 
range C = 1..numC; 
range W = 1..numW; 
range K = 1..numK; 
range I = 1..numI; 
range T = 1..numT; 

//3 SKUs
//range Carton = 1..1;   // Recycle
//range Plastic = 2..2; // Recycle
//range Glass = 3..3;  // Reused
//range Indirect = 1..2;
//range Direct = 3..3;

//5 SKUs
//range Carton = 1..2;   // Recycle
//range Plastic = 3..4; // Recycle
//range Glass = 5..5;  // Reused
//range Indirect = 1..4;
//range Direct = 5..5;

//7 SKUs
range Carton = 1..3;   // Recycle
range Plastic = 4..5; // Recycle
range Glass = 6..7;  // Reused
range Indirect = 1..5;
range Direct = 6..7;

//9 SKUs
//range Carton = 1..3;   // Recycle
//range Plastic = 4..6; // Recycle
//range Glass = 7..9;  // Reused
//range Indirect = 1..6;
//range Direct = 7..9;

//11 SKUs
//range Carton = 1..3;   // Recycle
//range Plastic = 4..7; // Recycle
//range Glass = 8..11;  // Reused
//range Indirect = 1..7;
//range Direct = 8..11;

//13 SKUs
//range Carton = 1..4;   // Recycle
//range Plastic = 5..8; // Recycle
//range Glass = 9..13;  // Reused
//range Indirect = 1..8;
//range Direct = 9..13;

//15 SKUs
//range Carton = 1..5;   // Recycle
//range Plastic = 6..10; // Recycle
//range Glass = 11..15;  // Reused
//range Indirect = 1..10;
//range Direct = 11..15;

range Recycled_pck = 1..2;

 
tuple IT_Set {
int I;
int T;
}

{IT_Set} IT =...;
float Dmd[IT][F] = ...;

int BigM = 10000000000000000;

//cost 
float Fc_d_small[D]    = ...; 
float Fc_d_medium[D]   = ...; 
float Fc_d_large[D]    = ...; 
float size_d_small[D]  = ...; 
float size_d_medium[D] = ...;
float size_d_large[D]  = ...;
  
float Fc_c_small[C]    = ...; 
float Fc_c_medium[C]   = ...; 
float Fc_c_large[C]    = ...; 
float size_c_small[C]  = ...; 
float size_c_medium[C] = ...; 
float size_c_large[C]  = ...;

float vb_p[P][I] = ...; 
float vb_d[D][I] = ...; 
float vb_c[C][K] = ...; 
float vb_w[W][K] = ...; 
float Pur_RM[K]  = ...; 
float Pur_PCK[K] = ...;
float vr[K]      = ...; 

float cost_pd[I] = ...; 
float cost_pf[I] = ...; 
float cost_df[I] = ...; 
float cost_dc[K] = ...; 
float cost_fc[K] = ...; 
float cost_cw[K] = ...; 
float cost_cp[K] = ...;

float distance_pd[P][D] = ...; 
float distance_pf[P][F] = ...; 
float distance_df[D][F] = ...; 
float distance_fc[F][C] = ...; 
float distance_dc[D][C] = ...; 
float distance_cw[C][W] = ...; 
float distance_cp[C][P] = ...;

//emission
float Emission_pd[I] = ...; 
float Emission_pf[I] = ...; 
float Emission_df[I] = ...; 
float Emission_dc[K] = ...; 
float Emission_fc[K] = ...; 
float Emission_cw[K] = ...; 
float Emission_cp[K] = ...; 

float Es_p[P][I] = ...;
float Es_d[D][I] = ...;
float Es_c[C][K] = ...;
float Es_w[W][K] = ...;
float Esr[K] = ...; //recycled content
float Esm[K] = ...; //virgin content

//other 
float theta[K]  = ...; //rate of return pck 
float phi[I]    = ...; //product-to-mainpck conversion rate 
float gamma[I]  = ...; //product-to-largeCTN conversion rate 
float Def[I]    = ...; //defective rate of product at DC 
float Qal[K]    = ...; //quality level of pck at CC 
float omega[K]  = ...; //component-to-product conversion rate 
float beta[K]   = ...; //product-to-component conversion rate 
float CP_p[P]   = ...; 
float CP_d[D]   = ...; 
float CP_c[C]   = ...; 
float CP_w[W]   = ...; 
 
//dvar 
dvar boolean Bd[D]; 
dvar boolean Bc[C]; 
dvar float+ Qp[P][I][T]; 
dvar float+ Qd[D][I][T]; 
dvar float+ Qc[C][K][T]; 
dvar float+ Qw[W][K][T]; 
dvar float+ RM[K][T]; 
dvar float+ NB[P][T]; 
dvar float+ Qof[F][I][T]; 

dvar float+ Qt_pd[P][D][I][T]; 
dvar float+ Qt_pf[P][F][I][T]; 
dvar float+ Qt_df[D][F][I][T]; 
dvar float+ Qt_dc[D][C][K][T]; 
dvar float+ Qt_fc[F][C][K][T]; 
dvar float+ Qt_cw[C][W][K][T]; 
dvar float+ Qt_cp[C][P][K][T]; 

dvar float+ QP[P][K][T]; 
dvar float+ QP_nextperiod[P][K];
dvar float+ Pck_f[F][K][T]; 
dvar float+ Pck_d[D][K][T];

dvar boolean small_d[D];
dvar boolean medium_d[D];
dvar boolean large_d[D];
dvar boolean small_c[C];
dvar boolean medium_c[C];
dvar boolean large_c[C];
dvar boolean b_of[F][I][T];

dvar float+ discount_rate[Recycled_pck];

//Cost
dexpr float Tc_pd[p in P][d in D][i in I] = distance_pd[p][d] * cost_pd[i]; 
dexpr float Tc_pf[p in P][f in F][i in I] = distance_pf[p][f] * cost_pf[i]; 
dexpr float Tc_df[d in D][f in F][i in I] = distance_df[d][f] * cost_df[i]; 
dexpr float Tc_dc[d in D][c in C][k in K] = distance_dc[d][c] * cost_dc[k]; 
dexpr float Tc_fc[f in F][c in C][k in K] = distance_fc[f][c] * cost_fc[k]; 
dexpr float Tc_cw[c in C][w in W][k in K] = distance_cw[c][w] * cost_cw[k]; 
dexpr float Tc_cp[c in C][p in P][k in K] = distance_cp[c][p] * cost_cp[k]; 
dexpr float Oc  [f in F][i in I] = (vb_p[1][i] + Tc_pf[1][f][i])*1.15; 

//Emission
dexpr float Ets_pd[p in P][d in D][i in I] = distance_pd[p][d] * Emission_pd[i];
dexpr float Ets_pf[p in P][f in F][i in I] = distance_pf[p][f] * Emission_pf[i];
dexpr float Ets_df[d in D][f in F][i in I] = distance_df[d][f] * Emission_df[i];
dexpr float Ets_dc[d in D][c in C][k in K] = distance_dc[d][c] * Emission_dc[k];
dexpr float Ets_fc[f in F][c in C][k in K] = distance_fc[f][c] * Emission_fc[k];
dexpr float Ets_cw[c in C][w in W][k in K] = distance_cw[c][w] * Emission_cw[k];
dexpr float Ets_cp[c in C][p in P][k in K] = distance_cp[c][p] * Emission_cp[k];
dexpr float Eos[f in F][i in I]   = (Es_p[1][i in I] + Ets_pf[1][f][i])*0.8;

//dexpr float company_cost =
dexpr float facility_cost =
  sum(d in D) (Fc_d_small [d]*small_d [d]    *(numT))
+ sum(d in D) (Fc_d_medium[d]*medium_d[d]    *(numT))
+ sum(d in D) (Fc_d_large [d]*large_d [d]    *(numT))						 
+ sum(t in T, d in D, i in I) vb_d[d][i]*Qd[d][i][t]  

+ sum(c in C) (Fc_c_small [c]*small_c [c]    *(numT))
+ sum(c in C) (Fc_c_medium[c]*medium_c[c]    *(numT))
+ sum(c in C) (Fc_c_large [c]*large_c [c]    *(numT))	
+ sum(t in T, c in C, k in K) vb_c[c][k]*Qc[c][k][t] 

+ sum(t in T, p in P, i in I) vb_p[p][i]*Qp[p][i][t] 
+ sum(t in T, w in W, k in K) vb_w[w][k]*Qw[w][k][t];

dexpr float packaging_cost = 
  sum(t in T, p in P, k in Recycled_pck) Pur_PCK[k] * QP[p][k][t] * (1 - discount_rate[k])
+ sum(t in T, p in P) Pur_PCK[3]*NB[p][t];

dexpr float outsource =
  sum(t in T, f in F, i in I) Oc[f][i] * Qof[f][i][t];

dexpr float transport =
  sum(t in T, p in P, d in D, i in I) Tc_pd[p][d][i] * Qt_pd[p][d][i][t]
+ sum(t in T, p in P, f in F, i in I) Tc_pf[p][f][i] * Qt_pf[p][f][i][t]
+ sum(t in T, d in D, f in F, i in I) Tc_df[d][f][i] * Qt_df[d][f][i][t]
+ sum(t in T, d in D, c in C, k in K) Tc_dc[d][c][k] * Qt_dc[d][c][k][t]
+ sum(t in T, f in F, c in C, k in K) Tc_fc[f][c][k] * Qt_fc[f][c][k][t]
+ sum(t in T, c in C, w in W, k in K) Tc_cw[c][w][k] * Qt_cw[c][w][k][t]
+ sum(t in T, c in C, p in P, k in K) Tc_cp[c][p][k] * Qt_cp[c][p][k][t];

dexpr float cleaning = sum(t in T, p in P, c in C) vr[3]*Qt_cp[c][p][3][t]; 						 

dexpr float supplier_cost[k in Recycled_pck] =  
  sum(t in T) Pur_RM[k]*RM[k][t] 								//purchasing raw material 
+ sum(t in T, p in P, c in C) vr[k]*Qt_cp[c][p][k][t]			//recycling 
+ sum(t in T, p in P) Pur_PCK[k]*QP[p][k][t]*discount_rate[k]; 	//revenue loss

dexpr float company_cost = facility_cost + packaging_cost + outsource + transport + cleaning;

dexpr float Z2_max = 
  sum(t in T, d in D, i in I) Es_d[d][i]*Qd[d][i][t]                      // DC
+ sum(t in T, c in C, k in K) Es_c[c][k]*Qc[c][k][t]                      // CC
+ sum(t in T, p in P, i in I) Es_p[p][i]*Qp[p][i][t]                      // plant
+ sum(t in T, w in W, k in K) Es_w[w][k]*Qw[w][k][t]                      // W
+ sum(t in T, p in P, k in Recycled_pck) Esm[k]*RM[k][t]                  //virgin content
+ sum(t in T, p in P)                    Esm[3]*NB[p][t]                  //virgin content
+ sum(t in T, p in P, c in C, k in K) Esr[k]*Qt_cp[c][p][k][t]            //recycled content
+ sum(t in T, p in P, d in D, i in I) Ets_pd[p][d][i] * Qt_pd[p][d][i][t] //transport emission
+ sum(t in T, p in P, f in F, i in I) Ets_pf[p][f][i] * Qt_pf[p][f][i][t]
+ sum(t in T, d in D, f in F, i in I) Ets_df[d][f][i] * Qt_df[d][f][i][t]
+ sum(t in T, d in D, c in C, k in K) Ets_dc[d][c][k] * Qt_dc[d][c][k][t]
+ sum(t in T, f in F, c in C, k in K) Ets_fc[f][c][k] * Qt_fc[f][c][k][t]
+ sum(t in T, c in C, w in W, k in K) Ets_cw[c][w][k] * Qt_cw[c][w][k][t]
+ sum(t in T, c in C, p in P, k in K) Ets_cp[c][p][k] * Qt_cp[c][p][k][t]
+ sum(t in T, f in F, i in I) Eos[f][i]   * Qof[f][i][t];                  //outsource


minimize company_cost + sum(k in Recycled_pck) supplier_cost[k]; 

subject to {
	forall(p in P, t in T) {//cons3 
    	sum(i in I) Qp[p][i][t] <= CP_p[p]; 
  	} 
  	
	forall(w in W, t in T) {//cons4 
    	sum(k in K) Qw[w][k][t] <= CP_w[w]; 
    } 

	forall(d in D, t in T) {//cons5 
		sum(i in I) Qd[d][i][t] <= Bd[d]*CP_d[d];
	} 

	forall(c in C, t in T) {//cons6
		sum(k in K) Qc[c][k][t] <= Bc[c]*CP_c[c];
	} 

  	forall(p in P: p==1, f in F, i in I , t in T) {//cons7 
    	Qt_pf[p][f][i][t] + sum(d in D)Qt_df[d][f][i][t] + Qof[f][i][t] == Dmd[<i,t>][f]; 
  	} 
  	
  	forall(d in D, f in F, i in Direct , t in T) {//cons8 
    	Qt_df[d][f][i][t] == 0; 
  	}
  	
  	forall(p in P: p==1, f in F, i in Indirect, t in T) {//cons9 
    	Qt_pf[p][f][i][t] == 0; 
  	}
  	
  	forall(p in P: p==1, i in I, t in T) {//cons10 
    	Qp[p][i][t] == sum(d in D) Qt_pd[p][d][i][t] + sum(f in F) Qt_pf[p][f][i][t]; 
  	} 

  	forall(d in D, i in I, t in T) {//cons11
	    sum(p in P) Qt_pd[p][d][i][t] == Qd[d][i][t]; 
  	} 

  	forall(d in D, i in I, t in T) {//cons12 
	    Qd[d][i][t] == sum(f in F) Qt_df[d][f][i][t] + Def[i] * Qd[d][i][t]; 
  	} 

	forall(t in T, p in P: p==1) {//cons13, 14, 15: 
	    QP[p][1][t] == sum(i in Carton)  Qp[p][i][t] * phi[i] + sum(i in I) Qp[p][i][t] * gamma[i];
	    QP[p][2][t] == sum(i in Plastic) Qp[p][i][t] * phi[i];
	    QP[p][3][t] == sum(i in Glass)   Qp[p][i][t] * phi[i];  
	} 
	
	forall(p in P: p==1, k in K) {//cons13, 14, 15: 
	    QP_nextperiod[p][k] == sum(t in T) QP[p][k][t] / numT;
	}

  	forall(p in P: p==1, k in Recycled_pck, t in T: t!= numT) {//cons16 k = 1,2 
 	    omega[k]*(beta[k] * sum(c in C) Qt_cp[c][p][k][t]) <= QP[p][k][t+1] => omega[k]*(RM[k][t] + beta[k] * sum(c in C) Qt_cp[c][p][k][t]) == QP[p][k][t+1]; 
 		omega[k]*(beta[k] * sum(c in C) Qt_cp[c][p][k][t]) >= QP[p][k][t+1] => RM[k][t] == 0;
 	} 

	forall(p in P: p==1, t in T: t!= numT) {//cons17 k = 3 
	    beta[3] * sum(c in C)Qt_cp[c][p][3][t] <= QP[p][3][t+1] => NB[p][t] + beta[3] * sum(c in C)Qt_cp[c][p][3][t] == QP[p][3][t+1]; 
		beta[3] * sum(c in C)Qt_cp[c][p][3][t] >= QP[p][3][t+1] => NB[p][t] == 0;
	} 

	forall(p in P: p==1, k in Recycled_pck, t in T: t == numT) {//cons16 k = 1,2 
 	    omega[k]*(beta[k] * sum(c in C) Qt_cp[c][p][k][t]) <= QP_nextperiod[p][k] => omega[k]*(RM[k][t] + beta[k] * sum(c in C) Qt_cp[c][p][k][t]) == QP_nextperiod[p][k]; 
 		omega[k]*(beta[k] * sum(c in C) Qt_cp[c][p][k][t]) >= QP_nextperiod[p][k] => RM[k][t] == 0;
 	} 

	forall(p in P: p==1, t in T: t == numT) {//cons17 k = 3 
	    beta[3] * sum(c in C)Qt_cp[c][p][3][t] <= QP_nextperiod[p][3] => NB[p][t] + beta[3] * sum(c in C)Qt_cp[c][p][3][t] == QP_nextperiod[p][3]; 
		beta[3] * sum(c in C)Qt_cp[c][p][3][t] >= QP_nextperiod[p][3] => NB[p][t] == 0;
	} 
	
	forall(f in F, t in T) {//cons18, 19, 20: 
		Pck_f[f][1][t] == sum(i in Carton)  Dmd[<i,t>][f] * theta[1] * phi[i]
		 				+ sum(i in Carton)  Dmd[<i,t>][f] * theta[1] * gamma[i]  
    				    + sum(i in Plastic) Dmd[<i,t>][f] * theta[2] * gamma[i] 
    			    	+ sum(i in Glass)   Dmd[<i,t>][f] * theta[3] * gamma[i]; 
		Pck_f[f][2][t] == sum(i in Plastic) Dmd[<i,t>][f] * theta[2] * phi[i];
		Pck_f[f][3][t] == sum(i in Glass)   Dmd[<i,t>][f] * theta[3] * phi[i]; 
	} 
	
	forall(d in D, t in T) {//cons18, 19, 20: 
		Pck_d[d][1][t] == sum(i in Carton)  Qd[d][i][t] * Def[i] * phi[i] 
						+ sum(i in I)       Qd[d][i][t] * Def[i] * gamma[i]; 
		Pck_d[d][2][t] == sum(i in Plastic) Qd[d][i][t] * Def[i] * phi[i];
		Pck_d[d][3][t] == sum(i in Glass)   Qd[d][i][t] * Def[i] * phi[i]; 
	}

  	forall(f in F, k in K, t in T) {//cons21
	    sum(c in C) Qt_fc[f][c][k][t] == Pck_f[f][k][t]; 
	} 
	
	forall(d in D, k in K, t in T) {//cons21
	    sum(c in C) Qt_dc[d][c][k][t] == Pck_d[d][k][t]; 
	}
	
  	forall(c in C, k in K, t in T) {//cons22
	    sum(f in F) Qt_fc[f][c][k][t] + sum(d in D) Qt_dc[d][c][k][t] == Qc[c][k][t]; 
	} 

  	forall(p in P: p == 1 , c in C, k in K, t in T) {//cons23 
	    Qc[c][k][t] ==  Qt_cp[c][p][k][t] + sum(w in W) Qt_cw[c][w][k][t]; 
	} 

  	forall(c in C, k in K, t in T) {//cons24 
	    Qal[k] * Qc[c][k][t] == sum(p in P) Qt_cp[c][p][k][t]; 
	} 

  	forall(c in C, k in K, t in T) {//cons25
	    (1 - Qal[k]) * Qc[c][k][t] == sum(w in W) Qt_cw[c][w][k][t]; 
	} 

	forall(w in W, k in K, t in T) {//cons26 
	    sum(c in C) Qt_cw[c][w][k][t] == Qw[w][k][t]; 
	} 
	
	forall(k in Recycled_pck) {
	  	discount_rate[k] >= 0.03;
	}
	discount_rate[1] == 0.2;
	discount_rate[2] == 0.1;
	
	forall (d in D) {//cons28: size of DC
	  forall (t in T){
	    sum (i in I) Qd[d][i][t] <= small_d[d]*size_d_small[d] + medium_d[d]*size_d_medium[d] + large_d[d]*size_d_large[d];
		}
	  small_d[d] + medium_d[d] + large_d[d] == Bd[d];
	}
	
	forall (c in C) {//cons29: size of CC
	  forall (t in T){
	    sum (k in K) Qc[c][k][t] <= small_c[c]*size_c_small[c] + medium_c[c]*size_c_medium[c] + large_c[c]*size_c_large[c];
	  }
	  small_c[c] + medium_c[c] + large_c[c] == Bc[c];
	}
    
    forall (f in F, i in I, p in P, t in T) {//impose forward outsource policy: Qof
        Qof[f][i][t] <= b_of[f][i][t] * BigM;
       	(b_of[f][i][t] == 1) => (sum(sku in I) Qp[p][sku][t] + Qof[f][i][t] >= CP_p[p]);
    }
}

execute
 {
writeln("company        = ", company_cost);
writeln("suppier 1      = ", supplier_cost[1]);
writeln("suppier 2      = ", supplier_cost[2]);
writeln("Z2_max         = ", Z2_max);
}
