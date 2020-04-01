Reference: Velarde et al 2019.
doi: 10.1016/j.neuroimage.2019.116031

# --------------------------------
Alpha oscillations vs 50 Hz

Connection	Synaptic Efficacy	Delay(ms)	Tau(ms)
1-1		0.50			35		40		
1-2		x			35		40		
1-3		y			5		20		
2-1		-2.5			35		40		
2-3		-1.0/-0.7		5		0.1		
3-2		1.40			5		0.1		

# --------------------------------

Ganglio 	H
1		0.01/0.03
2		
3		0/0.015

# --------------------------------
Remark:
* Beta Oscillations  tau=[5. 20 20 5. 0.1 0.1] d=[20 5. 5 15 5 5]
* Alpha Oscillations tau=[40 40 20 40 0.1 0.1] d=[35 35 5 35 5 5]

# --------------------------------

Observación sobre la amplitud DBS:

En Velarde et al. 2017, para el modelo de dos poblaciones usamos H_1 = 0.8, H_2 = 0, T1 = 0.1, T2= -0.1 y las amplitudes de DBS en el rango [0,12]

En Velarde et al. 2019, ver equation (21): Tomar H_3 = 0

a = H_1/D [G11 - 1; -(G12 + G13*G32); -(G13 + G12*G23)]

La activación de a_í no depende del valor H1 (positivo) y por tanto, las rectas de activación en el plano G son invariantes cambiando H_1.

En particular, cambiamos H1=0.01 -> 0.8 para usar amplitudes de DBS en el rango [0,12]
 
# -------------------------------

