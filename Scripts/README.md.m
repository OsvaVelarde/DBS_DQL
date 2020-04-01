README.m

==========================================================================================================
==========================================================================================================
MODULES -----
Dependences 

1- bgtc_network
		numpy

2- dqagent
		numpy
		random
		collections
		tensorflow.keras

3- rewards
		numpy
		biomarkers

4- biomarkers
		numpy
		scipy.signal
		sklearn.preprocessing
		comodulogram
		filtering 
		segmentation

5- comodulogram
		numpy
		scipy.signal
		sklearn.preprocessing
		filtering

6- downsampling 
		numpy 
		filtering

7- filtering 
		numpy 
		spectrum

8- spectrum 
		scipy.signal

9- segmentation
		numpy 

==========================================================================================================
==========================================================================================================
ANALYSIS
1- Rewards_vs_DBS.py:
	Crea o carga simulaciones del modelo de BGTC variando los parametros G12,G13 y con diferentes amplitudes de DBS. 
	Guarda dichas simulaciones (si fueron creadas) y guarda un archivo con informacion del Power Bands, PLV y Reward.

	- sys
	- os
	- numpy
	- itertools
	- sklearn.preprocessing

	- bgtc_network_v2
	- rewards
	- downsampling

2- Generate_signals.py:
	xxxx

==========================================================================================================
==========================================================================================================
PLOTS


Respecto a la primera version "/mnt/BTE2b/DBS/Enero-Julio-2019/DQLearning/"

En el file 

"/mnt/BTE2b/DBS/Enero-Julio-2019/DQLearning/Scripts/Info.txt"

hay informacion respecto al aprendizaje utilizando una recompensa que depende de Beta + Accion.

La configuración optima hasta el 21/10/2019 es la implementada el dia 02-09-2019
Ver archivo: "/mnt/BTE2b/DBS/Enero-Julio-2019/DQLearning/Trainings/Cfg_2019-09-02.json"

Los resultados se muestran en 
"/mnt/BTE2b/DBS/Enero-Julio-2019/DQLearning/Validation/2019-09-02.dat"

Correr el script "EndTraining.ipynb" 

______________________________________________________________________

Sampling episodes:

We have five states (Steady - LF - HF - SH - PEI).
N_episodes: Number of episodes for DQL training.

Separamos equiprobablemente los episodios en estos 5 estados.

Las regiones en el espacio de parametros (g12,g13) pueden verse en el archivo 
'/mnt/BTE2b/DBS/Agosto-Diciembre-2019/CFC_vs_DBS/Results/[1, 19]_vs_[20, 200]/PLV_Node3_Amp_0.0.eps'

De la observación del mapa, determinamos elegir valores en las siguientes regiones:

Steady: [0.25:0.50] x [0:0.25]
LF: [1:2] x [0:0.5]
HF: [0:0.25] x [0.5:1]
SH: [0.5:1] x [1.30:2]
PEI: [1.25:2] x [1:2]
__________________________________________________________________

Reward function:

R() = exp(coef*Adbs + coef*PLV + coef*LF)
Coef_Adbs = -0.25
Coef_PLV  = -3	
Coef_LF   = -50	

[1 -50 -3 -0.25]

__________________________________________________________________

python3.5 -m cProfile -s 'time' DQLearning.py &> profdql.txt


---------------------------------------------------------------------------------------
import sys
sys.path.append('CARPETA DONDE ESTÁN LOS MODULOS')

-------------------

De los gráficos en:
'/mnt/BTE2b/DBS/Agosto-Diciembre-2019/CFC_vs_DBS/Results/PowerBands/FullMap/BiomarkvsDBS_Node3_States.eps'
'/mnt/BTE2b/DBS/Agosto-Diciembre-2019/CFC_vs_DBS/Results/PowerBands/SH_vs_DBS/BiomarkvsDBS_Node3_SH.eps'

Para el estado Steady:
- Máx PLV = 0.1
- Máx LFPower = 0
- Máx HFPower = 0

Para el estado HF:
- Máx PLV = 0.05
- Máx LFPower = 0
- Máx HFPower = 0.005

Para el estado SH:
- Máx PLV = 0.9
- Máx LFPower = 0.02
- Máx HFPower = 0.005

Para el estado LF:
- Máx PLV = 0.1
- Máx LFPower = 0.05
- Máx HFPower = 0

Para el estado PEI:
- Máx PLV = 0.5
- Máx LFPower = 0.05
- Máx HFPower = 0.002

____________________________________

Resumen:

Amp \in (0,10)
PLV \in (0,1)
LF \in (0,0.05)
HF \in (0,0.01)

Objetivo:
Deseamos plantear una reward function de la forma
S = CombLinear (Amp,PLV,LF,HF)
Reward = exp(S)
_________________________________

Tomemos x,y,z = Amp, PLV, LF

Planteemos la suma pesada a*x + b*y + c*z.
Si queremos un cada sumando contribuya en el mismo orden, se debera cumplir:

a libre.
b = 10 * a 
c = 200 * a

____________________________

Estrategia:
Fijo "a", analizo la forma de la reward function para: b \in (0,20*a) y c \in (0,400*a)
Script: plot_rewardSH.py

Figura:
'/mnt/BTE2b/DBS/Agosto-Diciembre-2019/DQLearning/RewardFunctionAnalysis/Rewards_SH_vs_DBS.eps'
OK!

___________________________

Coeff normalizados para la comparación:

a, b/10, c/200

Ejemplo: (Esta fue la elección)
a = -0.25	---	-0.25
b = -3  	---	-0.3
c = -50		---	-0.25

______________________________



