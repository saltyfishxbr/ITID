This is the codes and raw results of paper Existence and Strength: Information-Theoretic Discovery of Vehicle Interaction Mechanisms in Mixed Traffic.

All results, including NGSIM, TGSIM L1, TGSIM L2 and synthetic results are provided in zip files and synthetic_validation_results.json

If you want to train this code, modify the directory in build_windows_xxx.py, this file will start to build windows.

Then you need to build the CTE table offline. Similarly, mofity the directory in build_cte_table_xxx.py and precomputed CTE will be saved.

Run train_xxx.py to train the model. You can use tensorboard to monitor the training. Finally, you can use eval_xxx.py to evaluate or use analyze_interaction_xxx_gate_diag.py to do the physical check.

Specifically, the code includes some conterfactual related parts. Counterfactual mechanism is discarded in our paper because the weak contribution but long-time training requirements. If you are interested in this part, you can try it.


We further conduct a reaction analysis on S2R dataset. This dataset is our conducted field experiment and accepted by ITSC2026. The papar name is \textit{S2R-DUAL Dataset: Sim-to-Real Driving with Look-Ahead and Look-Behind Information}. This experiment contains a three-vehicle platoon under different information topology and display mode. You can access the data at https://drive.google.com/file/d/12MCDsni5whefACqVhGJcArYaNigCEKJ8/view?usp=drive_link
