# TD-Placer
## [TD-Placer] Critical Path Aware Timing-Driven Global Placement for Large-Scale Heterogeneous FPGAs
![The overview of TD-Placer to perform timing-driven FPGA placement](./framework.png)
## Introduction:
TD-Placer has three main components: (1) net information extraction, (2) net delay prediction, and (3) global placement. The extraction component captures net topology under multi-pin driving scenarios using a graph-based method to handle crosstalk and coupling. The delay prediction component uses this topology with timing features in a nonlinear model to estimate end-to-end net delays. The global placement component jointly optimizes timing and wirelength via a unified quadratic objective, applying a fine-grained weighting scheme along critical paths. TD-Placer also includes a lightweight delay prediction algorithm for efficient inference.

**Requirements:**
1. **Python 3.8**
2. **pytorch >= 2.0.0**
3. **cudatoolkit >= 12.0**
4. **torch-geometric >= 2.0.0**
***

## Dataset:
You can directly use the preprocessed dataset:
`./STA/dataset/processed_data.zip` — preprocessed dataset ready to use.

If you need to build the dataset yourself, we provide data preprocessing scripts with Vivado interface support:
- **`./STA/data_preprocess/`** — scripts to extract device and netlist information directly from Vivado projects.
1. After performing placement and routing in Vivado, run `extract_dataset.tcl` to obtain the necessary raw netlist information.
2. run `extractElements_error_catch_cong.tcl` to obtain the placement congestion.(This step is very time-consuming; for netlists with more than 200K LUTs, it may take up to one week.)
You can also modify the congestion handling module in process_ori_dataset.py to disable congestion information. This may result in approximately a 3% performance drop.
3. Running `process_ori_dataset.py` will generate netlist information files in .pkl format under `/STA/dataset/processed_data/`.
4. Run `split_train_val_test.py` and `get_final_train_val_test.py` sequentially to obtain the final TD-Placer STA-standard training dataset.

## Training:
1. Modify the training hyperparameters and path information in `config.py` under `/STA/config/`.
2. Run `python train.py`
3. After training is completed, run `python test.py` to obtain performance evaluation on the test set.

## Global Placement Flow in TD-Placer:
1. You can run `python get_sequence.py` to export model checkpoints compatible with C++ LibTorch, enabling integration into a C++-based placer implementation.
2. The global placement flow of TD-Placer is developed based on AMF-Placer2.0. Due to copyright restrictions of the advanced version, we cannot open-source the placer tool code. You can adapt it using https://github.com/zslwyuan/AMF-Placer/.
* Note: After embedding, 2–7% of samples may produce extreme outputs (greater than or less than twice the average delay) due to special device placements. Applying min-max constraints or replacing them with lookup-table delays allows normal usage.

## Acknowledgements
We hope this FPGA placer will be helpful to the community.
We thank the original authors of AMF-Placer for providing the source code.




