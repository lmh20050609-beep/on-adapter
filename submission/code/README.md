# The implementation of ADAPT-Z (Automatic Delta Adjustment via Persistent Tracking in Z-space)
This code is the implementation of ADAPT-Z (Automatic Delta Adjustment via Persistent Tracking in Z-space) algorithm, along with supplementary experiments from the paper and implementations of baselines.

This codebase is primarily based on DSOF (Fast and Slow Streams for Online Time Series Forecasting Without Information Leakage, ICLR 2025; paper: https://openreview.net/pdf?id=I0n3EyogMi, repository: https://github.com/yyalau/iclr2025_dsof)
## 1) Train basemodel
run train_basemodel.py to obtain base model

## 2) Online deployment
run adapt-z.py to conduct online prediction

## Baselines and supplementary experiments
run run_dsof.py to conduct experiments using DSOF

run ADCSD.py to conduct experiments using ADCSD

run parameter_finetune to conduct experiments using OGD

run z_finetune to conduct experiments using f-OGD

run train_FAN.py/train_DishTS.py and ada_online2 fan.py/ada_online dishts.py to conduct experiments using FAN/DishTS and FAN+/DishYS+

## Pseudo code
![sample-wise](sample_version.png)
![batch-wise](batch_version.png)

export PATH="$PWD/miniconda3/bin:$PATH"
. miniconda3/etc/profile.d/conda.sh
conda activate cpiri

 python iadapter_multimodal_traffic.py                    # Traffic.csv (default)
 python iadapter_multimodal_traffic.py --data Economy     # Economy.csv
 python iadapter_multimodal_traffic.py --data Health --seq-len 48

python adapt_z_multimodal.py --data Traffic --pred_len 6 --seeds 2023,2024,2025 \
  --text_source preds --dropout 0.3 --weight_decay 5e-4

nohup python adapt_z_multimodal.py >  iadapteronlinelongtermgpt2M_align.log 2>&1 &

nohup python adapt_z_multimodal.py >  Timesnet96gpt2M.log 2>&1 &

nohup python adapt_z_multimodal.py >  PATCHnot96gpt2M.log 2>&1 &

 nohup python adapt_z_multimodal.py --data Environment --backbone GPT4MTS_z --e_layers 6 --csv finally_4mts_z.csv >  xihe4mts.log 2>&1 &
