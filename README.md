# LMCP
data
└── electricity.csv
└── ETTh1.csv
└── ETTh2.csv
└── ETTm1.csv
└── ETTm2.csv
└── traffic.csv
└── weather.csv
└── solar_AL.txt

bash ./scripts/ETTh1.sh

After training:

- Your trained model will be safely stored in `./checkpoints`.
- Numerical results in .npy format can be found in `./results`.
- A comprehensive summary of quantitative metrics is accessible in `./result_long_term_forecast.txt`.

numpy==1.26.4
pandas==2.2.3
scikit_learn==1.5.2
torch==2.3.1
torchaudio==2.3.1
torchvision==0.18.1

Overall Architecture:
Overall structure of LMPC, which is composed of two parallel branches: spatial and temporal. Each branch consists of three modules:(1) Independent Embedding
 Layer, which projects time segment and time point features into high-dimensional spaces
 separately;(2) Feature Interaction Layer, where Patch-GCN captures diverse spatial col
laboration patterns, while Time-Attention models long-term temporal dependencies and
 rich temporal features;(3) Feature Fusion Layer, which employs a balance controller to
 facilitate effective integration of spatial and temporal features, ultimately enabling the
 prediction of future time series.

Main Result of Multivariate Forecasting:
presents thecomplete forecastingresultsalongwithperformance improvement
 analysis.Forallbaselinemodels,theinputsequencelengthLissetto96.Theresultsare
 averagedoverfourdifferentpredictionhorizons: T∈{96,192,336,720}. LowerMSEand
 MAEvaluesindicatebetterperformance,withthebestresultshighlightedinbold.

Acknowledgement：
[DSIN-PMA](https://github.com/yejunjiePhD/DSIN-PMA)
https://github.com/thuml/TimesNet
https://github.com/YoZhibo/MSGNet
https://github.com/hqh0728/CrossGNN
https://github.com/Thinklab-SJTU/Crossformer
https://github.com/huangsiyuan924/MEAformer
https://github.com/yuqinie98/patchtst
[iTransformer](https://github.com/thuml/iTransformer)

Contact:
If you have any questions or concerns, please contact us: 12024215196@stu.yun.edu.cn or zhaochunna@ynu.edu.cn or submit an issue.
