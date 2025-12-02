# LMCP
data  
└── electricity.csv  
└── ETTh1.csv  
└── ETTh2.csv  
└── ETTm1.csv  
└── ETTm2.csv  
└── traffic.csv  
└── weather.csv  
└── solar_AL.  

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

## Overall Architecture:
Overall structure of LMPC, which is composed of two parallel branches: spatial and temporal. Each branch consists of three modules:(1) Independent Embedding Layer, which projects time segment and time point features into high-dimensional spaces separately;(2) Feature Interaction Layer, where Patch-GCN captures diverse spatial collaboration patterns, while Time-Attention models long-term temporal dependencies and rich temporal features;(3) Feature Fusion Layer, which employs a balance controller to facilitate effective integration of spatial and temporal features, ultimately enabling the prediction of future time series.
<img width="911" height="588" alt="image" src="https://github.com/user-attachments/assets/a67d67ee-1bee-433b-bad8-d7f376845d32" />


## Main Result of Multivariate Forecasting:
presents thecomplete forecastingresultsalongwithperformance improvement analysis.Forallbaselinemodels,theinputsequencelengthLissetto96.Theresultsar averagedoverfourdifferentpredictionhorizons: T∈{96,192,336,720}. LowerMSEand MAEvaluesindicatebetterperformance,withthebestresultshighlightedinbold.
<img width="925" height="842" alt="image" src="https://github.com/user-attachments/assets/657a205c-115a-41a5-af6d-fce5df5071fd" />
##   5.8. Evaluation on Real-World Scenario Datasets
dataset  
└── Beijing Air Quality.csv  
└── Energy.csv  

## Acknowledgement：
[DSIN-PMA]https://github.com/yejunjiePhD/DSIN-PMA  
[TimesNet] https://github.com/thuml/TimesNet  
[MSGNet] https://github.com/YoZhibo/MSGNet  
[CrossGNN] https://github.com/hqh0728/CrossGNN  
[Crossformer] https://github.com/Thinklab-SJTU/Crossformer  
[MEAformer]https://github.com/huangsiyuan924/MEAformer  
[patchtst] https://github.com/yuqinie98/patchtst  
[iTransformer]https://github.com/thuml/iTransformer

## Contact:
If you have any questions or concerns, please contact us: 12024215196@stu.yun.edu.cn or zhaochunna@ynu.edu.cn or submit an issue.
