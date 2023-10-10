# Time Series Classic to Deep

Time Series Classic to Deep is a library dedicated to the field of time series forecasting. Our primary objective is to provide a comprehensive collection of both classical and deep learning-based algorithms for tackling time series forecasting tasks. 

We will gradually enhance and expand our library as the TSA(Time Series Analysis) course progresses.

[TSA home page](https://www.lamda.nju.edu.cn/yehj/TSA2023/)


## Usage examples

```
python main.py --data_path ./dataset/ETT/ETTh1.csv --dataset ETT --target OT --model MeanForecast
```

```
python main.py --data_path ./dataset/ETT/ETTh1.csv --dataset ETT --target OT --model TsfKNN --n_neighbors 1 --msas MIMO --distance euclidean
```

## Datasets
All datasets can be found [here](https://box.nju.edu.cn/d/b33a9f73813048b8b00f/).
- [x] M4
- [x] ETT
- [ ] Traffic
- [ ] Electricity
- [ ] Exchange-Rate
- [ ] Weather
- [ ] ILI(illness)

## Models
- [x] ZeroForecast
- [x] MeanForecast
- [x] TsfKNN
- [ ] LinearRegressionForecast
- [ ] ExponentialSmoothingForecast

## Transformations
- [x] IdentityTransform
- [ ] Normalization
- [ ] Standardization
- [ ] Mean Normalization
- [ ] Box-Cox

## Metrics
- [x] mse
- [ ] mae
- [ ] mase
- [ ] mape
- [ ] smape