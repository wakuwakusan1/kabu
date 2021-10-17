import codecs
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import metrics
import optuna.integration.lightgbm as lgb

def csv_to_df(path):
  with codecs.open(path, "r", "UTF-8", "ignore") as file:
      df = pd.read_table(file, delimiter=",")
  return df

nikkei = csv_to_df("nikkei_stock_average_daily_jp.csv")
nydow = csv_to_df("nydow2.csv")
usdjpy = csv_to_df("USJPY.csv")
shinko = csv_to_df("1681_jp_d.csv")
toyota = csv_to_df("7911_jp_d.csv")

# csvの結合
for name, df in zip(["nikkei", "nydow", "usdjpy", "shinko", "toyota"],[nikkei, nydow, usdjpy, shinko, toyota]):
  df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
  columns_list = list(df.columns)
  columns_dict = {columns_list[0]:"date"}
  for column in columns_list[1:]:
    columns_dict[column] = f"{name}_{column}"
  df = df.rename(columns=columns_dict)

  if "data" not in globals():
    data = df
  else:
    data = pd.merge(data,df)


# 日付順にソート
data = data.sort_index()

def weighted_average(df):
  ans = df*0
  for i in range(1,6):
    ans += ((6 - i) * df.shift(i) / 15)
  return ans

for column in list(data.columns)[1:]:
  # 直近5日間の加重平均との差分
  data[f"{column}_diff_5Dmean"] = (data[column] - weighted_average(data[column]))
  # 前日比
  data[f"{column}_DoD"] = data[column] / data[column].shift(1)


data["y"] = ((data["toyota_終値"] - data["toyota_終値"].shift(-1)) < 0)*1

train = data[data["date"] < "2020-01-01"]
test = data[("2020-12-31" >= data["date"]) & (data["date"] >= "2020-01-01")]

train = train.set_index('date')
test = test.set_index('date')

lgb_train = lgb.Dataset(train.drop("y", axis=1), train["y"])
lgb_eval = lgb.Dataset(test.drop("y", axis=1), test["y"], reference=lgb_train)


params = {
    'objective': 'binary',
    'metric': 'auc',
    "verbosity": -1,
    "boosting_type": "gbdt",
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=99999999,
                valid_sets=[lgb_train, lgb_eval],
                early_stopping_rounds=10
               )


# テストデータに対し予測を実施
y_pred = gbm.predict(test.drop("y", axis=1), num_iteration=gbm.best_iteration)
# AUCを計算
fpr, tpr, thresholds = metrics.roc_curve(np.asarray(test["y"]), y_pred)
auc = metrics.auc(fpr, tpr)
print("AUC", auc)

# ROC曲線をプロット
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.show()

# accuracy, precisionを計算
acc = metrics.accuracy_score(np.asarray(test["y"]), np.round(y_pred))
precision = metrics.precision_score(np.asarray(test["y"]), np.round(y_pred))
print("accuracy", acc)
print("precision", precision)

# 混同行列をプロット
y_pred = np.round(y_pred)
cm = metrics.confusion_matrix(np.asarray(test["y"]), np.where(y_pred < 0.5, 0, 1))
cmp = metrics.ConfusionMatrixDisplay(cm, display_labels=[0,1])
cmp.plot(cmap=plt.cm.Blues)
plt.show()