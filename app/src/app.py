#app.py
import db
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdt
import pandas as pd
import math
import datetime
from decimal import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


print("##################")
print("##  Starting..  ##")
print("##################")


def fft_plot(prod_id="All"):
  #plot the discrete fourier transform of the date - naively
  prodQuery = """
  SELECT DATE_ORDER, SUM(QTY_ORDER) FROM sales
  WHERE PROD_ID = '{}'
  AND DATE_ORDER < '2019-10-1'
  GROUP BY DATE_ORDER
  ORDER BY DATE_ORDER;
  """
  allQuery = """
  SELECT DATE_ORDER, SUM(QTY_ORDER) FROM sales
  WHERE DATE_ORDER < '2019-10-1'
  GROUP BY DATE_ORDER
  ORDER BY DATE_ORDER;
  """
  if prod_id == "All": query=allQuery
  else: query = prodQuery
  ret = db.selectQueryDB(query.format(prod_id))
  sampleSize = len(ret); print("SampleSize",sampleSize)
  data = [(_["DATE_ORDER"],_["SUM(QTY_ORDER)"]) for _ in ret]
  values = [_[1] for _ in data]

  #print("L25"); print(np.fft.rfftfreq(len(values)))

  FFT = np.fft.rfft(values, norm="ortho")
  absFFT = np.abs(FFT)[1:]
  period = 1./np.fft.rfftfreq(len(values))[1:]
  normFFT = absFFT/np.amax(absFFT)
  fftList = [(1./i, _) for i,_ in zip(np.fft.rfftfreq(len(values))[1:],absFFT)]

  fig, ax = plt.subplots()

  ax.stem(period, normFFT)
  ax.set_xlabel('Periodicidade (dias)')
  ax.set_ylabel('Frequency Domain (Spectrum) Normalized Magnitude')
  ax.set_xlim(0,  np.amax(period))
  ax.set_ylim(0, 1)
  plt.title("Max at {}".format(period[np.where(normFFT==1.0)]))
  plt.savefig('{}_fft.png'.format(prod_id))

  ax.stem(period, normFFT)
  ax.set_xlabel('Período (dias)')
  ax.set_ylabel('Frequency Domain (Spectrum) Normalized Magnitude')
  ax.set_xlim(0,  35)
  ax.set_ylim(0, 1)
  plt.title("Max at {}".format(period[np.where(normFFT==1.0)]))
  plt.savefig('{}_fftzoom.png'.format(prod_id))
 
def fill_missing_sale_date(data, start_date, end_date):
  #fill missing date with a order with 0 quantity and prev day price
  idx = 0
  for date in pd.date_range(start=start_date, end=end_date):
    if date.date() == data[idx]["DATE_ORDER"]:
      pass
    else:
      filler = [{'DATE_ORDER': date.date(), 
                 'WEEKDAY(DATE_ORDER)+1' : date.isoweekday(), 
                 'SUM(QTY_ORDER)': Decimal("0"),
                 'AVG(REVENUE/QTY_ORDER)': data[idx-1]["AVG(REVENUE/QTY_ORDER)"]}]
      data = data[:idx] + filler + data[idx:]
      #print(filler)
    idx += 1
  return data

def plot_data(data, fit, predicts, saveName):
  #print(len(fit[0]), fit[1].shape)
  #print(predicts.shape)

  rule = mdt.rrulewrapper(mdt.DAILY, interval=15)
  loc = mdt.RRuleLocator(rule)
  formatter = mdt.DateFormatter('%d/%m/%y')
  fig, ax = plt.subplots()
  if data != None:
    plt.plot_date(data[0], data[1], ls='solid')
  if fit != None:
    plt.plot_date(fit[0], fit[1], ls='solid')
  #date_idx = 0

  if predicts != None:
  #for p in predicts:
    #dates = [begin_predict_date + datetime.timedelta(days = date_idx + i) for i in range(0,len(p))]
    plt.plot_date(predicts[0], predicts[1], ls='dashed')
    #date_idx += 1
  ax.xaxis.set_major_locator(loc)
  ax.xaxis.set_major_formatter(formatter)
  ax.xaxis.set_tick_params(rotation=30, labelsize=10)
  plt.savefig('{}_data_plot.png'.format(saveName))


def first_difference(dataset):
  #takes the difference from the dataset and return. the first value is lost
  ret = [Decimal("0")]
  for i in range(1, len(dataset)):
    value = dataset[i] - dataset[i - 1]
    ret.append(value)
  return np.asarray(ret)

def recover_first_difference(dataset, startValue):
  #takes the diff'ed dataset and recover the original values from the startValue
  ret = [float(startValue)]
  for d in dataset:
    ret.append(ret[-1]+d)
  return np.asarray(ret)

def organize_data(dataFrame, past_sight, future_sight):
  #print(dFrame)
  colsX = []
  colsY = []
  
  x = dataFrame.iloc[:, :-1]
  y =  dataFrame.iloc[:, -1]
  #recording the past
  for i in range(past_sight, 0, -1):
    colsX.append(x.shift(i))
    colsY.append(y.shift(i))
  #recording the future
  for i in range(0, future_sight):
    colsX.append(x.shift(-i))
    colsY.append(y.shift(-i))
  #print(cols)
  ret = pd.concat(colsX+colsY, axis=1).dropna()
  #print(ret)
  return ret


def model_LSTM(prod_id, past_sight,  future_sight, epochs, batch_size):
  """The	main	objective	is	to	create	a	model	to	predict	the	
  quantity	sold	for	 each	product	given	a	prescribed	price"""
  
  print("Starting {}_{}_{}_{}_{}".format(prod_id, past_sight, future_sight, epochs, batch_size))
  query = """
  SELECT DATE_ORDER, WEEKDAY(DATE_ORDER)+1, SUM(QTY_ORDER), AVG(REVENUE/QTY_ORDER) FROM sales
  WHERE PROD_ID = '{}'
  AND DATE_ORDER < '2019-05-01'
  GROUP BY DATE_ORDER
  ORDER BY DATE_ORDER;
  """
  
  dataset = db.selectQueryDB(query.format(prod_id))
  #print("dataset")
  #print(dataset)
  #print(*dataset, sep='\n')
  #there are some days with no sell, therefore a missing day
  dataset = fill_missing_sale_date(dataset, dataset[0]["DATE_ORDER"], dataset[-1]["DATE_ORDER"])
  

  dates = [_["DATE_ORDER"] for _ in dataset]
  qty = [_["SUM(QTY_ORDER)"] for _ in dataset]
  weekday = [_["WEEKDAY(DATE_ORDER)+1"] for _ in dataset]
  avgprice = [_["AVG(REVENUE/QTY_ORDER)"] for _ in dataset]
  #qty_diff = qty#first_difference(qty)
  
  dataFrame = pd.concat([ 
                   pd.DataFrame(weekday),
                   pd.DataFrame(avgprice),
                   pd.DataFrame(qty)], axis=1)
  
  org_data = organize_data(dataFrame, past_sight, future_sight)
  #fit the serie
  mmscaler = MinMaxScaler(feature_range=(0, 1))
  org_data_fit = mmscaler.fit_transform(org_data)

  #remove the last 15 records to test
  test_size = 15
  x, y = org_data_fit[:-test_size, :-future_sight], org_data_fit[:-test_size, -future_sight:]
  if False:
    print("org data")
    print(org_data_fit)
    print("x")
    print(x)
    print("y")
    print(y)
  x = x.reshape(x.shape[0], 1 , x.shape[1])
  
  #get the last 15 records for test
  test_x, test_y = org_data_fit[-test_size:, :-future_sight], org_data_fit[-test_size:, -future_sight:]
  test_x = test_x.reshape(test_x.shape[0], 1 , test_x.shape[1])
  
  #network params
  network = Sequential()
  network.add(LSTM(50, input_shape=(x.shape[1], x.shape[2])))
  network.add(Dense(future_sight))
  network.compile(loss='mae', optimizer='adam')
  #fit network
  history = network.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)

  model_fit = network.predict(x)
  #print(model_fit)
  predict = network.predict(test_x)
  #print(predict)
  x = x.reshape(x.shape[0], x.shape[2])
  test_x = test_x.reshape(test_x.shape[0], test_x.shape[2])
  model_fit = np.concatenate([x,model_fit], axis=1)
  predict = np.concatenate([test_x, predict], axis=1)
  
  inv_model_fit = mmscaler.inverse_transform(model_fit)
  inv_predict = mmscaler.inverse_transform(predict)
  fit = inv_model_fit[:,-future_sight:] #recover_first_difference(inv_model_fit[:,-1], qty[past_sight-1])
  predict = inv_predict[:,-future_sight:]
  #print(fit)
  plot_data((dates[past_sight:], qty[past_sight:]), 
            (dates[past_sight:-test_size], fit), 
            (dates[-test_size:], predict),
             #dates[-test_size], predict, 
             "{}_{}_{}_{}_{}".format(prod_id, past_sight, future_sight, epochs, batch_size))

  rmse = math.sqrt(mean_squared_error(predict, qty[-test_size:] ))
  return rmse

def metrics():
  """Along	with	the	statistical	model,	we	
  need	metrics,	relationships	and	descriptions	of	these	data	in	order	to	understand	the	sales	
  behavior.	What	does	the	data	tell	us?	How	are	the	different	data	sources	related?	Is	there	a	
  particular	competitor	that	seems	more	important?"""

  salesQuery = """
  SELECT DATE_ORDER, PROD_ID, WEEKDAY(DATE_ORDER)+1, SUM(QTY_ORDER), AVG(REVENUE/QTY_ORDER) 
  FROM sales
  GROUP BY PROD_ID, DATE_ORDER
  ORDER BY PROD_ID, DATE_ORDER;
  """
  #PROD_ID, DATE_EXTRACTION, COMPETITOR, COMPETITOR_PRICE, PAY_TYPE
  compQuery = """
  SELECT PROD_ID, DATE_FORMAT(DATE_EXTRACTION, '%d-%m-%Y') as DATE, 
  COMPETITOR, COUNT(COMPETITOR_PRICE) as COUNT_COMP_PRICE,
  MIN(COMPETITOR_PRICE) as MIN, MAX(COMPETITOR_PRICE) as MAX, 
  AVG(COMPETITOR_PRICE) as AVG_COMP_PRICE, PAY_TYPE
  FROM comp_prices
  GROUP BY comp_prices.PROD_ID, DATE, comp_prices.COMPETITOR, comp_prices.PAY_TYPE
  ORDER BY comp_prices.PROD_ID, STR_TO_DATE(DATE, '%d-%m-%Y');
  """
  joinQuery = """
  CREATE TEMPORARY TABLE IF NOT EXISTS #AggSales as (\
  SELECT PROD_ID, DATE_ORDER, WEEKDAY(DATE_ORDER)+1 as WEEKDAY, SUM(QTY_ORDER) as SUM_QTY, \
  AVG(REVENUE/QTY_ORDER) as AVG_PRICE \
  GROUP BY PROD_ID, DATE_ORDER \
  FROM sales \
  );

  CREATE TEMPORARY TABLE IF NOT EXISTS #AggComp as (\
  SELECT PROD_ID, DATE_FORMAT(DATE_EXTRACTION, '%d-%m-%Y') as DATE, \
  COMPETITOR, COUNT(COMPETITOR_PRICE) as COUNT_COMP_PRICE, \
  MIN(COMPETITOR_PRICE) as MIN, MAX(COMPETITOR_PRICE) as MAX, \
  AVG(COMPETITOR_PRICE) as AVG_COMP_PRICE, PAY_TYPE \
  GROUP BY PROD_ID, DATE, COMPETITOR, PAY_TYPE \
  FROM comp_prices \
  );

  SELECT *
  FROM #AggComp
  LEFT JOIN #AggSales ON #AggComp.PROD_ID = #AggSales.PROD_ID 
        AND #AggComp.DATE_EXTRACTION = #AggSales.DATE_ORDER
  GROUP BY #AggComp.PROD_ID, #AggComp.DATE, #AggComp.COMPETITOR, #AggComp.PAY_TYPE
  ORDER BY #AggComp.PROD_ID, STR_TO_DATE(#AggComp.DATE, '%d-%m-%Y');

  """

  sales = db.selectQueryDB(salesQuery)
  sales = fill_missing_sale_date(sales, sales[0]["DATE_ORDER"], sales[-1]["DATE_ORDER"])
  print(sales[0])
  print(len(sales))
  salesDF = pd.DataFrame(sales)
  print(salesDF)

  comp = db.selectQueryDB(compQuery)
  print(comp[-1])
  print(len(comp))
  compDF = pd.DataFrame(comp)
  print(compDF)




#fft_plot("P6")
#fft_plot("P5")
#fft_plot("P4")
#fft_plot("P3")
#fft_plot("P2")
#fft_plot("P1")
#fft_plot("All")

rmse_dict={}
if False:
  for P in ["P5"]:
    for batch_size in [15, 32]:
      for epochs in [5, 50, 500, 5000]:
        for past_sight in [1, 5, 15]:
          for future_sight in [1]:
            key = "{}_{}_{}_{}_{}".format(P, past_sight, future_sight, epochs, batch_size)
            err = model_LSTM(P, past_sight, future_sight, epochs, batch_size)
            print(err, flush=True)
            rmse_dict[key] = err

  print(rmse_dict)
  sorted_rmse = sorted(rmse_dict.items(), key=lambda x: x[1])
  print(sorted_rmse)
  with open('models_rmse.csv', 'w') as f:
    for s in sorted_rmse:
      f.write(s[0]+", "+s[1])

if True:
  metrics()



db.closeDB()

