#app.py
import db
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdt
import pandas as pd
import math
import datetime as dt
from decimal import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose


print("##################")
print("##  Starting..  ##")
print("##################")


def fft_plot(prod_id="All"):
  #plot the discrete fourier transform of the date - naively
  prodQuery = """
  SELECT PROD_ID, DATE_ORDER, SUM(QTY_ORDER), AVG(REVENUE/QTY_ORDER), WEEKDAY(DATE_ORDER)+1
  FROM sales
  WHERE PROD_ID = '{}'
  AND DATE_ORDER < '2019-10-1'
  GROUP BY DATE_ORDER
  ORDER BY DATE_ORDER;
  """
  allQuery = """
  SELECT DATE_ORDER, SUM(QTY_ORDER), AVG(REVENUE/QTY_ORDER), WEEKDAY(DATE_ORDER)+1
  FROM sales
  WHERE DATE_ORDER < '2019-10-1'
  GROUP BY DATE_ORDER
  ORDER BY DATE_ORDER;
  """
  if prod_id == "All": query=allQuery
  else: query = prodQuery

  ret = db.selectQueryDB(query.format(prod_id))
  sampleSize = len(ret); print("SampleSize",sampleSize)
  ret = fill_missing_sale_date(ret, ret[0]["DATE_ORDER"], ret[-1]["DATE_ORDER"])
  
  data = [(_["DATE_ORDER"],_["SUM(QTY_ORDER)"]) for _ in ret]
  values = [_[1] for _ in data]

  #print("L25"); print(np.fft.rfftfreq(len(values)))

  FFT = np.fft.rfft(values, norm="ortho")
  absFFT = np.abs(FFT)[1:]
  period = 1./np.fft.rfftfreq(len(values))[1:]
  normFFT = absFFT/np.amax(absFFT)
  fftList = [(1./freq, val) for freq,val in zip(np.fft.rfftfreq(len(values))[1:],absFFT)]

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
  _prev = -1
  _next = 0
  for date in pd.date_range(start=start_date, end=end_date):
    if date.date() == data[idx]["DATE_ORDER"]:
      pass
    else:
      i = _next if idx == 0 else _prev
      #print(data[idx-i])
      filler = [{'DATE_ORDER': date.date(), 
                 'PROD_ID': data[idx-i]["PROD_ID"],
                 'WEEKDAY(DATE_ORDER)+1' : date.isoweekday(), 
                 'SUM(QTY_ORDER)': Decimal("0"),
                 'AVG(REVENUE/QTY_ORDER)': data[idx+i]["AVG(REVENUE/QTY_ORDER)"]}]
      data = data[:idx] + filler + data[idx:]
      #print(filler)
    idx += 1
  return data

def fill_missing_comp_date(data, start_date, end_date):
  #fill missing date with a prev day data
  idx = 0
  _prev = -1
  _next = 0
  #print(len(data))
  #print(data[-1])
  for date in pd.date_range(start=start_date, end=end_date):
    if idx < len(data): #if index exist
      #print(data[idx])
      if type(data[idx]["DATE"]) == str:
        data[idx]["DATE"] = dt.datetime.strptime(data[idx]["DATE"], "%d-%m-%Y").date()
      if date.date() == data[idx]["DATE"]:
        pass
      else:
        #print("filler inc")
        i = _next if idx == 0 else _prev
        filler = [{'DATE': date.date(), 
                  'PROD_ID': data[idx+i]["PROD_ID"],
                  'COMPETITOR' : data[idx+i]["COMPETITOR"],
                  'COUNT_COMP_PRICE': 0,
                  'MIN': data[idx+i]["AVG_COMP_PRICE"],
                  'MAX': data[idx+i]["AVG_COMP_PRICE"],
                  'AVG_COMP_PRICE': data[idx+i]["AVG_COMP_PRICE"],
                  'PAY_TYPE': data[idx+i]["PAY_TYPE"]}]
        data = data[:idx] + filler + data[idx:]
        #print(filler)
    else:
      #print(data[-1])
      i = _prev
      filler = {'PROD_ID': data[-1]["PROD_ID"],
                'DATE': date.date(), 
                'COMPETITOR' : data[-1]["COMPETITOR"],
                'COUNT_COMP_PRICE': 0,
                'MIN': data[-1]["AVG_COMP_PRICE"],
                'MAX': data[-1]["AVG_COMP_PRICE"],
                'AVG_COMP_PRICE': data[-1]["AVG_COMP_PRICE"],
                'PAY_TYPE': data[-1]["PAY_TYPE"]}
      data.append(filler)
      
    idx += 1
  return data  

def plot_data(data, fit, predicts, ylabel, saveName):
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
  ax.set_xlabel('Data')
  ax.set_ylabel(ylabel)
  plt.title(saveName)
  plt.savefig('{}_data_plot.png'.format(saveName))

def plot_sales():
  salesQuery = """
  SELECT DATE_ORDER, SUM(QTY_ORDER), AVG(REVENUE/QTY_ORDER), AVG(QTY_ORDER)
  FROM sales
  GROUP BY DATE_ORDER
  ORDER BY DATE_ORDER;
  """
  sales = db.selectQueryDB(salesQuery)
  dates = [_["DATE_ORDER"] for _ in sales]
  qty = [_["SUM(QTY_ORDER)"] for _ in sales]
  avgprice = [_["AVG(REVENUE/QTY_ORDER)"] for _ in sales]
  avgqty = [_["AVG(QTY_ORDER)"] for _ in sales]
  plot_data((dates, qty), None, None, "Quantidade de protudos", "sales_qty")
  plot_data((dates, avgprice), None, None, "Preço Médio", "sales_avg_price")
  plot_data((dates, avgqty), None, None, "Quantidade média de produtos", "sales_avg_qty")


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

def _mean(data, index):
  values = [_[index] for _ in data]
  return np.mean(values)

def model_LSTM(prod_id, past_sight,  future_sight, epochs, batch_size, units):
  """The	main	objective	is	to	create	a	model	to	predict	the	
  quantity	sold	for	 each	product	given	a	prescribed	price"""
  
  print("Starting {}_{}_{}_{}_{}_{}".format(prod_id, past_sight, future_sight, epochs, batch_size, units))
  query = """
  SELECT PROD_ID, DATE_ORDER, WEEKDAY(DATE_ORDER)+1, SUM(QTY_ORDER), AVG(REVENUE/QTY_ORDER) FROM sales
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
  network.add(LSTM(units, input_shape=(x.shape[1], x.shape[2])))
  network.add(Dense(future_sight))
  network.compile(loss='mean_squared_error', optimizer='adam')
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
             "Quantidade de protudos",
             "{}_{}_{}_{}_{}_{}".format(prod_id, past_sight, future_sight, epochs, batch_size, units))

  rmse = math.sqrt(mean_squared_error(predict, qty[-test_size:] ))
  return rmse

def metrics():
  """Along	with	the	statistical	model,	we	
  need	metrics,	relationships	and	descriptions	of	these	data	in	order	to	understand	the	sales	
  behavior.	What	does	the	data	tell	us?	How	are	the	different	data	sources	related?	Is	there	a	
  particular	competitor	that	seems	more	important?"""

  salesQuery = """
  SELECT DATE_ORDER, PROD_ID, WEEKDAY(DATE_ORDER)+1, SUM(QTY_ORDER), AVG(REVENUE/QTY_ORDER), AVG(QTY_ORDER)
  FROM sales
  GROUP BY PROD_ID, DATE_ORDER
  ORDER BY DATE_ORDER;
  """
  #PROD_ID, DATE_EXTRACTION, COMPETITOR, COMPETITOR_PRICE, PAY_TYPE
  compQuery = """
  SELECT PROD_ID, DATE_FORMAT(DATE_EXTRACTION, '%d-%m-%Y') as DATE, 
  COMPETITOR, COUNT(COMPETITOR_PRICE) as COUNT_COMP_PRICE,
  MIN(COMPETITOR_PRICE) as MIN, MAX(COMPETITOR_PRICE) as MAX, 
  AVG(COMPETITOR_PRICE) as AVG_COMP_PRICE, PAY_TYPE
  FROM comp_prices
  GROUP BY PROD_ID, DATE, COMPETITOR, PAY_TYPE
  ORDER BY STR_TO_DATE(DATE, '%d-%m-%Y');
  """

  sales = db.selectQueryDB(salesQuery)
  prod_ids = set([row["PROD_ID"] for row in sales])
  sales_by_product = {}
  for p in prod_ids:
    sales_by_product[p] = [row for row in sales if row["PROD_ID"] == p]
    print(len(sales_by_product[p]))
    print("{} daily qty mean is: {}".format(p, _mean(sales_by_product[p], "SUM(QTY_ORDER)")))
    print("{} mean price is: {}".format(p, _mean(sales_by_product[p], "AVG(REVENUE/QTY_ORDER)")))
    print("{} avg client qty mean is: {}".format(p, _mean(sales_by_product[p], "AVG(QTY_ORDER)")))
    sales_qty =  [float(_["SUM(QTY_ORDER)"]) for _ in sales_by_product[p]]
    analysis = seasonal_decompose(sales_qty, model='additive', freq=7)
    analysis.plot()
    plt.savefig('{}_analysis.png'.format(p))

    sales_by_product[p] = fill_missing_sale_date(sales_by_product[p], sales[0]["DATE_ORDER"], sales[-1]["DATE_ORDER"])
    
  filled_sales = [item for _list in sales_by_product.values() for item in _list]
  #print(len(filled_sales))

  comp = db.selectQueryDB(compQuery)
  prod_ids = set([row["PROD_ID"] for row in comp])
  comp_ids = set([row["COMPETITOR"] for row in comp])
  pay_type_ids = set([row["PAY_TYPE"] for row in comp])
  comp_dict = {}
  skip_index = ['C5.P8.1', 'C5.P4.1', 'C4.P1.1', 'C4.P1.2', 'C5.P6.1', 'C5.P6.2', 
                'C2.P4.1', 'C2.P4.2', 'C5.P4.2', 'C3.P4.1', 'C3.P4.2', 'C1.P4.1', 
                'C1.P4.2', 'C4.P5.1', 'C4.P5.2', 'C5.P5.1', 'C5.P5.2', 'C6.P5.1', 
                'C6.P5.2', 'C4.P9.1', 'C4.P9.2', 'C5.P8.2', 'C3.P1.1', 'C3.P1.2', 
                'C6.P1.1', 'C6.P1.2', 'C5.P3.1', 'C5.P3.2']
  for p in prod_ids:
    for c in comp_ids:
      for pt in pay_type_ids:
        index = "{}.{}.{}".format(c, p, pt)
        
        if index in skip_index: continue
        temp = [row for row in comp if (row["COMPETITOR"]==c and row["PROD_ID"]==p and row["PAY_TYPE"]==pt)]
        if len(temp) < 200:
          skip_index.append(index)
          #print("not enough data: "+index)
          continue
        #print(index)
        comp_dict[index] = fill_missing_comp_date(temp, sales[0]["DATE_ORDER"], sales[-1]["DATE_ORDER"])
        #print(len(comp_dict[index]))
  filled_comp = [item for _list in comp_dict.values() for item in _list]        
  print(skip_index)
  print(len(skip_index))

  corrDict = {}

  #DATE WEEKDAY VENDOR_P1_QTY VENDOR_P1_PRICE ... VENDOR_P6_QTY VENDOR_P6_PRICE COMPETITOR_C1_P1_PT1_AVG_PRICE COMPETITOR_C1_P1_PT2_AVG_PRICE ... COMPETITOR_C6_P9_PT2_AVG_PRICE
  qty_str_format = "VENDOR_{}_QTY"
  price_str_format = "VENDOR_{}_PRICE"
  for row in filled_sales:
    #print(row)
    
    date = row["DATE_ORDER"].toordinal()
    index = str(date)
    weekday = row["WEEKDAY(DATE_ORDER)+1"]
    product_id = row["PROD_ID"]
    qty = row["SUM(QTY_ORDER)"]
    avg_price = row["AVG(REVENUE/QTY_ORDER)"]
    qty_str = qty_str_format.format(product_id)
    price_str = price_str_format.format(product_id)
    
    if not index in corrDict:
      corrDict[index] = {} 

    corrDict[index]["DATE"] = float(date)
    corrDict[index]["WEEKDAY"] = float(weekday)
    corrDict[index][qty_str] = float(qty)
    corrDict[index][price_str] = float(avg_price)

  for row in filled_comp:
    date = row["DATE"].toordinal()
    index = str(date)
    product_id = row["PROD_ID"]
    comp_id = row["COMPETITOR"]
    pt_id = row["PAY_TYPE"]
    avg_comp_price = row["AVG_COMP_PRICE"]
    comp_price_str = "{}_{}_{}".format(comp_id, product_id, pt_id)

    if not index in corrDict:
      corrDict[index] = {} 

    corrDict[index][comp_price_str] = float(avg_comp_price)

  #print(corrDict["735883"])
  df = pd.DataFrame(corrDict)
  #print(df)
  #print(df.T)
  correlation = df.T.corr()
  n = 3
  for prod in sorted(prod_ids):
    qty_str = qty_str_format.format(prod)
    price_str = price_str_format.format(prod)
    qty_nlargest = correlation.nlargest(n, [qty_str]).loc[:,qty_str]
    qty_nsmallest = correlation.nsmallest(n, [qty_str]).loc[:,qty_str]
    price_nlargest = correlation.nlargest(n, [price_str]).loc[:,price_str]
    price_nsmallest = correlation.nsmallest(n, [price_str]).loc[:,price_str]
    #print(qty_nlargest)
    #print(qty_nsmallest)
    print(price_nlargest)
    print(price_nsmallest)
    print("\n \n \n")

if False:
  plot_sales()

if True:
  metrics()
if False:
  fft_plot("P9")
  fft_plot("P8")
  fft_plot("P7")
  fft_plot("P6")
  fft_plot("P5")
  fft_plot("P4")
  fft_plot("P3")
  fft_plot("P2")
  fft_plot("P1")
  fft_plot("All")

rmse_dict={}
if False:
  for P in ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]:
    for batch_size in [32]:
      for epochs in [5000]:
        for past_sight in [1]:
          for future_sight in [1]:
            for units in [5]:
              key = "{}_{}_{}_{}_{}_{}".format(P, past_sight, future_sight, epochs, batch_size, units)
              err = model_LSTM(P, past_sight, future_sight, epochs, batch_size, units)
              print(err, flush=True)
              rmse_dict[key] = err

  print(rmse_dict)
  sorted_rmse = sorted(rmse_dict.items(), key=lambda x: x[1])
  print(sorted_rmse)
  with open('models_rmse.csv', 'a+') as f:
    for s in sorted_rmse:
      f.write(str(s[0])+", "+str(s[1])+"\n")





db.closeDB()

"""
challenge_app_1  | {'P5_1_1_5_15_50': 291.86062865258083, 'P5_5_1_5_15_50': 253.88880432854472, 'P5_15_1_5_15_50': 206.00863541694358, 'P5_1_1_50_15_50': 180.45464314581525, 'P5_5_1_50_15_50': 170.0052887654057, 'P5_15_1_50_15_50': 172.24991171403158, 'P5_1_1_500_15_50': 130.48948731509498, 'P5_5_1_500_15_50': 146.3519134437674, 'P5_15_1_500_15_50': 164.76761334743568, 'P5_1_1_5000_15_50': 81.69812414208025, 'P5_5_1_5000_15_50': 121.96869227099165, 'P5_15_1_5000_15_50': 171.47151205446042, 'P5_1_1_5_32_50': 297.355395075467, 'P5_5_1_5_32_50': 250.7692329291358, 'P5_15_1_5_32_50': 231.22514063829513, 'P5_1_1_50_32_50': 185.21227558507658, 'P5_5_1_50_32_50': 187.127899244369, 'P5_15_1_50_32_50': 183.0240562290225, 'P5_1_1_500_32_50': 148.68017559599951, 'P5_5_1_500_32_50': 149.88947778661296, 'P5_15_1_500_32_50': 178.84582474965308, 'P5_1_1_5000_32_50': 79.33903623561281, 'P5_5_1_5000_32_50': 84.52737712146606, 'P5_15_1_5000_32_50': 199.7605221527233}
challenge_app_1  | [('P5_1_1_5000_32_50', 79.33903623561281), ('P5_1_1_5000_15_50', 81.69812414208025), ('P5_5_1_5000_32_50', 84.52737712146606), ('P5_5_1_5000_15_50', 121.96869227099165), ('P5_1_1_500_15_50', 130.48948731509498), ('P5_5_1_500_15_50', 146.3519134437674), ('P5_1_1_500_32_50', 148.68017559599951), ('P5_5_1_500_32_50', 149.88947778661296), ('P5_15_1_500_15_50', 164.76761334743568), ('P5_5_1_50_15_50', 170.0052887654057), ('P5_15_1_5000_15_50', 171.47151205446042), ('P5_15_1_50_15_50', 172.24991171403158), ('P5_15_1_500_32_50', 178.84582474965308), ('P5_1_1_50_15_50', 180.45464314581525), ('P5_15_1_50_32_50', 183.0240562290225), ('P5_1_1_50_32_50', 185.21227558507658), ('P5_5_1_50_32_50', 187.127899244369), ('P5_15_1_5000_32_50', 199.7605221527233), ('P5_15_1_5_15_50', 206.00863541694358), ('P5_15_1_5_32_50', 231.22514063829513), ('P5_5_1_5_32_50', 250.7692329291358), ('P5_5_1_5_15_50', 253.88880432854472), ('P5_1_1_5_15_50', 291.86062865258083), ('P5_1_1_5_32_50', 297.355395075467)]
"""