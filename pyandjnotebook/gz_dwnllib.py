#!/usr/bin/env python
# coding: utf-8

#Модуль загрузки и сохранения биржевых данных

from gz_import import *
from gz_const import *
from gz_mainlib import *


dwnl_ver = '71'

# In[8]:
def dwnlver():
    print('Download ver:', dwnl_ver)

    
#Классы для многопоточности    
class Mythread(Thread):
    def __init__(self, func, args, name=''):
        Thread.__init__(self)
        self.name = name
        self.func = func
        self.args = args
    
    def run(self):
        self.func(*self.args)

class Threadpool():
    
    def __init__(self, numthreads):
        self.maxpoolsize = numthreads
        self.poolsize = 0
        self.threadlist = []
        self.completeworks = 0
        
        
    def isfull(self):
        if self.poolsize >= self.maxpoolsize:
            return True
        else:
            return False
       
    def addthread(self, t):
        if self.poolsize < self.maxpoolsize:
            self.poolsize += 1
            self.threadlist.append(t)
            
    def start_and_wait(self):
        for t in self.threadlist:
            t.start()
        
        for t in self.threadlist:
            t.join()

        self.completeworks += 1
        
        self.poolsize = 0
        self.threadlist = []
        
def print_mt(q): #, tic_err):
    while True:
        if not q.empty():
            msg = q.get()
            if msg == 100:
                print('\nPrint thread stopped')
                break
            else:
                #msgspl = msg.split()
                print('\r' + msg, end='', flush=True)

                
def que_str(load_list, err_list, addtext=''):
    nload = len(load_list)
    nerror = len(err_list)
    err_str = ', '.join(err_list) 
    que_str = 'Load: {0:<3}. Errs({1:}): {2:}'.format(nload, nerror, err_str)
    #print(que_str)
    return que_str
    

#загрузка инфо    
def yfinfo_tic(tic, yfd, que):
    stock = yf.Ticker(tic)
    yfinfo =  stock.info
    yfinfo.update({'ticker':tic})
    yfd.update({tic:yfinfo})
    que.put(tic+' info ok.')
    
def yfinfo_mt_load(tics, pool_size=75):
    yf_list = []
    yf_dict = {}
    #один тикер
    if type(tics) is str:
        tics = [tics]
    que = Queue()
    error_list = []
    t_print = Mythread(print_mt, (que,))  
    t_print.start()
    t_pool = Threadpool(pool_size)

    for tic in tics:
        if t_pool.isfull():
            t_pool.start_and_wait()
        t = Mythread(yfinfo_tic, 
                     (tic, yf_dict, que), tic)
        t_pool.addthread(t)
    
    #если остались невыполненные работы дорабатываем        
    if t_pool.poolsize > 0: 
        t_pool.start_and_wait()
    que.put(100) #сигнал о завершении
    sleep(0.1)
    return yf_dict #pd.DataFrame(yf_list)
    
#загрузка в многопотоке с Yahoo
def download_yahoo_mt(tics, 
                      t1=datetime.date(2020,1,1), t2=datetime.datetime.today().date(),
                      pool_size=75,
                      cols_rename = cols_rename_yahoo, cols_names = cols_name_yahoo,
                      add_daysec = False,
                      days_nodata = 5,
                      interpolate_NaN = True,
                      load_index=True):
    
    #один тикер
    if type(tics) is str:
        tics = [tics]
    
    que = Queue()
    error_list = []
    df_list = []
    
    
    if load_index:
        print('Start load SP500, NASDAQ index... ', end='')
        
        read_opencloselowhigh_yahoo_mt('^GSPC', t1, t2, cols_rename, cols_names, 
                                       df_list, error_list, que,
                                       nodata_days = 50000)
        
        read_opencloselowhigh_yahoo_mt('^IXIC', t1, t2, cols_rename, cols_names, 
                                       df_list, error_list, que,
                                       nodata_days = 50000)
        
        cols0 = ['date', 'SP500_high', 'SP500_low', 'SP500_open', 'SP500_close', 'SP500_adjclose']
        
        cols1 = ['date', 'NASDAQ_high', 'NASDAQ_low', 'NASDAQ_open', 'NASDAQ_close', 'NASDAQ_adjclose']
        
        
        df_list[0].columns=cols0
        df_list[1].columns=cols1
        
        print('loaded')

    #поток печати сообщений
    t_print = Mythread(print_mt, (que,)) #, error_list,), 't_print')         
    t_print.start()

    t_pool = Threadpool(pool_size)

    for tic in tics:
        if t_pool.isfull():
            t_pool.start_and_wait()
        t = Mythread(read_opencloselowhigh_yahoo_mt, 
                     (tic, t1, t2, cols_rename, cols_names, df_list, error_list, que, 
                      days_nodata), tic)
        t_pool.addthread(t)
        
    #если остались невыполненные работы дорабатываем        
    if t_pool.poolsize > 0: 
        t_pool.start_and_wait()
    
    que.put(100) #сигнал о завершении

    sleep(0.1)
    
    #слияние dataframe-ов
    #doclh - date, open, close, low, high
    #df_doclh = reduce(lambda x, y: pd.merge(x, y, on = 'date'), df_list)
    
    len_dflist = len(df_list)
    
    if len_dflist == 0 or (len_dflist==1 and load_index): #загрузился только индекс SP500
        print('No loads')
        return None, None, None, 0, ['No loads']
    
    #слияние dataframe-ов
    df_doclh = df_list[0]
    for dfx in df_list[1:]:        
        df_doclh = pd.merge(df_doclh, dfx, on='date', how='left')
        
    #корректировка пустых или неопр. значений
    if interpolate_NaN:
        correct_nan(df_doclh)
    
    #только столбец закрытие (adj close)
    df_dc = make_date_adjustedclose_df(df_doclh)
    
    #добавляем столбец даты в секундах и в днях от ноля
    if add_daysec:
        df_dc = add_datedaysandsec_columns(df_dc)

    #корреляция через дневные доходности
    df_corr = correlation_returns(df_dc)

    #return df_doclh, df_dc, df_corr, error_list, len(df_list)
    
    if len(error_list) == 0:
        error_list = ['No errors']

    return df_dc, df_corr, df_doclh, len(df_list), error_list
    
def read_opencloselowhigh_yahoo_mt(ticker, 
                                   date_start, date_end, 
                                   cols_rename, cols_out, 
                                   df_list, error_list, 
                                   que,
                                   nodata_days = 5):
    try:
        df = read_raw_stock_yahoo(ticker, date_start, date_end)
        
    except:
        error_list.append(ticker)
        qstr = que_str(df_list, error_list)
        que.put(qstr)
        #print('err read_opencloselowhigh_yahoo_mt()')
        return
    
    if pd.isnull(df['date'].min()):
        error_list.append(ticker+'_NaT_nodata')
        qstr = que_str(df_list, error_list)
        que.put(qstr)        
        return
    
    if df['date'].min() > pd.Timestamp(date_start + datetime.timedelta(days=nodata_days)):
        error_list.append(ticker+'_nodata_' + str(nodata_days)+ '_days')
        qstr = que_str(df_list, error_list)
        que.put(qstr)        
        return
    
    #cr = {i : (ticker + '_' + val) if val != 'date' else 'date' for i, val in cols_rename.items()}
    
    cout = [(ticker + '_' + x) if x != 'date' else 'date' for x in cols_out]
    
    cols = df.columns
    
    cr = {i : (ticker + '_' + i) if i != 'date' else 'date' for i in cols}
    
    #print(cr)
    
    df = df.rename(cr, axis='columns')
    cols_drop = [x for x in df.columns if x not in cout] 
    df.drop(cols_drop, inplace=True, axis=1)
    
    df_list.append(df)
    qstr = que_str(df_list, error_list)
    que.put(qstr)
        
def read_raw_stock_yahoo_old(ticker, date_start, date_end):
    df = DataReader(ticker, 'yahoo', date_start, date_end)
    df.reset_index(level=0, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df = df[~df[['Date']].duplicated()]
    df.reset_index(drop=True, inplace=True)
    #correct_nan_value(df, 'Close')
    return df


def read_raw_stock_yahoo(ticker, date_start, date_end):
    df = download_yahoo_doclh(ticker, date_start, date_end)  
    return df 

#загрузка через библи yfinance
def download_yf_tics(tics, date_start=datetime.date(2020,1,1), date_end=datetime.datetime.today().date(), load_index=True):
    
    if type(tics) is str:
        tics = [tics]
    
    if (load_index):
        tics = ['^GSPC', '^IXIC'] + tics
   
    if isinstance(date_start, str):
        date_start = datetime.datetime.strptime(date_start, '%Y-%m-%d').date()
        date_end = datetime.datetime.strptime(date_end, '%Y-%m-%d').date()
    
    tdelta = datetime.timedelta(days=1)
    
    date_start1 = date_start + tdelta 
    date_end1 = date_end + tdelta 
    
    #print(date_start1, date_end1)
    #print(date_start, date_end)
    
    dfa = yf.download(tics, date_start1, date_end1)
    
    if len(tics) == 1:
        dfa.columns = [tics[0] + '_' + x.lower().replace(" ", "") for x in dfa.columns]
    else:
        dfa.columns = [x[1] + '_' + x[0].lower().replace(" ", "") for x in dfa.columns]
    
    dfa.columns = [x.replace('^GSPC', 'SP500').replace('^IXIC', 'NASDAQ') for x in dfa.columns]
    
    dfa.reset_index(level=0, inplace=True)
    dfa.rename({'Date':'date'}, axis='columns', inplace=True)
    dfa = dfa[~dfa[['date']].duplicated()]
    dfa.reset_index(drop=True, inplace=True)
    
    #dfa = add_datedaysandsec_columns(dfa)
    
    dfa = dfa[(dfa['date'].dt.date >= date_start) & (dfa['date'].dt.date <= date_end)]
    
    dfa.reset_index(drop=True, inplace=True)
    
    
    
    dfb = dfa.copy()
    cols_drop = [x for x in dfb.columns if ('_close' not in x) and ('date' not in x)]
    dfb.drop(cols_drop, inplace=True, axis=1)
    cols_rename = [x.replace('_close', '') for x in dfb.columns]
    dfb.columns = cols_rename
    
    return dfb, dfa


#загрузка в многопотоке с MOEX
def download_moex_mt(tics, 
                     t1=datetime.date(2020,1,1), t2=datetime.datetime.today().date(),
                     pool_size=75,
                     cols_rename = cols_rename_moex, cols_names = df_colsname_common,
                     add_daysec=False,
                     days_nodata = 5,
                     boardid='TQBR',
                     interpolate_NaN=True,
                     load_index=True):
    #один тикер
    if type(tics) is str:
        tics = [tics]
    
    que = Queue()
    error_list = []
    df_list = []
    add_text = []
    
    if load_index:
        print('Start load MOEX, RTSI index... ', end='')
        read_opencloselowhigh_moex_mt('IMOEX', t1, t2, cols_rename, cols_names, 
                                      df_list, error_list, que,
                                      nodata_days = 50000,
                                      boardid='SNDX')
        
        read_opencloselowhigh_moex_mt('RTSI', t1, t2, cols_rename, cols_names, 
                                      df_list, error_list, que,
                                      nodata_days = 50000,
                                      boardid='RTSI')
        
        #print(df_list[1])
        
        print('loaded')

    t_print = Mythread(print_mt, (que,), 't_print')         
    t_print.start()

    t_pool = Threadpool(pool_size)

    for tic in tics:
        if t_pool.isfull():
            t_pool.start_and_wait()

        t = Mythread(read_opencloselowhigh_moex_mt, 
                     (tic, t1, t2, cols_rename, cols_names, df_list, error_list, que, 
                      days_nodata, boardid), tic)

        t_pool.addthread(t)

    #если остались невыполненные работы дорабатываем        
    if t_pool.poolsize > 0: 
        t_pool.start_and_wait()
    
    que.put(100) #сигнал о завершении

    sleep(0.1)
    
    len_dflist = len(df_list)
    
    if len_dflist == 0 or (len_dflist==1 and load_index): #загрузился только индекс MOEX
        print('No loads')
        return None, None, None, 0, ['No loads']
    
    #слияние dataframe-ов
    #doclh - date, open, close, low, high
    #df_doclh = reduce(lambda x, y: pd.merge(x, y, on = 'date'), df_list)

    #df_imoex, _, _, _, _  = multithread_load_moex('IMOEX', boardid='SNDX', add_daysec=False)
    
    #слияние dataframe-ов
    df_doclh = df_list[0]
    
    for dfx in df_list[1:]:        
        df_doclh = pd.merge(df_doclh, dfx, on='date', how='outer')
    
    #корректировка пустых или неопр. значений
    if interpolate_NaN:
        correct_nan(df_doclh)
    
    #только столбец закрытие
    df_dc = make_date_close_df(df_doclh)
    
    #добавляем столбец даты в секундах и в днях от ноля
    if add_daysec:
        df_dc = add_datedaysandsec_columns(df_dc)

    #корреляция через дневные доходности
    df_corr = correlation_returns(df_dc)
    
    if len(error_list) == 0:
        error_list = ['No errors']
       
    return df_dc, df_corr, df_doclh, len_dflist, error_list
                
def read_opencloselowhigh_moex_mt(ticker, 
                                  date_start, date_end, 
                                  cols_rename, cols_out, 
                                  df_list, error_list, que,
                                  nodata_days = 5,
                                  boardid = 'TQBR'):

    try:
        df = read_raw_stock_moex(ticker, date_start, date_end, boardid)
    except:
        error_list.append(ticker)
        qstr = que_str(df_list, error_list)
        que.put(qstr)
        return
    
    if pd.isnull(df['TRADEDATE'].min()):
        error_list.append(ticker+'_NaT_nodata')
        qstr = que_str(df_list, error_list)
        que.put(qstr)        
        return
    
    if df['TRADEDATE'].min() > pd.Timestamp(date_start + datetime.timedelta(days=nodata_days)):
        error_list.append(ticker+'_nodata_' + str(nodata_days)+ '_days')
        qstr = que_str(df_list, error_list)
        que.put(qstr)        
        return
    
    cr = {i : (ticker + '_' + val) if val != 'date' else 'date' for i, val in cols_rename.items()}
    cout = [(ticker + '_' + x) if x != 'date' else 'date' for x in cols_out]
    df = df.rename(cr, axis='columns')
    cols_drop = [x for x in df.columns if x not in cout] 
    df.drop(cols_drop, inplace=True, axis=1)
                
    df_list.append(df)
    
    qstr = que_str(df_list, error_list)
    que.put(qstr)

    
def read_opencloselowhigh_moex(ticker, 
                               date_start, date_end, 
                               cols_rename, cols_out, 
                               boardid = 'TQBR' ):
    try:
        df = read_raw_stock_moex(ticker, date_start, date_end, boardid)
    except:
        print(ticker, 'load error')
        return
    
    cr = {i : (ticker + '_' + val) if val != 'date' else 'date' for i, val in cols_rename.items()}
    cout = [(ticker + '_' + x) if x != 'date' else 'date' for x in cols_out]
    df = df.rename(cr, axis='columns')
    cols_drop = [x for x in df.columns if x not in cout] 
    df.drop(cols_drop, inplace=True, axis=1)
    return df
    
def read_raw_stock_moex(ticker, date_start, date_end, boardid = 'TQBR'):
    df = DataReader(ticker, 'moex', date_start, date_end)
    df.reset_index(level=0, inplace=True)
    df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'], format='%Y%m%d')
    if boardid != 'ALLBOARDID':
        df = df.loc[df['BOARDID'] == boardid]
        df = df[~df[['TRADEDATE']].duplicated()]
        df.reset_index(drop=True, inplace=True)
        #correct_nan_value(df, 'CLOSE')
    return df


def correct_nan(df):
    tic_list = []
    for tic in cols_wo_dates(df):
        nan_index = df.loc[pd.isnull(df.loc[:, tic])].index
        if len(nan_index) > 0:
            df[tic].interpolate(method='nearest', inplace=True)
            tic_list.append(tic)
    return tic_list        


def check_for_NaN(df):
    nan_tics = []
    for x in cols_wo_dates(dfm):
        nan_index = dfm[pd.isnull(dfm[x])].index
        if len(nan_index) > 0:
            nan_tics.append(x)
    return nan_tics

def load_missing_tics(df, tics, exch='yahoo'):
    existtics = list(set(df.columns) & set (tics))
    misstics = list(set(existtics) ^ set (tics))
    if exch == 'yahoo':
        load_func = multithread_load_yahoo        
    else:
        load_func = multithread_load_moex
    if len(misstics) > 0:
        dfmiss,_,_,_,_ = load_func(misstics, 
                                   t1=df['date'].min().date(), 
                                   t2=df['date'].max().date(), add_daysec=False)
        print(misstics, 'missing loaded.')
        return misstics, pd.merge(df[['date'] + existtics], dfmiss, on='date')
    else:
        print(tics, 'already present in dataframe.')
        return [], df[['date'] + tics]

    
    
    
def tic_info(tic):
    t = yf.Ticker(tic)
    #tinfo = t.info
    return t

 
def download_ticker_info(tic):
    scrape_url = 'https://finance.yahoo.com/quote'
    ticker_url = '{}/{}'.format(scrape_url, tic)
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) \
            Chrome/39.0.2171.95 Safari/537.36'}
    
    html = requests.get(ticker_url, headers=headers).text
   
    
    #несколько попыток
    if 'QuoteSummaryStore' not in html:
        html = requests.get(ticker_url).text
        if 'QuoteSummaryStore' not in html:
            html = requests.get(ticker_url).text
            if 'QuoteSummaryStore' not in html:
                return {}
    
    json_str = html.split('root.App.main =')[1].split('(this)')[0].split(';\n}')[0].strip()
    data = json.loads(json_str)['context']['dispatcher']['stores']['QuoteSummaryStore']
    
    if 'summaryProfile' not in html:
        return {}
    
    ticprofile = data['summaryProfile']
    
    return ticprofile



base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"

def build_url(ticker, start_date = None, end_date = None, interval = "1d"):
    if end_date is None:  
        end_seconds = int(pd.Timestamp("now").timestamp())
    else:
        end_seconds = int(pd.Timestamp(end_date).timestamp())
        
    if start_date is None:
        start_seconds = 7223400    
    else:
        start_seconds = int(pd.Timestamp(start_date).timestamp())
    
    site = base_url + ticker
    
    params = {"period1": start_seconds, "period2": end_seconds,
              "interval": interval.lower(), "events": "div,splits"}
    
    return site, params

def download_yahoo_doclh(tic, start_date = None, end_date = None):
    
    headers = {'User-Agent': 
               'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 \
               Safari/537.36'}
    
    site, params = build_url(tic, start_date, end_date, interval = '1d')
               
    resp = requests.get(site, params = params, headers = headers)
               
    data = resp.json()
    frame = pd.DataFrame(data["chart"]["result"][0]["indicators"]["quote"][0])
    temp_time = data["chart"]["result"][0]["timestamp"]
    frame["adjclose"] = data["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"]   
    frame.index = pd.to_datetime(temp_time, unit = "s")
    frame.index = frame.index.map(lambda dt: dt.floor("d"))
    frame = frame[["open", "high", "low", "close", "adjclose", "volume"]]
    frame = frame.reset_index()
    frame.rename(columns = {"index": "date"}, inplace = True)
    
    return frame


#можно загрузить через download csv но работает хуже
def download_yahoo_doclh_2(ticker, start_date = datetime.date(2020, 1, 1), end_date = datetime.datetime.today().date()):
    t1 = datetime.datetime.combine(start_date, datetime.time(23,59))
    t2 = datetime.datetime.combine(end_date, datetime.time(23,59))
    period1 = int(time.mktime(t1.timetuple()))
    period2 = int(time.mktime(t2.timetuple()))
    interval = '1d' # 1d, 1m
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)
    cols_rename = {'Date':'date', 'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Adj Close':'adjclose',
                   'Volume':'volume'}
    df.rename(columns=cols_rename, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    return df


















 