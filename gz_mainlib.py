#!/usr/bin/env python
# coding: utf-8

# In[4]:


from gz_import import *
from gz_const import *


# In[5]:

main_ver = '67'

# In[8]:
def mainver():
    print('Main ver:', main_ver)
    
def get_dictkey(dct, val):
    return [x for x, y in dct.items() if val in y]
    
def return_risk_stock(df, tic, trade_days=252, short=False):
    returns = df[tic].pct_change().dropna()
    if short:
        returns = -returns
    ret = returns.mean() * trade_days 
    risk = returns.std() * np.sqrt(trade_days)
    sharpe_ratio = ret/risk
    return round(ret, 3), round(risk, 3), round(sharpe_ratio, 3)
    
def return_risk_portfolio(df, long_tics, short_tics, w=[], trade_days=252):
    len_lstics = len(long_tics) + len(short_tics)
    if len(w) == 0:
        w = [1 / len_lstics] * len_lstics
    weights = np.array(w)
    df_returns = df[long_tics + short_tics].pct_change().dropna()
    if len(short_tics) > 0:
        df_returns[short_tics] = -df_returns[short_tics] #np.negative(df_returns[short_tics])    
    dfy_cov_returns = df_returns.cov() * trade_days
    var = np.dot(weights.T, np.dot(dfy_cov_returns, weights))
    ret = np.sum(df_returns.mean() * weights) * trade_days
    risk = np.sqrt(var)
    sharpe_ratio = ret/risk
    return round(ret, 3), round(risk, 3), round(sharpe_ratio, 3)


#lt, st - множественные long short
#n_longtics, n_shorttics - для формирования комбинаций из lt, st
def return_risk_combination_df(df, lt, st, n_longtics, n_shorttics, w = [], trade_days=252):
    if n_longtics > len(lt) or n_shorttics > len(st):
        print('Error. Lenght lt or st less number for combination.')
        return
    d = []
    ltst = lt + st
    n_tics = n_longtics + n_shorttics
    df_returns0 = df[ltst].pct_change().dropna()
    if len(w) == 0:
        w = [1 / n_tics] * n_tics
    weights = np.round(np.array(w), 2)
    if len(st) > 0:
        df_returns0[st] = -df_returns0[st]
    for x, y in product(combinations(lt, n_longtics), 
                        combinations(st, n_shorttics)):
        df_returns = df_returns0[list(x)+list(y)]
        df_cov_returns = df_returns.cov() * trade_days
        var = np.dot(weights.T, np.dot(df_cov_returns, weights))
        ret = np.sum(df_returns.mean() * weights) * trade_days
        risk = np.sqrt(var)
        sharpe_ratio = ret/risk
        #print(x, y, ret, risk, sharpe_ratio)
        d.append({'lt':x,'st':y, 
                  'ret':round(ret, 3), 
                  'risk':round(risk, 3), 
                  'sharpe':round(sharpe_ratio, 3),
                  'weights':tuple(weights)})
    df_portf = pd.DataFrame(d)
    return df_portf

def optimize_portfolio(df, lt, st, 
                   x0=[], bounds_t=[]):
    n_longtics = len(lt)
    n_shorttics = len(st)
    ltst = lt + st
    n_tics = n_longtics + n_shorttics
    misstics = list((set(df.columns) & set (ltst)) ^ set (ltst))
    if len(misstics) > 0:
        print('Missing tics', misstics)
        return False, False
    rat1n = 1 / n_tics
    if len(x0) == 0:
        x0 = [rat1n / 2] * n_tics
        print('Init weights:', x0)
    if len(bounds_t) == 0:
        #bounds_t = [(0, 2 * rat1n)] * n_tics
        bounds_t = [(0, 1)] * n_tics
        print('Bound:', bounds_t)
    df_ret = df[ltst].pct_change().dropna()
    out = minimize(sharpe_formin, x0, 
               method = 'SLSQP', bounds=bounds_t, 
               args = (df_ret, lt, st))
    sharpe = round(-out['fun'], 3)
    optweights = out['x'].tolist()
    norm_optweights = np.round(np.array(optweights)/np.array(optweights).sum(), 3).tolist()
    return sharpe, norm_optweights

def sharpe_formin(w, df_rets, long_tics, short_tics, trade_days=252):
    weights = np.array(w)
    df_returns = df_rets.loc[:, long_tics + short_tics]
    if len(short_tics) > 0:
        df_returns[short_tics] = -df_returns[short_tics] #np.negative(df_returns[short_tics])    
    dfy_cov_returns = df_returns.cov() * trade_days
    var = np.dot(weights.T, np.dot(dfy_cov_returns, weights))
    ret = np.sum(df_returns.mean() * weights) * trade_days
    risk = np.sqrt(var)
    sharpe_ratio = ret/risk
    return -sharpe_ratio

def cols_wo_dates(df):
    cols = [x for x in df.columns if 'date' not in x]
    return cols

def make_date_close_df(df):
    dfn = df.copy()
    drop_cols = [x for x in dfn.columns if 'date' not in x and '_close' not in x]
    dfn.drop(drop_cols, axis=1, inplace=True)
    dfn.columns = [x.replace('_close','') for x in dfn.columns]
    return dfn

def make_date_adjustedclose_df(df):
    dfn = df.copy()
    drop_cols = [x for x in dfn.columns if 'date' not in x and '_adjclose' not in x]
    dfn.drop(drop_cols, axis=1, inplace=True)
    dfn.columns = [x.replace('_adjclose','') for x in dfn.columns]
    return dfn

def add_datedaysandsec_columns(df):
    #время в секунды
    df['date_sec'] = (df['date'].apply(lambda x: time.mktime(x.timetuple()))) 
    #(3600 * 24)  c началом в нуле и в днях
    df['date_days'] = (df['date_sec'] - df['date_sec'].min()) / 86400
    
    return df

def correlation_returns(df):
    #без дат
    cols = cols_wo_dates(df)
    #корреляция через дневные доходности
    df_dayearn = df[cols].pct_change().dropna()
    df_corr = df_dayearn.corr()
    return df_corr

def beta_1(df, s0, s0_ref, ses='close'): #s, s_ref это close например SBER_close, ROSN_close
    
    if ses == 'fullname':
        s = s0
        s_ref = s0_ref
    else:
        s = s0 + '_' + ses
        s_ref = s0_ref + '_' + ses
    
    df2 = df.loc[:, [s, s_ref]]
    r = 'r_' + s
    r_ref = 'r_' + s_ref
    df2[r]     = (df[s]     - df[s].shift(1))     / df[s].shift(1)
    df2[r_ref] = (df[s_ref] - df[s_ref].shift(1)) / df[s_ref].shift(1)
    df2 = df2[1:]
    covmatrix = df2[[r, r_ref]].cov()
    
    #print(covmatrix, '\n', df2[r_ref].var())
    return covmatrix.loc[r, r_ref] / df2[r_ref].var()


def beta(df, tic1, tic2):
    #наоборот ибо x, y
    k_slope, _, _ = LR(df[tic2].pct_change().dropna(), df[tic1].pct_change().dropna())
    return k_slope


def stock_returns(df, tic):
    return df[tic].pct_change().dropna()

def corr_tics(df, tic1, tic2):
    return df[[tic1, tic2]].corr().loc[tic1, tic2]
    
def beta_centrbank_30day(dfo, t_eq, t_der):
    df = dfo[['date', t_eq+'_close', t_der+'_close']].copy()
    
    d_eq = t_eq + '_d_eq'
    d_der = t_der + '_d_der'
    
    d_eq2 = t_eq + '_d_eq2'
    d_der2 = t_der + '_d_der2'
    
    d_eq_30 = t_eq + '_d_eq_30'
    d_der_30 = t_der + '_d_der_30'
    
    d_der_30_2 = t_der + '_d_der_30_2'    
    
    d_mul_eq_der_30 = t_eq + '_d_mul_eq_der_30'
    
    df[d_eq] = df[t_eq + '_close'] / df[t_eq + '_close'].shift(1)
    df[d_der] = df[t_der + '_close'] / df[t_der + '_close'].shift(1)
    
    S30d_eq = df.iloc[-30:, df.columns.get_loc(d_eq)].sum() / 30
    S30d_der = df.iloc[-30:, df.columns.get_loc(d_der)].sum() / 30
    
    df[d_eq_30] = df[d_eq] - S30d_eq
    df[d_der_30] = df[d_der] - S30d_der
    
    df[d_mul_eq_der_30] = df[d_eq_30] * df[d_der_30]
    df[d_der_30_2] = df[d_der_30] * df[d_der_30]
   
    beta = df.iloc[-30:, df.columns.get_loc(d_mul_eq_der_30)].sum() / df.iloc[-30:,df.columns.get_loc(d_der_30_2)].sum() 
    
    return beta

def df_close_columns(df):
    dfc = df[[x for x in df.columns if '_close' in x or x == 'date']]
    close_columns = {x:x.replace('_close', '') for x in df.columns if 'close' in x}
    return dfc.rename(close_columns, axis='columns')


def atr(df_doclh, ticker, period=14): #сохранение индекса
    
    close_col = ticker + '_close'
    high_col  = ticker + '_high'
    low_col = ticker + '_low'
    open_col = ticker + '_open'
    atr_col = ticker + '_ATR'
    atrprevopen = ticker + '_ATR_to_prevopen'
    atrprevclose = ticker + '_ATR_to_prevclose'
    
    
    prevclose = ticker + '_prevclose'
    
    df = df_doclh[['date', open_col, close_col, low_col, high_col]].copy()
    
    df.reset_index(drop=True, inplace=True) #можно сбросить индекс
    
    
    
    df[prevclose] = df[close_col].shift(1)
    df.loc[df.index[0], prevclose] = df.loc[df.index[0], open_col]
    
    #prev_close = df[close_col].shift(1)
    
    hl = df[high_col] - df[low_col]
    hpc = abs(df[high_col] - df[prevclose])
    lpc = abs(df[low_col] - df[prevclose])
    
    TR = np.maximum(np.maximum(hl.fillna(0), hpc.fillna(0)), lpc.fillna(0))
    
    df[atr_col] = 0
    
    col_index = df.columns.get_loc(atr_col)
    
    df.iloc[period-1, col_index] = TR.iloc[0:period].sum() / period
    
    for i, (idx, row) in enumerate(df.iloc[period:].iterrows(), period):
        
        df.iloc[i, col_index] =             (df.iloc[i-1, col_index] * (period - 1) + TR.iloc[i]) / period

    
    df[atrprevopen] = df[atr_col] / df[open_col].shift(1)
    df.iloc[0, df.columns.get_loc(atrprevopen)] = 0
    
    
    df[atrprevclose] = df[atr_col] / df[prevclose]
    
    return df[df[atr_col] > 0.0]        

def LR(x0, y0): #x - date from df
    
    if isinstance(x0, pd.Series):
        
        x0.reset_index(drop=True, inplace=True)
        
        if isinstance(x0[0], pd.Timestamp):
            ind = np.array(x0.index)
            x = ind.reshape(-1, 1)
        else:
            x = x0.values.reshape(-1, 1)
    elif isinstance(x0, np.ndarray):
        if isinstance(x0[0], np.datetime64):
            ind = np.arange(0, len(x0))
            x = ind.reshape(-1, 1)
        else:
            x = x0.reshape(-1, 1)
    else:
        x = x0.values.reshape(-1, 1)

    y = y0.values
    
    linear_regressor = LinearRegression()
    linear_regressor.fit(x, y)
    
    y_predict = linear_regressor.predict(x)
    
    #r2 - коэффициент детерминации единица минус доля необъяснённой дисперсии (дисперсии случайной ошибки модели, 
    #или условной по факторам дисперсии зависимой переменной) в дисперсии зависимой переменной. 
    #Его рассматривают как универсальную меру зависимости одной случайной величины от множества других. 
    #В частном случае линейной зависимости {\displaystyle R^{2}}R^2 является квадратом так называемого 
    #множественного коэффициента корреляции между зависимой переменной и объясняющими переменными. 
    #В частности, для модели парной линейной регрессии коэффициент детерминации равен квадрату обычного коэффициента
    #корреляции между y
    
    r2 = r2_score(y, y_predict) 
    
    k_slope = linear_regressor.coef_[0]
    alpha = linear_regressor.intercept_
   
    return k_slope, r2, y_predict

def LR_koefs(x0, y0): #x - date from df
    
    x = x0.values.reshape(-1, 1)
    y = y0.values
    
    linear_regressor = LinearRegression()
    linear_regressor.fit(x, y)
    
    y_predict = linear_regressor.predict(x)
    
    #r2 - коэффициент детерминации единица минус доля необъяснённой дисперсии (дисперсии случайной ошибки модели, 
    #или условной по факторам дисперсии зависимой переменной) в дисперсии зависимой переменной. 
    #Его рассматривают как универсальную меру зависимости одной случайной величины от множества других. 
    #В частном случае линейной зависимости {\displaystyle R^{2}}R^2 является квадратом так называемого 
    #множественного коэффициента корреляции между зависимой переменной и объясняющими переменными. 
    #В частности, для модели парной линейной регрессии коэффициент детерминации равен квадрату обычного коэффициента
    #корреляции между y
    
    r2 = r2_score(y, y_predict) 
    #alpha пересечение с осью y
    alpha = linear_regressor.intercept_
    #beta тангенс угла наклона линии регрессии
    beta = linear_regressor.coef_[0] #угол наклона прямой регрессии
   
    return alpha, beta, y_predict, r2

def LR_spred(x0, y0): #x - date from df
    xlog = np.log(x0)
    ylog = np.log(y0)
    x = xlog.values.reshape(-1, 1)
    y = ylog.values
    
    linear_regressor = LinearRegression()
    linear_regressor.fit(x, y)
    #linear_regressor.fit(x_constant, y)
    
    y_predict = linear_regressor.predict(x)
    
    #Коэффициент детерминации единица минус доля необъяснённой дисперсии (дисперсии случайной ошибки модели, 
    #или условной по факторам дисперсии зависимой переменной) в дисперсии зависимой переменной. 
    #Его рассматривают как универсальную меру зависимости одной случайной величины от множества других. 
    #В частном случае линейной зависимости {\displaystyle R^{2}}R^2 является квадратом так называемого 
    #множественного коэффициента корреляции между зависимой переменной и объясняющими переменными. 
    #В частности, для модели парной линейной регрессии коэффициент детерминации равен квадрату обычного коэффициента
    #корреляции между y

    #r2 = r2_score(y, y_predict) 
    
    alpha = linear_regressor.intercept_
    beta = linear_regressor.coef_[0]
    spred = ylog - (alpha + xlog * beta)

    return spred #alpha, beta, y_predict

def coef_var(x, y):
    if x.dtype == 'datetime64[ns]':
        lpred = LR_wdate(x, y)
    else:
        lpred = LR(x, y)
        
    return (y - lpred).std() / y.mean()

def create_position(long_tics, short_tics, long_betas, short_betas, long_prices, short_prices, size):
    
    dflong = pd.DataFrame()
    dfshort = pd.DataFrame()

    ser_longbeta  = pd.Series(long_betas)
    ser_shortbeta = pd.Series(short_betas)
    
    sum_longbeta = ser_longbeta.sum()
    sum_shortbeta = ser_shortbeta.sum()
    
    total_beta = sum_longbeta + sum_shortbeta
    
    dflong['npp']  = pd.Series(list(range(1, len(long_tics)+1)))
    dfshort['npp'] = pd.Series(list(range(1, len(short_tics)+1)))    
    
    dflong['tic']  = long_tics
    dfshort['tic'] = short_tics
    
    dflong['beta']  = long_betas
    dfshort['beta'] = short_betas
    
    dflong['b2wb']  = ser_longbeta  / total_beta
    dfshort['b2wb'] = ser_shortbeta / total_beta

    dflong['b2sb']  = ser_longbeta  / sum_longbeta
    dfshort['b2sb'] = ser_shortbeta / sum_shortbeta

    sumlongs_1_b2wb  = 1 - dflong['b2wb'].sum()
    sumshorts_1_b2wb = 1 - dfshort['b2wb'].sum()

    dflong['rsum1']  = dflong['b2sb']  * sumlongs_1_b2wb
    dfshort['rsum1'] = dfshort['b2sb'] * sumshorts_1_b2wb

    dflong['pos']  = (1 - dflong['rsum1']  / dflong['rsum1'].sum())  / (len(dflong) - 1)  * dflong['rsum1'].sum()
    dfshort['pos'] = (1 - dfshort['rsum1'] / dfshort['rsum1'].sum()) / (len(dfshort) - 1) * dfshort['rsum1'].sum()

    dflong['pos%']  = dflong['pos'] * 100
    dfshort['pos%'] = dfshort['pos'] * 100
    
    dflong['ticprice']  = pd.Series(long_prices)
    dfshort['ticprice'] = pd.Series(short_prices)
    
    dflong['vol']  = size * dflong['pos']
    dfshort['vol'] = size * dfshort['pos']
    
    dflong['ntics'] =  dflong['vol']  / dflong['ticprice']
    dfshort['nticsk'] = dfshort['vol'] / dfshort['ticprice']
    
    dflong.drop(['b2wb', 'b2sb', 'rsum1'], axis='columns', inplace=True)
    dfshort.drop(['b2wb', 'b2sb', 'rsum1'], axis='columns', inplace=True)
    
    #dflong['shares_round'] = round(dflong['shares'])
    #dfshort['shares_round'] = round(dfshort['shares'])
    
    #dflong['val_round'] = dflong['shares_round'] * dflong['price']
    #dfshort['val_round'] = dfshort['shares_round'] * dfshort['price']      
    
    return dflong, dfshort
    
    
# In[ ]:

def pos_wlots(dflong, lots_long):
    dflong['lot'] = lots_long
    dflong['nlots'] = (dflong['shares'] / dflong['lot']).round().astype(int)
    dflong['shares2'] = dflong['nlots'] * dflong['lot']
    dflong['val2'] = dflong['shares2'] * dflong['price']
    dflong['pval'] = dflong['val'] / dflong['val2']
   
    
#дата время в строку    
def t2s(t):
    
    r = 'datetime_err'
    if type(t) is datetime.date:
        r = datetime.datetime.strftime(t, '%d.%m.%Y')
    elif type(t) is datetime.datetime:
        r = datetime.datetime.strftime(t, '%d.%m.%Y %H:%M')
    elif type(t) is datetime.time:
        r = datetime.time.strftime(t, '%H:%M')
    
    return r      

def ctrends_tin(df_candles, df_instr, tic1, tic2):
    
    dftemp = pd.merge(df_candles, df_instr, on='figi')
    
    dftemp.drop(['interval', 'v', 'figi'], axis=1, inplace=True)
    
    dft1 = dftemp[dftemp['ticker'] == tic1].copy()
    dft2 = dftemp[dftemp['ticker'] == tic2].copy()
    
    dft1.rename(columns={'o':'o_1', 'c':'c_1', 'h':'h_1', 'l':'l_1', 'ticker':'tic_1'}, inplace=True)
    dft2.rename(columns={'o':'o_2', 'c':'c_2', 'h':'h_2', 'l':'l_2', 'ticker':'tic_2'}, inplace=True)
    
    #dft1 = dft1[['time', 'tic_1', 'o_1', 'c_1', 'l_1', 'h_1']]
    #dft2 = dft2[['time', 'tic_2', 'o_2', 'c_2', 'l_2', 'h_2']]
    
    dft1 = dft1.reindex(columns=['time', 'tic_1', 'o_1', 'c_1', 'l_1', 'h_1'])
    dft2 = dft2.reindex(columns=['time', 'tic_2', 'o_2', 'c_2', 'l_2', 'h_2'])

    df = pd.merge(dft1, dft2, on='time')
    
    df['date'] = df['time'].apply(lambda x: x.replace(hour=0, minute=0,  second=0))
    
    df_group = df.loc[df.groupby(['date'])['time'].idxmax()]
    df = pd.merge(df, df_group[['date', 'c_1', 'c_2']], on='date')
    
    df.rename({'c_1_x':'c_1', 'c_1_y':'closeday_1', 'c_2_x':'c_2', 'c_2_y':'closeday_2'}, 
              axis='columns', inplace=True)
    
    
    
    dfclos = df.loc[df.groupby(['date'])['closeday_1'].idxmax()]
    
    dfclos['prevclose1d_1'] = dfclos['closeday_1'].shift(1)
    dfclos['prevclose2d_1'] = dfclos['closeday_1'].shift(2)
    dfclos['prevclose3d_1'] = dfclos['closeday_1'].shift(3)
    dfclos['prevclose1d_2'] = dfclos['closeday_2'].shift(1)
    dfclos['prevclose2d_2'] = dfclos['closeday_2'].shift(2)
    dfclos['prevclose3d_2'] = dfclos['closeday_2'].shift(3)


    dfclos = dfclos[['date', 'prevclose1d_1', 'prevclose1d_2', 
                             'prevclose2d_1', 'prevclose2d_2', 
                             'prevclose3d_1', 'prevclose3d_2']]
    
    df = pd.merge(df, dfclos, on = 'date')
    
    df.dropna(inplace=True)
    
    #среднее 
    df['a_1'] = (df['o_1'] + df['c_1'] + df['h_1'] + df['l_1']) / 4
    df['a_2'] = (df['o_2'] + df['c_2'] + df['h_2'] + df['l_2']) / 4
    
    df['a_1_pc1d'] = df['a_1'] / df['prevclose1d_1'] - 1
    df['a_2_pc1d'] = df['a_2'] / df['prevclose1d_2'] - 1
    
    df['a_1_pc2d'] = df['a_1'] / df['prevclose2d_1'] - 1
    df['a_2_pc2d'] = df['a_2'] / df['prevclose2d_2'] - 1
    
    df['a_1_pc3d'] = df['a_1'] / df['prevclose3d_1'] - 1
    df['a_2_pc3d'] = df['a_2'] / df['prevclose3d_2'] - 1

    
    df['a1-a2_1d'] = df['a_1_pc1d'] - df['a_2_pc1d']
    df['a1-a2_2d'] = df['a_1_pc2d'] - df['a_2_pc2d']
    df['a1-a2_3d'] = df['a_1_pc3d'] - df['a_2_pc3d']
    
    av1 = df.loc[df.groupby(['date'])['a1-a2_1d'].idxmin()][['date', 'time', 'a1-a2_1d']]
    av2 = df.loc[df.groupby(['date'])['a1-a2_2d'].idxmin()][['date', 'time', 'a1-a2_2d']]
    av3 = df.loc[df.groupby(['date'])['a1-a2_3d'].idxmin()][['date', 'time', 'a1-a2_3d']]
    
    av4 = df.loc[df.groupby(['date'])['a1-a2_1d'].idxmax()][['date', 'time', 'a1-a2_1d']]
    av5 = df.loc[df.groupby(['date'])['a1-a2_2d'].idxmax()][['date', 'time', 'a1-a2_2d']]
    av6 = df.loc[df.groupby(['date'])['a1-a2_3d'].idxmax()][['date', 'time', 'a1-a2_3d']]

    av_1d = pd.merge(av1, av4, on='date')
    av_2d = pd.merge(av2, av5, on='date')
    av_3d = pd.merge(av3, av6, on='date')

    av_1d.columns = ['date', 't_min', 'a1-a2_1d_min', 't_max', 'a1-a2_1d_max']
    av_2d.columns = ['date', 't_min', 'a1-a2_2d_min', 't_max', 'a1-a2_2d_max']
    av_3d.columns = ['date', 't_min', 'a1-a2_3d_min', 't_max', 'a1-a2_3d_max']

    av_1d['earn'] = av_1d['a1-a2_1d_max'] - av_1d['a1-a2_1d_min']
    av_2d['earn'] = av_2d['a1-a2_2d_max'] - av_2d['a1-a2_2d_min']
    av_3d['earn'] = av_3d['a1-a2_3d_max'] - av_3d['a1-a2_3d_min']
    
    df['o_1_pc1d'] = df['o_1'] / df['prevclose1d_1'] - 1
    df['c_1_pc1d'] = df['c_1'] / df['prevclose1d_1'] - 1
    df['h_1_pc1d'] = df['h_1'] / df['prevclose1d_1'] - 1
    df['l_1_pc1d'] = df['l_1'] / df['prevclose1d_1'] - 1
    df['o_2_pc1d'] = df['o_2'] / df['prevclose1d_2'] - 1
    df['c_2_pc1d'] = df['c_2'] / df['prevclose1d_2'] - 1
    df['h_2_pc1d'] = df['h_2'] / df['prevclose1d_2'] - 1
    df['l_2_pc1d'] = df['l_2'] / df['prevclose1d_2'] - 1

    df['o_1_pc2d'] = df['o_1'] / df['prevclose2d_1'] - 1
    df['c_1_pc2d'] = df['c_1'] / df['prevclose2d_1'] - 1
    df['h_1_pc2d'] = df['h_1'] / df['prevclose2d_1'] - 1
    df['l_1_pc2d'] = df['l_1'] / df['prevclose2d_1'] - 1
    df['o_2_pc2d'] = df['o_2'] / df['prevclose2d_2'] - 1
    df['c_2_pc2d'] = df['c_2'] / df['prevclose2d_2'] - 1
    df['h_2_pc2d'] = df['h_2'] / df['prevclose2d_2'] - 1
    df['l_2_pc2d'] = df['l_2'] / df['prevclose2d_2'] - 1

    df['o_1_pc3d'] = df['o_1'] / df['prevclose3d_1'] - 1
    df['c_1_pc3d'] = df['c_1'] / df['prevclose3d_1'] - 1
    df['h_1_pc3d'] = df['h_1'] / df['prevclose3d_1'] - 1
    df['l_1_pc3d'] = df['l_1'] / df['prevclose3d_1'] - 1
    df['o_2_pc3d'] = df['o_2'] / df['prevclose3d_2'] - 1
    df['c_2_pc3d'] = df['c_2'] / df['prevclose3d_2'] - 1
    df['h_2_pc3d'] = df['h_2'] / df['prevclose3d_2'] - 1
    df['l_2_pc3d'] = df['l_2'] / df['prevclose3d_2'] - 1
    
    
    df['h1-l2_1d'] = df['h_1_pc1d'] - df['l_2_pc1d']
    df['h2-l1_1d'] = df['h_2_pc1d'] - df['l_1_pc1d']
    
    df['h1-l2_2d'] = df['h_1_pc2d'] - df['l_2_pc2d']
    df['h2-l1_2d'] = df['h_2_pc2d'] - df['l_1_pc2d']
    
    df['h1-l2_3d'] = df['h_1_pc3d'] - df['l_2_pc3d']
    df['h2-l1_3d'] = df['h_2_pc3d'] - df['l_1_pc3d']
  
    
    a1 = df.loc[df.groupby(['date'])['h1-l2_1d'].idxmax()][['date', 'h1-l2_1d']]
    a2 = df.loc[df.groupby(['date'])['h2-l1_1d'].idxmax()][['date', 'h2-l1_1d']]

    a3 = df.loc[df.groupby(['date'])['h1-l2_1d'].idxmin()][['date', 'h1-l2_1d']]
    a4 = df.loc[df.groupby(['date'])['h2-l1_1d'].idxmin()][['date', 'h2-l1_1d']]


    a1 = pd.merge(a1, a2, on='date')
    a1 = pd.merge(a1, a3, on='date')
    a1 = pd.merge(a1, a4, on='date')

    a1.columns = ['date', 'h1-l2_1d_max', 'h2-l1_1d_max', 'h1-l2_1d_min', 'h2-l1_1d_min']


    a1['earn'] = a1['h1-l2_1d_max'] + a1['h2-l1_1d_max']
    
    
    return df, a1, av_1d, av_2d, av_3d 



