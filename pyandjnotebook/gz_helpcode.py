#!/usr/bin/env python
# coding: utf-8

#Просто кусочки кода чтобы не забывать

#расчет позиций по бета (Крейл) проверочные данные
    
#Tsize = 20000
#llong = {'ticker' : ['Weir', 'Mulberry', 'Burberry', 'Imagination', 'Man', 'Arm'], 
#         'price':[16.20,13.00,13.00,4.70,0.73,5.20],
#         'beta' : [1.59,1.78,1.36,1.01,0.95,0.72]}
#
#lshort = {'ticker' : ['Home Retal', 'ITV', 'Talk Talk', 'Nat Grid'],
#          'price' : [0.99,1.99,4.99,5.99,],
#          'beta': [0.99,1.98,0.35,0.49]}
#
#dflong = pd.DataFrame(llong)
#dfshort = pd.DataFrame(lshort)
#dflong
#dfshort
#create_position(dflong, dfshort, Tsize)

#long_tics = ['Weir', 'Mulberry', 'Burberry', 'Imagination', 'Man', 'Arm']
#short_tics = ['Home Retal', 'ITV', 'Talk Talk', 'Nat Grid']

#long_betas = [1.59,1.78,1.36,1.01,0.95,0.72]
#short_betas = [0.99,1.98,0.35,0.49]

#long_prices = [16.20,13.00,13.00,4.70,0.73,5.20]
#short_prices = [0.99,1.99,4.99,5.99,]

#dfl_kreil, dfs_kreil = create_position2(long_tics, short_tics, 
#                                        long_betas, short_betas, 
#                                        long_prices, short_prices, 
#                                        20000)

    
#коинтеграция
#from statsmodels.tsa.vector_ar.vecm import coint_johansen
#import statsmodels.tsa.stattools as ts 
#ts.coint(dfy['IBM'], dfy['HPE'])
#jres = coint_johansen(dfm[['RTKM','MTSS', 'AFKS']], det_order=0, k_ar_diff=1)    
    
    
#примеры


#df = df[df['BOARDID'] == 'TQBR']
#len(df)
#df = df.reset_index(drop=True)
#df.reset_index(level=0, inplace=True)
#len(set(dfy['date']).intersection(set(dfm['date'])))
#dfm[dfm['date'].duplicated()]
#df3 = dmm[dmm['BOARDID'] == 'TQBR']
#df4 = df3[~df3[['TRADEDATE']].duplicated()]
#dfmoex.describe()[['AFLT']]
#dfm = read_stock_moex('RTSI', t1, t2, 'RTSI')
#merge_df = df if type(merge_df) is int else pd.merge(merge_df, df, on='date')

#vlt = [(x, volatility_ret(dfmoex, x).std()) for x in ru_tickers]
#vv = sorted(vlt, key=lambda v: v[1], reverse=True)

#dfa['timestamp'] = dfa['date'].apply(lambda x: time.mktime(x.timetuple()))

#corr = dfmoex.corr()
#corr[corr.iloc[:] < 0.8] = '< 0.8'
#corr

#dfrtsi = read_stock_moex('RTSI', t1, t2, 'RTSI')
#dfmoex = pd.merge(dfmoex, dfrtsi, on='date')

#corr.loc[:, 'RTSI']

#print('Intercept: {0} {1} {2:.2f}'.format(1, 2, 3))

#lpred = linear_regr_predict_wdate(dfmoex['date'], dfmoex['RTSI'])
#plot_wdate(dfmoex['date'], dfmoex['RTSI'])
#plt.plot(dfmoex['date'],lpred)

#for x in ru_tickers:
    #print('Len yahoo:', len(dfyahoo), 'moex:', len(dfmoex))

#plot_LRpredict(dfmoex[dfmoex['date'] > '2020-4-1'], 'MVID')
#plot_LRpredict(dfmoex[dfmoex['date'] > dt.datetime(2020,4,1)], 'SBER')

#a = read_raw_stock_yahoo('SBER', t1, now)
#b = read_raw_stock_moex('SBER', t1, now)
#a = a.rename(ren_yahoo, axis='columns')
#b = b.rename(ren_moex, axis='columns')
#drop_a = [v for i,v in ren_yahoo.items() if v not in remain] 
#drop_b = [v for i,v in ren_moex.items() if v not in remain] 
#a.drop(drop_a, inplace=True, axis=1)
#b.drop(drop_b, inplace=True, axis=1)

#dfm2 = dfm.copy()
#_col = {x:x.replace('_close', '') for x in dfm.columns if 'close' in x}
#dfm2 = dfm2.rename(close_col, axis='columns')
#dfm3 = dfm2[[x for x in dfm2.columns if '_' not in x]]
#dfatr_sber.iloc[0:5][['ATR2']]
#if dfm['date'].dtype == 'datetime64[ns]':
#plot_LRpredict2(dfm['date'], dfm['AFKS_close']/dfm['AFKS_high'], 'AFKS')
#plot_LRpredict2(dfm['AFKS_close'], dfm['SBER_close'], 'SBER')
#l = {y : LR_wdatenormtomax(dfm[-60:]['date'], dfm[-60:][y]) for y in [x + '_close' for x in ru_tickers]}          
#pl =pd.DataFrame.from_dict(l, orient='index', columns=['knorm'])
#pl.sort_values(by=['knorm'], ascending=False)
#sorted(l.items(), key=lambda x:x[1], reverse=True)

#col2 = [x for x in dfm.columns if '_close' in x or x == 'date']
#dfm.loc[:, col2].corr().sort_values(by='AFKS_close')
#[plot_LRpredict3( atr(dfm, x), x + '_ATR', ses='none')  for x in ru_tickers]

#dfm = read_multiple_opencloselowhigh_ru_moex(ru_tickers, t1, now)
#dfrtsi = read_opencloselowhigh_ru_stoc('moex', 'RTSI', t1, now, cols_rename_moex, cols_out_2, boardid='RTSI')
#dfusdrub = read_opencloselowhigh_ru_stoc('moex', 'USD000UTSTOM', t1, now, cols_rename_moex, cols_out_2, boardid='CETS')
#ren = [x.replace('USD000UTSTOM', 'USDRUB') for x in dfusdrub.columns]
#dfusdrub.columns = ren
#dfeurrub = read_opencloselowhigh_ru_stoc('moex', 'EUR_RUB__TOM', t1, now, cols_rename_moex, cols_out_2, boardid='CETS')
#ren = [x.replace('EUR_RUB__TOM', 'EURRUB') for x in dfeurrub.columns]
#dfeurrub.columns = ren

#dfrtsi = read_opencloselowhigh_ru_stoc('moex', 'RTSI', t1, now, cols_rename_moex, cols_out_2, boardid='RTSI')
#dfimoex = read_opencloselowhigh_ru_stoc('moex', 'IMOEX', t1, now, cols_rename_moex, cols_out_2, boardid='SNDX')
#dfrvi = read_opencloselowhigh_ru_stoc('moex', 'RVI', t1, now, cols_rename_moex, cols_out_2, boardid='RTSI')

#dfusdrub = read_opencloselowhigh_ru_stoc('moex', 'USD000UTSTOM', t1, now, cols_rename_moex, cols_out_2, boardid='CETS')
#ren = [x.replace('USD000UTSTOM', 'USDRUB') for x in dfusdrub.columns]
#dfusdrub.columns = ren
#dfeurrub = read_opencloselowhigh_ru_stoc('moex', 'EUR_RUB__TOM', t1, now, cols_rename_moex, cols_out_2, boardid='CETS')
#ren = [x.replace('EUR_RUB__TOM', 'EURRUB') for x in dfeurrub.columns]
#dfeurrub.columns = ren
#dfeurusd = read_opencloselowhigh_ru_stoc('moex', 'EURUSD000TOM', t1, now, cols_rename_moex, cols_out_2, boardid='CETS')
#ren = [x.replace('EURUSD000TOM', 'EURUSD') for x in dfeurusd.columns]
#dfeurusd.columns = ren

#dfsp500 = read_opencloselowhigh_orig_stoc('yahoo', '^GSPC', t1, now, cols_rename_yahoo, cols_out_2)
#dfsp500.columns = ['date', 'SP500_high', 'SP500_low', 'SP500_open', 'SP500_close']

#dfdax = read_opencloselowhigh_orig_stoc('yahoo', '^GDAXI', t1, now, cols_rename_yahoo, cols_out_2)
#dfdax.columns = ['date', 'DAX_high', 'DAX_low', 'DAX_open', 'DAX_close']

#dfbtc = read_opencloselowhigh_orig_stoc('yahoo', 'BTC-USD', t1, now, cols_rename_yahoo, cols_out_2)
#dfbtc.columns = ['date', 'BTC_high', 'BTC_low', 'BTC_open', 'BTC_close']

#dfetf = read_multiple_opencloselowhigh_ru_moex(['SBSP', 'VTBA', 'FXIT', 'FXUS'], t1, now, boardid='TQTF')



#dfm[[x for x in dfm.columns if 'USD' in x or 'date' in x]]
#dfc = read_multiple_opencloselowhigh_orig_yahoo(crypto_yahoo, t1, now)

#gm = set(dfra['date'].datetime.month)
#gy = set(dfra['date'].datetime.year)

#x = pd.DataFrame({'a':[1, 5, 9, 3, 5]})
#y = pd.Series([6,7,8,9,10])

#slr = LinearRegression()
#slr.fit(x, y)

#print('Slope: {:.2f}'.format(slr.coef_[0]))
#print('Intercept: {:.2f}'.format(slr.intercept_))

#corr = dfmoex.corr()
#for i, row in corr.iterrows():
#    for j, x in row.iteritems():
#        if abs(x) > 0.95 and abs(x) < 1.0 and i != j:
#            print('{0} - {1} corr: {2:.2f}'.format(i, j, x))

#plot_LRpredict(dfmoex[dfmoex['date'] > '2020-4-1'], 'MVID')
#to_rat = corr[corr['MVID'] > 0.6]['MVID'].index.tolist()

#dfm[-1:][['date', 'AFKS_close']]
#diff_LRstd(dfm[-20:], 'ROSN_close')
#plot_norm_stocks(dfm[-30:], 'ROSN', 'LKOH', 'TATN')
#plot_rel(dfm[-30:], 'ROSN', 'LKOH', 'TATN', 'SNGS', 'IMOEX')

#d = pd.DataFrame({'a':[1,2,3], 'b':[4,52,6]}, index=['dd', 'ff', 'ee'])
#for i, row in d.iterrows():
#    print(i)
#dfm.filter(like='AFKS')
#plot_stocks(dfm, 'AFKS_open', 'SBER_open', ses='fullname', mode='norm')
#tics = [x for x in dfm.columns if 'close' in x]
#kv = [[x, coef_var2(dfm, x, ses='fullname')] for x in tics]
#corr = dfm[tics].corr()
#a = list(filter(lambda x: 'close' in x, list(dfm20.columns)))
#p = permutations(a,2)
#b = [[x, coef_var(dfm20['date'], dfm20[x[0]] / dfm20[x[1]])] for x in p]
#dfm['SBER_close'].pct_change().dropna()
#col2 = [x for x in dfm.columns if '_close' in x or x == 'date']
#dfm.loc[:, col2].corr().sort_values(by='AFKS_close')

#dfm2 = dfm[[x for x in dfm.columns if 'close' in x or x == 'date' ]].copy()
#dfm2.columns = [x.replace('_close', '') for x in dfm2.columns]
#corr = dfm2.corr()
#beta = corr.copy()

#corr[corr['AFKS'] > 0.9][['AFKS']]

#cvar_dict = {x:coef_var2(dfm, x) for x in corr.columns}
#cvar_dict['SBER']

#corr[corr['AFKS'] > 0.8][['AFKS']].values.tolist()
#corr[corr['AFKS'] > 0.8][['AFKS']].to_dict()
#for a , b in product(k1, k2):
#    print(a, b)

#[(x,y, dfcorr.loc[x,y]) for x, y in product(dfcorr.index, dfcorr.columns) if dfcorr.loc[x,y] > 0.9]
#dfcorr[dfcorr > 0.4]

#вычисление корреляций, бета, ковариаций

#calc_days = 50

#dfmtemp = dfm[[x for x in dfm.columns if 'close' in x or 'date' in x]]
#dfmtemp = dfmtemp.copy()
#dfmtemp.columns = [x.replace('_close','') for x in dfmtemp.columns]
#dfcorr = dfmtemp.corr()
#dfbeta = corr.copy()

#for x,y in product(dfbeta.index, dfbeta.columns):
#    dfbeta.loc[x, y] = beta2(dfmtemp[calc_days:], x, y, ses='fullname')
    
#calc_days = 50
#coefvar_dict = {x:coef_var2(dfmtemp[calc_days:], x, ses='fullname') for x in dfcorr.columns}
#coefvar_dict
#dfcoefvar = pd.DataFrame(coefvar_dict, index=['coefvar'])
#dfcoefvar.T #траснпонируем есл нужно   

#sum(cl1, [])
#sum([['a']], [])

#dfcorr[pd.isna(dfcorr[dfcorr > 0.9])]

#df_5mf[df_5mf['time'].dt.date == datetime.date(2021, 1, 4)]
#dftemp.loc[dftemp.groupby(['date'])['time'].idxmax()] #группировка по дате с поиском макс датывремени

#dftemp['time'].apply(lambda x: x.replace(hour=0, minute=0,  second=0))
#pd.Timestamp(dftemp['time'][0].replace(hour=0, minute=0, second=0))

#dftemp.rename({'c_1_y':'close'}, axis='columns', inplace=True)

#Excel не понимает дат с зонами поэтому надо убрать
#dftemp3['time'] = dftemp3['time'].dt.tz_localize(None)
#dftemp3['date'] = dftemp3['date'].dt.tz_localize(None)
#dftemp3.to_excel('_222.xlsx')

#замер времени
#t1 = time.monotonic()
#
#t2 = time.monotonic()
#print(t2-t1)

#for x in range(0,10):
#    print('\r', x, end='')
#    #sys.stdout.flush()   

#return_risk_portfolio(dfy, ['AAPL','FB', 'GE', 'GM', 'WMT'],
#                           ['TSLA', 'F', 'MRNA', 'T', 'V'],
#                           [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])


#return_risk_portfolio(dfy, ['AAPL','FB', 'GE', 'GM', 'WMT', 'TSLA', 'F', 'MRNA', 'T', 'V'], 
#                           [],
#                           [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])


#stock = yf.Ticker('AAPL')
#stock.info
#stock.financials

#копирование структуры быстро это просто copy
#df2 = df1.copy()

#перестановки без повторений
#[x  for x in combinations(build_house, 5)]

#list(product(combinations(a, 3), combinations(b, 3)))

#optimize_portfolio(dfy, ['BIDU', 'BA'], ['F', 'T']) #, bounds_t = [(0.1, 1)] * 5)
#df_rrs  = return_risk_combination_df(dflogmed[-75:], long1, short1, 5, 5) #долго
#isinstance(dfm['date'][0], pd.Timestamp)
#random.sample(us_tics, 10)
#df.to_dict()
#list(dict_yfinfo['AAPL'].keys())

#l_sector = [(k,v['sector'], v['industry']) for (k,v) in yfinfo.items() if 'sector' in v]
#df_sector = pd.DataFrame(l_sector, columns=['tic', 'sector', 'industry'])

#df_sector.sort_values(by='sector')
#df_sector.groupby('sector').count()
#l_industry = df_sector.groupby('industry').count().index
#sorted(l_industry)

#df_grouped = df_sector.groupby('industry') #.count()
#df_grouped.get_group('Airlines')

#df_sector.groupby('industry')['tic'].apply(lambda group_series: group_series.tolist()).reset_index()
#df_sector.groupby('sector')['tic'].apply(lambda group_series: group_series.tolist()).reset_index()
#list(df_sector.groupby('industry').groups.keys())
#[name for name,unused_df in df_sector.groupby('industry')]

#ddd = df_sectortics.set_index('sector').T.to_dict('records')[0] #'list')
#ddd['Industrials']
#df_sector[df_sector['tic'].isin(l_reit_short)]

#преобр. из date в timestamp (секунды)
#x = (date_ymd.apply(lambda z: time.mktime(z.timetuple()))).values.reshape(-1, 1) 
#x = (x - x.min()) / 86400 #(3600 * 24)  c началом в нуле и в днях

#{x:list(set(f1) & set(y)) for x,y in d_industry_tics.items() if len(set(f1) & set(y)) > 0}


#беты обычно usa считаются за 3 года на основен месяных закрытий

#беты по отношению к IMOEX

#dfmi3y_beta = dfmi3y_corr.loc[:,['IMOEX']].copy()
#dfmi3y_beta.iloc[:,:] = 1.00
#for x,y in product(dfmi3y_beta.index, ['IMOEX']):
#    dfmi3y_beta.loc[x,y] = round(beta(dfmi3y, x, y), 2)


#столбцы в которых есть NaN
#dfy.columns[dfy.isnull().any()].tolist()
#dfy.loc[:, dfy.isnull().any()]
#строки с NaN
#dfy[dfy.isnull().any(axis=1)][['date', 'TOT']]

#сортировка словаря по значению
#dict(sorted(d_betasp500.items(), key=lambda item: item[1]))































