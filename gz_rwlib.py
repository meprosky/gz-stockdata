#!/usr/bin/env python
# coding: utf-8

import os, fnmatch, time, sys
import numpy as np
import pandas as pd
import string
import unicodedata as ud
import datetime
from pandas_datareader.data import DataReader
from xlsxwriter import Workbook
import xlsxwriter
from gz_mainlib import *
from gz_const import *
from threading import Thread, Event, Lock
from queue import Queue
from time import sleep
import yfinance as yf
import json
import requests

rw_ver = '67'

# In[8]:
def rwver():
    print('RW ver:', rw_ver)

def xlcoor(row, col):
    return xlsxwriter.utility.xl_rowcol_to_cell(row, col)
    
def xlc_formula(df, col_name, offset_row, offset_col, func='SUM', argadd = ''):
    
    pos_row = offset_row + len(df)
    pos_col = offset_col + df.columns.get_loc(col_name)
    
    a1b1 = xlcoor(offset_row + 1, pos_col) + ':' + xlcoor(offset_row + len(df), pos_col)
    
    fstring = '=' + func + '(' + a1b1 + argadd + ')'
    
    return fstring, pos_row + 1, pos_col
    
def xlwrite_df(ws, df, offset_row, offset_col, colformats):
    
    crow = offset_row
   
    #write header
    for i, col_name in enumerate(df.columns):
        ws.write(offset_row, offset_col + i, col_name)
    
    for index, row in df.iterrows():
        crow += 1
        for i, val in enumerate(row):
            colname = df.columns[i]
            if colname in colformats:
                ws.write(crow, offset_col + i, val, colformats[colname])
            else:
                ws.write(crow, offset_col + i, val)

            
    return [offset_row, offset_col, offset_row + len(df), offset_col + len(df.columns) - 1]


#запись в excel табл. расчета позиций по бета
def xlwrite_betapos(file, total_val, dflong, dfshort, offset_row=2, offset_col=0):
    
    wb = Workbook(file)
    ws = wb.add_worksheet("New Sheet")
    
    bold = wb.add_format({'bold': True})
    
    num2_format = wb.add_format({'num_format': '0.00'})
    num2bold = wb.add_format({'num_format': '0.00', 'bold': True })
    
    num6_format = wb.add_format({'num_format': '0.000000'})
    num0_format = wb.add_format({'num_format': '0.00'})
    
    prc_format = wb.add_format({'num_format': '0.00%'})

    #'price':num6_format, 
    #colformats = {'beta':num2_format, 'corrindex':num2_format, 'corrshort':num2_format, 
    #              'b2wb':num2_format, 'b2sb':num2_format, 'rsum1':num2_format, 
    #              'pos':num2_format, 'pos%':num2_format, 
    #              'shares':num2_format,
    #              'val':num2_format, 'val2':num2_format, 'pval':num2_format}
    
    #['npp', 'tic', 'beta', 'pos', 'pos%', 'ticprice', 'vol', 'ntics']

    colformats = {'beta':num2_format, 
                  'pos':num2_format, 'pos%':num2_format, 
                  'ticprice':num2_format, 'vol':num2_format,'ntics':num2_format}
    
    crow = offset_row
    ccol = offset_col
    
    ws.write(crow, ccol, 'Exposure:', bold)
    ws.write(crow, ccol+1, total_val, bold)
    
    crow += 2
    
    ws.write(crow, ccol, 'Long', bold)
    
    crow += 1
    
    box_long = xlwrite_df(ws, dflong, crow, ccol, colformats)
    
    f1, rf1, cf1 = xlc_formula(dflong, 'vol', box_long[0], box_long[1])
    
    #f2, rf2, cf2 = xlc_formula(dflong, 'val2', box_long[0], box_long[1])
    #f3, rf3, cf3 = xlc_formula(dflong, 'shares2', box_long[0], box_long[1])
    
    ws.write_formula(xlcoor(rf1, cf1), f1, num2bold)
    
    #ws.write_formula(xlcoor(rf2, cf2), f2, num2bold)
    #ws.write_formula(xlcoor(rf3, cf3), f3, bold)
    
    crow += len(dfshort) + 1
      
    ws.write(crow, ccol, 'Total long', bold)

    crow += 3
    
    ws.write(crow, ccol, 'Short', bold)
    
    crow += 1
    
    box_short = xlwrite_df(ws, dfshort, crow, ccol, colformats)
    
    f4, rf4, cf4 = xlc_formula(dfshort, 'vol', box_short[0], box_short[1])
    
    #f5, rf5, cf5 = xlc_formula(dfshort, 'val2', box_short[0], box_short[1])
    #f6, rf6, cf6 = xlc_formula(dfshort, 'shares2', box_short[0], box_short[1])
    
    ws.write_formula(xlcoor(rf4, cf4), f4, num2bold)
    
    #ws.write_formula(xlcoor(rf5, cf5), f5, num2bold)
    #ws.write_formula(xlcoor(rf6, cf6), f6, bold)
    
    crow += len(dfshort) + 1
      
    ws.write(crow, ccol, 'Total short', bold)
    
    crow += 2
    
    ws.write(crow, ccol, 'TOTAL', bold)
    ws.write_formula(crow, cf1, '=' + xlcoor(rf1, cf1) + '+' + xlcoor(rf4, cf4), num2bold)
    
    #ws.write_formula(crow, cf2, '=' + xlcoor(rf2, cf2) + '+' + xlcoor(rf5, cf5), num2bold)
    #ws.write_formula(crow, addr2[1], '=SUM(' + xl_range2 + ')', bold)
    
    wb.close()

def xlwrite_longshortsprade(file, total_exposure, 
                            long_tics, short_tics,
                            long_entry, short_entry,                               
                            offset_row=0, offset_col=0):
    
    wb = Workbook(file)
    ws = wb.add_worksheet("New Sheet")
    
    bold = wb.add_format({'bold': True})
    num2 = wb.add_format({'num_format': '0.00'})
    num2bold = wb.add_format({'num_format': '0.00', 'bold': True })
    prc = wb.add_format({'num_format': '0.00%'})
    
    #current_long = (dfm.loc[dfm['date'] == date_calc][[x+'_close' for x in long_tics]]).iloc[0].tolist()
    #current_short = (dfm.loc[dfm['date'] == date_calc][[x+'_close' for x in short_tics]]).iloc[0].tolist()
    
    
    header = ['npp', 'Longs', 'Shorts', 'Long Entry', 'Short Entry', 'Spread', 'bbbb1',
             'Long Entry copy', 'Short Entry copy', 'Spread copy', 'bbbb2',
             'LongCurrent', 'ShortCurrent', 'SpreadCurrent',
              'Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6', 'Week 7']
    
    
    crow = offset_row
    ccol = offset_col
    
    ws.write(crow, ccol, 'Exposure:', bold)
    
    crow += 1
    
    ws.write(crow, ccol, total_exposure, bold)
    
    crow += 2
    
    for i, col_name in enumerate(header):
        ws.write(crow, ccol + i, col_name, bold)
    
    len_max = max(len(long_tics), len(short_tics))
    
    crow += 1
    
    for x in range(1, len_max + 1):
        ws.write(crow + x, ccol + header.index('npp'), x)
    
    for i, x in enumerate(long_tics, 1):
        ws.write(crow + i, ccol + header.index('Longs'), x)
        
    for i, x in enumerate(short_tics, 1):
        ws.write(crow + i, ccol + header.index('Shorts'), x)
        
    for i, x in enumerate(long_entry, 1):
        ws.write(crow + i, ccol + header.index('Long Entry'), x)
        
    for i, x in enumerate(short_entry, 1):
        ws.write(crow + i, ccol + header.index('Short Entry'), x)
        
    #for i, x in enumerate(long_shares, 1):
    #    ws.write(crow + i, ccol + header.index('Long Shares'), x)
        
    #for i, x in enumerate(short_shares, 1):
    #    ws.write(crow + i, ccol + header.index('Short Shares'), x)
  
    for i in range(len_max):
        ws.write_formula(crow + i + 1, ccol + header.index('Spread'), 
                 '=' + xlcoor(crow + i + 1, ccol + header.index('Long Entry')) + '/' + \
                       xlcoor(crow + i + 1, ccol + header.index('Short Entry')))      
 
        
    wb.close()

def invest_idea_usd(df, lt, st, date_entry, datecalc, vol_long_entry=-1, vol_short_entry=-1):
    
    df_lt = round(df[(df['date'] == date_entry) | (df['date'] == datecalc)][lt],2)
    df_st = - round(df[(df['date'] == date_entry) | (df['date'] == datecalc)][st],2)
    
    df_lt = (df_lt.T).reset_index()
    df_lt['ls'] = 'long'
    df_st = (df_st.T).reset_index()
    df_st['ls'] = 'short'
    
    dfc = df_lt.append(df_st).reset_index(drop=True)
    
    #dfc['ns'] = 1
    
    dfc.columns = ['tic','entry', 'curr', 'ls']
    
    dfc['ns'] = 1.0
    dfc['nsr'] = 1.0
    
    if vol_long_entry > 0:
        pertic = vol_long_entry / len(lt)
        dfc.loc[dfc['ls']=='long','nsr'] = round(pertic / dfc[dfc['ls']=='long']['entry'], 2)
        
    #dfc.loc[dfc['ls']=='short','ns'] = 1.0    
    
    if vol_short_entry > 0:
        pertic = vol_short_entry / len(st)
        dfc.loc[dfc['ls']=='short','nsr'] = round(pertic / dfc[dfc['ls']=='short']['entry'], 2)
    
    dfc['ns'] = round(dfc['nsr'],0)
    dfc.loc[dfc['ns'] == 0, 'ns'] = 1.0 
            
    dfc['ventry'] = dfc['entry'] * np.abs(dfc['ns'])
    dfc['vcurr'] = dfc['curr'] * np.abs(dfc['ns'])
    
    dfc['earn'] = round((dfc['vcurr'] - dfc['ventry']),2)
    dfc['earn%'] = round(dfc['earn'] / np.abs(dfc['ventry'])  * 100, 2)
    
    #ddfc = dfc.sort_values(by='earn%')
    ddfc = dfc[['tic', 'entry', 'curr', 'ns', 'nsr', 'ventry', 'vcurr', 'ls', 'earn', 'earn%']]
    
    ddfc = ddfc.sort_values(by='earn%')
    
    ddfc = ddfc.reset_index(drop=True)
    
    return ddfc #.sort_values(by='earn%')

def read_frtin_tics(file):
    with open(file, encoding='utf-8', mode='r') as f:
        txt = f.read()

    txt = txt.split('\n')

#    d = {x: \
#         [int(y.replace(' шт.','').replace('\N{MINUS SIGN}', '-')),
#          float(z.replace(' $','').replace(',', '.').replace(' ', '').replace('\N{MINUS SIGN}', '-')),
#          float(x1.replace(' $','').replace(',', '.').replace(' ', '').replace('\N{MINUS SIGN}', '-')),
#          float(y1.replace('%','').replace(',', '.').replace(' ', '').replace('\N{MINUS SIGN}', '-'))] \
#         for x,y,z,x1,y1 in zip(txt[1::7], txt[2::7], txt[3::7], txt[5::7], txt[6::7])}
 
    d = {}
    for name, tic, entry, curr, currvol, ns, earn, earnproc in zip(txt[0::8], txt[1::8],
                                                                   txt[2::8], txt[3::8], 
                                                                   txt[4::8], txt[5::8], 
                                                                   txt[6::8], txt[7::8]):
        entry1 = float(entry.replace(' $','').replace(',', '.').replace(' ', ''))
        ns1    = int(ns.replace(' шт.',''))
        d.update({tic:[entry1, ns1]})

    df = pd.DataFrame(d).T
    
    return (list(df.index), df)

def read_invests_usd(file, datecalc, df):

    tics_frtin, dfc = read_frtin_tics(file)
    
    #dfc['price_entry'] = abs((dfc[1] - dfc[2])/dfc[0])
    
    dfc.reset_index(inplace=True)
    
    dfc.columns = ['tic', 'entry', 'ns']
    
    dfc = dfc[['tic', 'ns', 'entry']]
    
    #dfc['price_entry'] = dfc[0]
    
    
    dftemp = df[df['date'] == datecalc]
    dftemp = (dftemp.T).reset_index()
    dftemp.columns = ['tic', 'dateprice']
    
    #dfc['price_current'] = abs(dfc[1]/dfc[0])
    
    #dfc.columns = ['tic', 'ns', 'vtin', 'etin', 'etin%', 'entry', 'curr']
    #dfc.columns = ['tic', 'entry', 'ns']
    
    ticlist = dfc['tic'].tolist()
    
    dfc = pd.merge(dfc, dftemp, on='tic')
    dfc['entry'] = np.round(dfc['entry'],2) #dfc['dateprice'].values.astype(np.float64),2)
    dfc['curr'] = np.round(dfc['dateprice'].values.astype(np.float64),2)
    dfc['ventry'] = dfc['entry'] * dfc['ns']
    dfc['vcurr'] = dfc['curr'] * dfc['ns']
    dfc['ls'] = 0
    dfc.loc[dfc['ns'] > 0, 'ls'] = 'long'
    dfc.loc[dfc['ns'] < 0, 'ls'] = 'short'

    #dfc.drop(['dateprice', 'vtin', 'etin', 'etin%'], inplace=True, axis=1)
    dfc.drop(['dateprice'], inplace=True, axis=1)
   
    dfc['earn'] = (dfc['curr'] - dfc['entry']) * dfc['ns']
    valentry = dfc['entry'] * dfc['ns']
    dfc['earn%'] = np.round(dfc['earn'] / abs(valentry) * 100, 2)
    
    ddfc = dfc.sort_values(by='earn%')
    ddfc = ddfc.reset_index(drop=True)
    
    return ddfc, ticlist



def read_invests_rub(file, datecalc, df):
    with open(file, encoding='utf-8', mode='r') as f:
        txt = f.read()
    
    txt = txt.split('\n')

    d = {x: \
         [int(y.replace(' шт.','').replace('\N{MINUS SIGN}', '-')),
          float(z.replace(' ₽','').replace(',', '.').replace(' ', '').replace('\N{MINUS SIGN}', '-')),
          float(x1.replace(' ₽','').replace(',', '.').replace(' ', '').replace('\N{MINUS SIGN}', '-')),
          float(y1.replace('%','').replace(',', '.').replace(' ', '').replace('\N{MINUS SIGN}', '-'))] \
         for x,y,z,x1,y1 in zip(txt[1::7], txt[2::7], txt[3::7], txt[5::7], txt[6::7])}

    dfc = pd.DataFrame(d).T
    
    dfc['price_entry'] = abs((dfc[1] - dfc[2])/dfc[0])
    
    dftemp = df[df['date'] == datecalc]
    dftemp = (dftemp.T).reset_index()
    dftemp.columns = ['tic', 'dateprice']
    
    dfc['price_current'] = abs(dfc[1]/dfc[0])
    
    dfc.reset_index(inplace=True)
    dfc.columns = ['tic', 'ns', 'vtin', 'etin', 'etin%', 'entry', 'curr']
    
    ticlist = dfc['tic'].tolist()
    
    dfc = pd.merge(dfc, dftemp, on='tic')
    
    dfc['entry'] = np.round(dfc['entry'],2) #dfc['dateprice'].values.astype(np.float64),2)
    dfc['curr'] = np.round(dfc['dateprice'].values.astype(np.float64),2)
    
    dfc['ventry'] = dfc['entry'] * dfc['ns']
    dfc['vcurr'] = dfc['curr'] * dfc['ns']
    
    dfc['ls'] = 0
    dfc.loc[dfc['ns'] > 0, 'ls'] = 'long'
    dfc.loc[dfc['ns'] < 0, 'ls'] = 'short'

    dfc.drop(['dateprice', 'vtin', 'etin', 'etin%'], inplace=True, axis=1)
    
   
    dfc['earn'] = (dfc['curr'] - dfc['entry']) * dfc['ns']
    valentry = dfc['entry'] * dfc['ns']
    dfc['earn%'] = np.round(dfc['earn'] / abs(valentry) * 100, 2)
    
    ddfc = dfc.sort_values(by='earn%')
    ddfc = ddfc.reset_index(drop=True)
    
    return ddfc, ticlist

    
def read_stocks_from_folder(fn):
    merge_df = 0
    for root, dirs, files in os.walk(fn): #'err_cert'):
        for filename in fnmatch.filter(files, '*'):
            fullname = os.path.join(root, filename)
            df = read_stock_finam(fullname, ',')
            ticker = df.ticker.iloc[1]
            drop_col = [x for x in df.columns if x not in ['date', 'close']]
            df.drop(drop_col, axis=1, inplace=True)
            df.columns = ['date', ticker]
            print(ticker)
            merge_df = df if type(merge_df) is int else pd.merge(merge_df, df, on='date')
    return merge_df
    
def read_stockclose_finam(file, dlm):
    df = pd.read_csv(file, delimiter=dlm)
    a = [(x.translate({ord(c):'' for c in "<>"})).lower() for x in df.columns]
    df.columns = a
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    return df
    
def hdf5_closeall():
    tables.file._open_files.close_all()

def hdf5_contain(file):
    with h5py.File(file, 'r') as f:
        l = list(f.keys())
    return l
    
def hdf5_create(file):
    with h5py.File(file, 'w') as f:
        pass
    return file  

def hdf5_delete_dataset(file, dsname):
    with h5py.File(file, 'r+') as f:
        del f[dsname]
    return dsname  

def hdf5_writejson(file, dsname, obj):
    keys = hdf5_contain(file)
    with h5py.File(file, 'r+') as f:
        jsondata = json.dumps(obj)
        if dsname in keys:
            del f[dsname]
        d = f.create_dataset(dsname, data=jsondata)
        name = d.name
    return name
        
def hdf5_readjson(file, dsname):
    with h5py.File(file, 'r') as f:
        jsondata = json.loads(f[dsname][()])
    return jsondata

def hdf5_rewrite(file1, file2):
    
    if file1 == file2:
        print('Error. File names Identical!')
        return 0
    
    hdf5_create(file2)
    keys = hdf5_contain(file1)
    for k in keys:
        if 'df' in k:
            df = pd.read_hdf(file1, k)
            df.to_hdf(file2, k)
        else:
            json_data = hdf5_readjson(file1, k)
            hdf5_writejson(file2, k, json_data)
    return keys

def file_size_mb(fn):
    return round(os.path.getsize(fn) / (1024*1024.0), 3)

