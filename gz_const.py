#константные и вспомогательные данные в основном тикеры акций

from gz_import import *

const_ver = '71'

# In[8]:
def constver():
    print('Const ver:', const_ver)


date_start = datetime.date(2021,1,1)
date_end = datetime.date(2021,1,1)
date_now = datetime.datetime.today().date()
now = date_now


def unic_value(df, col):
    return sorted(list(set(df[col].tolist())))
    #return list(df.groupby(col).groups.keys())

def groupbycol(df, group_col, val_col):
    dfres = df.groupby(group_col)[val_col].apply(lambda group_series: sorted(list(set(group_series.tolist())))).reset_index()
    return dfres      

def hdf5_gz_const_readjson(file, dsname):
    with h5py.File(file, 'r') as f:
        jsondata = json.loads(f[dsname][()])
    return jsondata


#df_instruments = pd.read_hdf('instruments.h5', 'instruments')

df_instruments = pd.read_hdf('dat.h5', 'df_instruments')

yfinfo = hdf5_gz_const_readjson('dat.h5', 'yfinfo')
sp500_tics = hdf5_gz_const_readjson('dat.h5', 'sp500_tics')

l_ticsectorindustry = [(k,v['sector'], v['industry'], v['longBusinessSummary']) for (k,v) in yfinfo.items() if 'sector' in v]
df_sector = pd.DataFrame(l_ticsectorindustry, columns=['tic', 'sector', 'industry', 'longBusinessSummary'])

l_sectors = unic_value(df_sector, 'sector')
l_industries = unic_value(df_sector, 'industry')

df_sector_tics = groupbycol(df_sector, 'sector', 'tic')
df_industry_tics = groupbycol(df_sector, 'industry', 'tic')
df_sector_industries = groupbycol(df_sector, 'sector', 'industry')

d_sector_tics = df_sector_tics.set_index('sector').T.to_dict('records')[0] #'list')
d_industry_tics = df_industry_tics.set_index('industry').T.to_dict('records')[0]
d_sector_industries = df_sector_industries.set_index('sector').T.to_dict('records')[0]

#all_tics = sorted(df_instruments['ticker'].tolist())

all_ru_ticks = sorted(df_instruments[df_instruments['currency'] == 'RUB']['ticker'].tolist())
all_us_ticks = sorted(df_instruments[df_instruments['currency'] == 'USD']['ticker'].tolist())


#ALB литий для электромобилей  WLK химия полиэтилен полимеры  

#WLK Westlake Chemical - международный производитель и поставщик нефтехимических продуктов, полимеров и готовых строительных изделий, которые имеют фундаментальное значение для различных потребительских и промышленных

#URI аренда оборудования в т.ч. строительного

#LEN DHI PHM NVR KBH TMHC LGIH MTH TPH cтроительство жилья

#ипотека TREE (и др. займы)

#MAS товары для дома холдинг в т.ч краски и др.
#CARR климат. оборудование в т.ч. для дома
#TREX композит. настилы перила из перераб. мат.
#OC изоляционные, кровельные и стеклопластиковые композиты, крупнейший в мире производитель композитов из #стекловолокна.
#IBP двери замки изоляция и др. строй товары для дома
#BECN кровельные материалы для жилых и нежилых зданий, а также сопутствующие строительные материалы
#AAON оборудование для отопления, вентиляции и кондиционирования воздуха для коммерческого и жилого использования.
#для новых рынков строительства и замены


#APH - Amphenol Corporation — американская компания, крупнейший производитель электрических и 
#оптоволоконных соединителей, кабелей и соединительных систем

#DLB Dolby Laboratories, Inc. — американская компания в области систем обработки звука, звукозаписи и
#звуковоспроизведения

#HUBB разрабатывает, производит и продает электрические и электронные продукты для нежилого и жилого строительства, #промышленного и коммунального назначения

#AYI освещение продукты и управление

#AEIS - Advanced Energyтехнологии питания и управления для производства полупроводников, плоскопанельных дисплеев, 
#устройств хранения данных, промышленных покрытий, медицинских приборов, солнечных батарей и архитектурного стекла


#MLM Martin Mariettaвходящая в индекс S & P 500. Компания является поставщиком агрегатов и 
#тяжелых строительных материалов, с операциями, охватывающими 26 государств, Канаду и Карибский бассейн. 
#В частности, Мартин Мариетта предоставляет ресурсы для дорог, тротуаров и фундаментов.

#VMC Vulcan Materials Company. В основном она занимается производством, распространением и продажей строительных
#материалов. Vulcan - крупнейший производитель строительных материалов, в первую очередь гравия, щебня и песка

#EXP Eagle Materials Inc. Компания производит цемент, бетон, строительный заполнитель, гипс, стеновые плиты, картон и
#песок для гидроразрыва пласта.

#NEE NextEra Energy, Inc. — энергетическая компания

#D - Dominion Resources Inc. — американская электроэнергетическая компания, занимающаяся электроснабжением
#в Виргинии и Северной Каролине, а также газоснабжением в Западной Виргинии, Огайо, ..

#XEL коммунальная холдинговая компания

#LNT Alliant Energy Corporation - коммунальная холдинговая компания,

#intrs long short 07.04.2021

#intrs 80 daysess
#0.5506870864719314 0.19621710727026553
#['LEN', 'DHI', 'LGIH', 'PHM', 'KBH']
#['SO', 'PCG', 'EIX', 'CNP', 'NEE']

#intrs 100 daysess
#0.5041637231303084 0.20238701863117717
#['LEN', 'KBH', 'LGIH', 'PHM', 'DHI']
#['ED', 'D', 'EIX', 'SO', 'CNP']

#intrs 60 daysess
#0.6127672686601972, 0.20567684502461234
# ['LGIH', 'LEN', 'MTH', 'DHI', 'PHM']
# ['ED', 'CNP', 'EIX', 'D', 'PPL']

#intrst for SHORT
#'RCL' 'Royal Caribbean Group operates as a cruise company worldwide. 
# The company operates cruises under the Royal Caribbean International, 
#Celebrity Cruises, Azamara, and Silversea Cruises


#Индустрия культуры, массового спорта и развлечений (Arts, Entertainment, and Recreation)
#Фитнес-клубы, прачечные, парикмахерские, массажные и косметические салоны (Personal and Laundry Services)
#Индустрия кино и звукозаписи (Motion Picture and Sound Recording Industries)
#Кафе, бары, рестораны (Food Services and Drinking Places)
#Гостиницы, отели, мотели (Accommodation)
#Производство одежды и обуви (Apparel)

#s = yf.Ticker, s.info

l_reits_forshort = ['AMT', 'ARE', 'BXP', 'MAC', 'O', 'SPG', 'VTR', 'WELL']
l_utilities_forshort = ['CNP', 'D', 'ED', 'EIX', 'EXC', 'NEE', 'NRG', 'PCG', 'PPL', 'SO']
l_fooddistr_forshort = ['CHEF', 'PFGC', 'USFD']
l_oilgas_forshort = ['APA', 'BKR', 'CHX', 'CLR', 'COP', 'CVX', 'DVN', 'EOG', 'EQT', 'ET', 
                     'ETRN', 'HAL', 'KMI', 'MPC', 'OII', 'OKE', 'OVV', 'OXY', 'PBF', 'PSX', 
                     'PUMP', 'REGI', 'RRC', 'SLB', 'SWN', 'VLO', 'WMB', 'XOM']

gas_pipe = ['OKE', 'KMI', 'WMB']

#'Specialty Retail'
specialty_retail = ['AAN', 'AAP', 'AZO', 'BBBY', 'BBY', 'DKS', 'EYE', 'FLWS', 'GPC', 
                    'HIBB', 'MUSA', 'ORLY', 'RH', 'SBH', 'SFIX', 'TCS', 'TSCO', 'ULTA', 'WSM']



multipliers = ['marketCap', 'trailingPE', 'forwardPE', 'priceToBook', 'bookValue', 
               'enterpriseToRevenue', 'enterpriseToEbitda', 'forwardEps', 'trailingEps']


#['Electrical Equipment & Parts']
el_equipment = ['AEIS', 'ATKR', 'AYI', 'BDC', 'BMI', 'ENR', 'ENS', 'HUBB']


#'Specialty Chemicals'
specialty_chemicals = ['ALB', 'ASH', 'AVNT', 'AVTR', 'BCPC', 'DD', 'ECL', 'FOE', 'FUL', 'GCP', 'GRA', 'IFF', 'IOSP', 'KWR', 'LIN',                            'LTHM', 'LYB', 'NEU', 'NGVT', 'PPG', 'RPM', 'SHW', 'SXT', 'WDFC', 'WLK']

medicine = ['ABBV', 'ABMD', 'ABT', 'ACAD', 'ALXN', 'AMGN', 'ARWR', 
            'BIIB', 'BMY', 'BSX', 'CAH', 'CNC', 'CVS', 
            'DHR', 'ENDP', 'EVH', 'GILD', 'GOSS', 'HCA', 'INCY', 'JNJ', 
            'LLY', 'MDT', 'MRK', 'MRNA', 'SNY', 'UNH', 'VRTX', 'ZYNE', 'ZYXI']


tobacco_forshort = ['BTI', 'MO', 'PM']

logistic_finviz = ['UPS', 'FDX', 'EXPD', 'JBHT', 'XPO', 'CHRW', 'LSTR', 
                       'FWRD', 'CYRX', 'HUBG', 'ATSG', 'ECHO', 'RLGT', 'STCN', 'SINO', 'AIRT']
logistic = ['XPO', 'HUBG', 'LSTR', 'CHRW', 'JBHT', 'EXPD', 'ECHO', 'UPS', 'FDX']


travel_finviz = ['BKNG', 'CCL', 'EXPE', 'RCL', 'NCLH', 'TRIP', 'TNL', 'LIND', 'TZOO', 'MKGI']
travel = ['EXPE', 'TRIP', 'TNL', 'CCL', 'BKNG', 'RCL']


education_finviz = ['EEIQ', 'ZVO', 'ASPU', 'LINC', 'UTI', 'GPX', 'APEI', 'PRDO', 'HMHC', 'LRN',
            'ATGE', 'STRA', 'LAUR', 'TWOU', 'GHC', 'LOPE', 'COUR', 'CHGG']
education = ['APEI', 'GHC', 'TWOU', 'STRA', 'LOPE', 'ATGE', 'CHGG', 'LRN']

build_house = ['LEN', 'DHI', 'PHM', 'KBH', 'TMHC', 'LGIH', 'MTH', 'TPH']

medicine = ['ABBV', 'ABMD', 'ABT', 'ACAD', 'ALXN', 'AMGN', 'ARWR', 'BIIB', 'BMY', 'BSX', 
            'CAH', 'CNC', 'CVS', 'DHR', 'ENDP', 'EVH', 'GILD', 'GOSS', 'HCA', 'INCY', 'JNJ', 
            'LLY', 'MDT', 'MRK', 'MRNA', 'SNY', 'UNH', 'VRTX', 'ZYNE', 'ZYXI']

iii = ['ALB', 'WLK', 'URI', 'LEN', 'DHI', 'PHM', 'NVR', 'KBH', 'TMHC', 'LGIH', 'MTH', 'TPH']
iiii = ['MAS', 'CARR', 'TREX', 'OC', 'IBP', 'BECN', 'AAON']
iii2 = ['TREE', 'WFC']

#intresting
intrsting1 = ['APH', 'DLB', 'HUBB', 'AYI', 'AEIS', 'ROG', 'VICR', 'ENS', 'ENR', 'BMI']
buldmat1 = ['MLM', 'VMC', 'EXP']

#фонды недвижимости
reit_est1 = ['AMT', 'ARE', 'IRM', 'MAC', 'O', 'SPG', 'VTR', 'WELL']

#for short
#коммуналка энергия газ и т.п.
utilities = ['NEE', 'SO', 'D', 'EXC', 'SRE', 'XEL', 'PEG', 'ES', 
             'WEC', 'AWK', 'ED', 'PCG', 'EIX', 'PPL', 'AEE', 'ETR', 'CMS', 'CNP', 'NRG', 'LNT']

utilities_noshort = ['ETR', 'XEL', 'SRE', 'PEG', 'ES', 'WEC', 'AWK', 'AEE', 'CMS', 'LNT']

#for short
real_estate = ['AMT', 'SPG', 'DLR', 'WELL', 'ARE', 'IRM', 'MAC', 'O', 'SPG', 'VTR']

#for short
cons = ['KMB', 'MCD', 'MDLZ', 'PEP', 'PG', 'PM', 'RACE', 'SYY', 'TJX', 'WMT', 'YUM']

#for short
telecom = ['T', 'VZ']

#for short
med = ['BMY', 'BIIB', 'BSX', 'CAH', 'CVS', 'GILD', 'GOSS', 'MDT', 'MRK', 'SNY', 'VRTX']
oth = ['MMM', 'WM']

us_banks = ['JPM', 'BAC', 'WFC', 'C', 'MS', 'GS', 'USB', 'BK', 'TFC', 'PNC']
    
us_tics_long = ['PPG', 'SHW']    
    
ru_tickers = ['AFKS', 'AFLT', 'AGRO', 'ALRS', 'APTK', 'BANE', 'CBOM', 'CHMF', 'CHMK', 'DSKY',
              'ENRU', 'FEES', 'FIVE',
              'GAZP', 'GMKN', 'HYDR', 'ISKJ', 'IRAO', 'KUZB', 'LKOH', 'LNTA', 'LIFE',
              'MAGN', 'MGNT', 'MOEX', 'MTLR', 'MTSS', 'MVID',
              'NLMK', 'NVTK', 'PHOR', 'PLZL', 'POLY', 'RASP', 'ROSN', 'RTKM', 'RSTI', 'RUAL', 
              'SBER', 'SIBN', 'SNGS', 'TATN', 'TGKA', 'TGKB', 'UPRO', 'USBN', 'VSMO', 'VTBR', 'YNDX']

moex_etf = ['SBSP', 'AKNX', 'AKSP', 'AKEU', 'VTBA', 'FXIT', 'FXUS']
              
crypto_yahoo = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'BCH-USD', 'LINK-USD', 'BNB-USD', 'LTC-USD', 'ADA-USD', 'EOS-USD',
                'XMR-USD', 'TRX-USD', 'XLM-USD', 'NEO-USD', 'XEM-USD', 'MIOTA-USD', 'VET-USD', 'DASH-USD', 
                'ZEC-USD', 'ETC-USD', 'OMG-USD', 'WAVES-USD', 'DOGE-USD']

brent_yahoo = ['BZ=F']
usdrub_moex = ['USD000UTSTOM']

ru_index = ['RTSI'] #BOARDID = RTSI

us_tics_download = \
['AA','AAPL','ABBV','ABMD','ABT','ACAD','ACN','ADBE','ADM','ADS','ADSK','AERI','AIG','ALB',
'ALK', 'AMAT','AMD','AMGN','AMT','AMZN','ANAB','ANET','APA','ARE','ARWR','ATRA','ATVI',
'AVGO','AXP','AYX','BA','BABA','BAC','BBY','BEN','BIDU','BIIB','BJRI','BK','BKNG','BKR','BLL',
'BMY','BSX','BTI','BUD','BXP','BYND','C','CAH','CAT','CCL','CFG','CHEF','CHX','CLF','CLR',
'CMCSA','CME','CNC','CNK','CNP','COP','COTY','CPRI','CRM','CRWD','CSCO','CTXS','CVS','CVX',
'D','DAL','DDOG','DDS','DFS','DHI','DHR','DIS','DISCA','DKNG','DOCU','DOW','DRI','DVN','DXC',
'DXCM','EA','EBAY','ED','EHTH','EIX','EMR','ENDP','ENPH','EOG','EPAM','EQT','ET','ETRN','ETSY',
'EVH','EXC','EXPE','F','FB','FCX','FDX','FNKO','FSLR','FTNT','GD','GE','GILD','GLW','GM','GOOG',
'GOOGL','GOSS','GPN','GPS','GRMN','GS','GT','GTHX','H','HA','HAL','HAS','HBAN','HCA','HD',
'HEAR','HIG','HII','HLT','HOG','HON','HP','HPE','HPQ','HRTX','IBM','IBN','IDXX','ILMN','INCY',
'INGR','INTC','IOVA','IRBT','IRM','IVZ','JCOM','JD','JNJ','JOYY','JP','JPM','JWN','KBH','KEY',
'KHC','KMB','KMI','KO','KR','LEN','LEVI','LGIH','LHX','LLY','LMT','LOW','LPL','LRCX','LTHM',
'LUV','LVS','LYB','LYFT','M','MA','MAC','MAR','MAS','MBT','MCD','MDB','MDLZ','MDT','MET','MFGP',
'MLCO','MMM','MNST','MO','MOMO','MOS','MPC','MRK','MRNA','MS','MSFT','MSTR','MTCH','MU','NDAQ',
'NEE','NEM','NET','NFLX','NKE','NRG','NTAP','NTES','NTNX','NVDA','O','OII','OKE','OMC','ORCL',
'OVV','OXY','PBCT','PBF','PBI','PCG','PEP','PFGC','PG','PHM','PINS','PLAY','PM','PPC','PPL','PRU',
'PSX','PTON','PTR','PUMP','PVH','PYPL','QCOM','QRVO','RACE','RCL','REGI','REGN','RF','RL','ROKU',
'ROST','RRC','RRGB','RTX','SAGE','SAVE','SBUX','SCHW','SEDG','SHI','SIG','SLB','SMAR','SNAP','SNPS',
'SNY','SO','SP','SPCE','SPG','SPGI','SPR','SQ','SRPT','STX','SWN','SYY','T','TAL','TDOC','TER','TGT',
'TJX','TMUS','TRIP','TROW','TSLA','TSM','TSN','TTD','TTM','TTWO','TWLO','TWOU','TWTR','TXN',
'UAA','UAL','UBER','UNH','UPS','UPWK','URI','USB','USFD','V','VALE','VEON','VIAC','VIPS','VLO','VREX',
'VRTX','VTR','VZ','W','WB','WBA','WDC','WELL','WFC','WM','WMB','WMT','WRK','WU','WYNN','XLNX','XOM','XRX',
'YUM','YY','Z','ZGNX','ZM','ZS','ZYNE','ZYXI']

us_tics_short = \
['AA', 'AAPL', 'ABBV', 'ABMD', 'ABT', 'ACAD', 'ACN', 'ADBE', 'ADM', 'ADS', 'ADSK', 'AERI', 'AIG', 'ALB', 'ALK', 'ALXN', 'AMAT', 'AMD', 'AMGN', 'AMT', 'ANAB', 'ANET', 'APA', 'ARE', 'ARWR', 'ATRA', 'ATVI', 'AVGO', 'AXP', 'AYX', 'BA', 'BAC', 'BBY', 'BEN', 'BIIB', 'BJRI', 'BK', 'BKNG', 'BKR', 'BLL', 'BMY', 'BSX', 'BXP', 'BYND', 'C', 'CAH', 'CAT', 'CFG', 'CHEF', 'CHX', 'CLF', 'CLR', 'CMCSA', 'CME', 'CNC', 'CNK', 'CNP', 'COIN', 'COP', 'COTY', 'CPRI', 'CRM', 'CRWD', 'CSCO', 'CTXS', 'CVS', 'CVX', 'D', 'DAL', 'DDOG', 'DDS', 'DFS', 'DHI', 'DHR', 'DIS', 'DISCA', 'DKNG', 'DOCU', 'DOW', 'DRI', 'DVN', 'DXC', 'DXCM', 'EA', 'EBAY', 'ED', 'EHTH', 'EIX', 'EMR', 'ENDP', 'ENPH', 'EOG', 'EPAM', 'EQT', 'ET', 'ETRN', 'ETSY', 'EVH', 'EXC', 'EXPE', 'F', 'FB', 'FCX', 'FDX', 'FNKO', 'FSLR', 'FTNT', 'GD', 'GE', 'GILD', 'GLW', 'GM', 'GOOG', 'GOSS', 'GPN', 'GS', 'GT', 'GTHX', 'H', 'HA', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HEAR', 'HIG', 'HII', 'HLT', 'HOG', 'HON', 'HPE', 'HPQ', 'HRTX', 'IBM', 'IDXX', 'ILMN', 'INCY', 'INGR', 'INTC', 'IOVA', 'IRBT', 'IRM', 'IVZ', 'JCOM', 'JNJ', 'JPM', 'JWN', 'KEY', 'KHC', 'KMB', 'KMI', 'KO', 'KR', 'LB', 'LEVI', 'LHX', 'LLY', 'LMT', 'LOW', 'LRCX', 'LTHM', 'LUV', 'LVS', 'LYB', 'LYFT', 'M', 'MA', 'MAC', 'MAR', 'MAS', 'MCD', 'MDB', 'MDLZ', 'MDT', 'MET', 'MMM', 'MNST', 'MO', 'MOS', 'MPC', 'MRK', 'MRNA', 'MS', 'MSFT', 'MSTR', 'MTCH', 'MU', 'NDAQ', 'NEE', 'NEM', 'NET', 'NFLX', 'NKE', 'NRG', 'NTAP', 'NTNX', 'NVDA', 'O', 'OII', 'OKE', 'OMC', 'ORCL', 'OVV', 'OXY', 'PBCT', 'PBF', 'PBI', 'PCG', 'PEP', 'PFGC', 'PG', 'PINS', 'PLAY', 'PLTR', 'PM', 'PPC', 'PPL', 'PRU', 'PSX', 'PTON', 'PUMP', 'PVH', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'REGI', 'REGN', 'RF', 'RL', 'ROKU', 'ROST', 'RRC', 'RRGB', 'RTX', 'SAGE', 'SAVE', 'SBUX', 'SCHW', 'SIG', 'SLB', 'SMAR', 'SNAP', 'SNPS', 'SO', 'SPCE', 'SPG', 'SPGI', 'SPR', 'SQ', 'SRPT', 'STX', 'SWN', 'T', 'TDOC', 'TER', 'TGT', 'TJX', 'TMUS', 'TRIP', 'TROW', 'TSLA', 'TSN', 'TTD', 'TTWO', 'TWLO', 'TWOU', 'TWTR', 'TXN', 'UAA', 'UAL', 'UBER', 'UNH', 'UPS', 'UPWK', 'USB', 'USFD', 'V', 'VALE', 'VIAC', 'VLO', 'VREX', 'VRTX', 'VTR', 'VZ', 'W', 'WBA', 'WDC', 'WELL', 'WFC', 'WM', 'WMB', 'WMT', 'WRK', 'WU', 'WYNN', 'XOM', 'XRX', 'YUM', 'Z', 'ZGNX', 'ZM', 'ZS', 'ZYNE', 'ZYXI']


ru_tics_nodata = ['GRNT', 'MSST', 'PRFN', 'RUAL_old', 'TORS',
                 'ENPL', 'RUSP', 'FTRE', 'OBUV', 'TRCN', 'DASB', 'PRTK', 'ALNU', 'ALXN', 'VZRZP',
                 'OBUV', 'KBTK', 'CHEP']

ru_tics_new = ['FIXP', 'ENPG', 'GLTR', 'FLOT', 'ETLN', 'MDMG', 'MAIL', 
               'OKEY', 'OZON', 'SMLT', 'POGR', 'ORUP', 'SGZH']

ru_tics_temp = all_ru_ticks + ru_tics_nodata + ru_tics_new
ru_tics_download = set(ru_tics_temp) ^ set(ru_tics_nodata + ru_tics_new)




cols_rename_yahoo = {'Date': 'date',
 'High': 'high',
 'Low': 'low',
 'Open': 'open',
 'Close': 'close',
 'Volume': 'volume',
 'Adj Close': 'adjclose'}

cols_rename_moex = {'TRADEDATE': 'date',
 'BOARDID': 'boardid',
 'SHORTNAME': 'shortname',
 'SECID': 'secid',
 'NUMTRADES': 'numtrades',
 'VALUE': 'value',
 'OPEN': 'open',
 'LOW': 'low',
 'HIGH': 'high',
 'LEGALCLOSEPRICE': 'legalcloseprice',
 'WAPRICE': 'waprice',
 'CLOSE': 'close',
 'VOLUME': 'volume',
 'MARKETPRICE2': 'marketprice2',
 'MARKETPRICE3': 'marketprice3',
 'ADMITTEDQUOTE': 'admittedquote',
 'MP2VALTRD': 'mp2valtrd',
 'MARKETPRICE3TRADESVALUE': 'marketprice3tradesvalue',
 'ADMITTEDVALUE': 'admittedvalue',
 'WAVAL': 'waval',
 'TRADINGSESSION': 'tradingsession'}

cols_out = ['date', 'secid', 'boardid', 'high', 'low', 'open', 'close']
cols_out_2 = ['date', 'high', 'low', 'open', 'close', 'adjclose']

df_colsname_common = ['date', 'high', 'low', 'open', 'close']

cols_name_yahoo = ['date', 'high', 'low', 'open', 'close', 'adjclose']




#list_dfinstr_usticsdownloads = df_instruments[df_instruments['ticker'].isin(us_tics_download)]['ticker'].tolist()
#list_usticsdiff= set(us_tics_download) ^ set(list_dfinstr_usticsdownloads)
#list_usticsdiff







