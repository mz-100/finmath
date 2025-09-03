import math
import re
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats as st

from . import implied_vol
from . import NelsonSiegelSvensson


_months = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
          "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}


def volatility_surface_from_cboe(cboe_file, feds_file=None, min_maturity=None, max_maturity=None):
    # Определение даты загрузки опционов из файла с доской опционов CBOE
    # Дата находится в 3й строке в виде "Date: Month Day, Year at HH:MM AM/PM EDT,Bid..."
    with open(cboe_file, 'r') as f:
        s = f.readlines()[2]
    d = re.search(r'Date:\s*(.+?),Bid', s).group(1).split()
    month = _months[d[0]]
    day = int(d[1][:-1])
    year = int(d[2])
    h = int(d[4].split(':')[0])
    m = int(d[4].split(':')[1])
    if d[5] == 'PM':
        h += 12
    load_date = datetime(year, month, day, h, m)

    # Калибровка модели Нельсона-Сигеля-Свенсона для процентной ставки
    if feds_file is not None:
        nss = NelsonSiegelSvensson.from_feds(load_date, feds_file)
        discount = nss.discount_factor
    else:
        # Нулевая ставка
        discount = lambda _: 1.0
        
    # Загрузка опционов колл и пут и объединение их в одну таблицу
    calls = pd.read_csv(cboe_file, skiprows=4, usecols=(0,4,5,11),
                        names=['maturity', 'bid', 'ask', 'strike'], parse_dates=['maturity'])
    puts = pd.read_csv(cboe_file, skiprows=4, usecols=(0,11,15,16),
                       names=['maturity','strike', 'bid', 'ask'], parse_dates=['maturity'])  
    calls['price'] = (calls.bid + calls.ask)/2
    puts['price'] = (puts.bid + puts.ask)/2
    calls['type'] = 'C'
    puts['type'] = 'P'
    options = pd.concat([calls[calls.price>0], puts[puts.price>0]])
    options['time_to_maturity'] = math.nan
    options['forward_price'] = math.nan
    options['discount_factor'] = math.nan
    options.drop(columns=['bid', 'ask'], inplace=True)

    # Для каждый даты исполнения заполняем столбцы time_to_maturity, forward_price, discount_factor
    for m in options.maturity.unique():
        time_to_maturity = (m - load_date).total_seconds()/365/24/3600
        discount_factor = discount(time_to_maturity)

        # Сначала найдем страйк, ближайший к ATMF (у него разность цен колл и пут минимальна)
        # и найдем форвардную цену через паритет колл-пут
        c = options[(options.type == 'C') & (options.maturity == m)].set_index('strike')
        p = options[(options.type == 'P') & (options.maturity == m)].set_index('strike')
        price_diff = (c.price - p.price).dropna().abs()
        atmf_strike = price_diff.idxmin()
        atmf_call = c.loc[atmf_strike].price
        atmf_put = p.loc[atmf_strike].price
        forward_price = atmf_strike + (atmf_call - atmf_put)/discount_factor
        
        options.loc[options.maturity == m, 'time_to_maturity'] = time_to_maturity
        options.loc[options.maturity == m, 'discount_factor'] = discount_factor
        options.loc[options.maturity == m, 'forward_price'] = forward_price

    # Вычисление IV
    # Берутся опционы OTMF (если какой-то страйк попадает на ATMF, то берем пут)
    options = options[((options.type == 'C') & (options.strike > options.forward_price))
                      | (options.type == 'P') & (options.strike <= options.forward_price)]
    def iv(option):
        return implied_vol(option.forward_price, option.time_to_maturity, option.strike,
                           option.price, 1 if option.type == 'C' else -1, option.discount_factor)
    options['implied_vol'] = options.apply(iv, axis=1)
    options = options.dropna()

    # Вычисление дельты
    def delta(option):
        call_delta = st.norm.cdf(
            (math.log(option.forward_price/option.strike) 
             + 0.5*option.implied_vol**2*option.time_to_maturity)
            /(option.implied_vol*math.sqrt(option.time_to_maturity)))
        if option.type == 'C':
            return call_delta
        else:
            return call_delta - 1
    options['delta'] = options.apply(delta, axis=1)

    return options[(options.time_to_maturity >= min_maturity) 
                   & (options.time_to_maturity <= max_maturity)].sort_values(by=['time_to_maturity', 'strike'])


def choose_from_iv_surface(options, maturities, n_strikes=None, min_call_delta=0, max_put_delta = 0):
    chosen_options = []
    for m in maturities:
        op = options[(options.maturity == m) 
                  & (((options.type == 'C') & (options.delta >= min_call_delta))
                     |((options.type == 'P') & (options.delta <= max_put_delta)))]
        if n_strikes is not None:
            strikes = np.linspace(op.strike.min(), op.strike.max(), n_strikes)
            idx =  np.unique([np.abs(op.strike - k).argmin() for k in strikes])
            chosen_options.append(op.iloc[idx])
        else:
            chosen_options.append(op)
    return pd.concat(chosen_options)
