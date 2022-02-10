#!/usr/bin/env python
# coding: utf-8

from gz_import import *
from gz_mainlib import *

# In[8]:
plot_ver = '67'

def plotver():
    print('Plot ver:', plot_ver)
        
#новые функции отрисовки        
        
def plot_d(x, y, 
           title='', lgnd='y=f(x)', 
           x_intervals=6, g_size = (5,3),  
           scatter=False,
           draw_LRline=True, draw_line0=False, draw_grid=True,
           xaxe_date=True, figax = ()): #x дата
    
    if len(figax) == 0:
        fig, ax = plt.subplots(figsize=g_size)
    else:
        fig = figax[0]
        ax = figax[1]
        
        
    x.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    ax.set_title(title)
    
    def format_date(xaxis, pos=None):
        thisind = np.clip(int(xaxis + 0.5), ind[0], ind[-1])
        return x[int(thisind)].strftime('%Y-%m-%d')
   
    if xaxe_date:
        ind = np.array(x.index)
        interval = (x.index.max() - x.index.min()) // x_intervals
        interval_list = list(range(x.index.min(), x.index.max(), interval)) + [x.index.max()]
        
        ax.xaxis.set_major_formatter(format_date)
        
        if scatter:
            ax.scatter(ind, y, label=lgnd)
        else:
            ax.plot(ind, y, label=lgnd)
        
        loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
        ax.xaxis.set_ticks(interval_list) 
        
        fig.canvas.draw()
    
        labels = ax.get_xticklabels()
        xticks = ax.xaxis.get_major_ticks()
    
        #labels[-2].set_visible(False)
        #xticks[-2].set_visible(False)
    
        ticdelta = labels[1].get_position()[0] - labels[0].get_position()[0] 
        lastdelta = labels[-1].get_position()[0] - labels[-2].get_position()[0]   
        
        if lastdelta < ticdelta / 3:
            labels[-2].set_visible(False)
            xticks[-2].set_visible(False)
            #ax2.set_xticklabels(labels)
            
        if draw_LRline:
            k_slope, r2_score, lpred = LR(ind, y)
            
            #print(lpred[-1], lpred[0], k_slope)
            
            #k_slope = y.mean() / k_slope / len(ind)
            
            k_slope2 =  100 * k_slope / (y.max() - y.min())
            
            #print(lpred[-1], lpred[0], k_slope)
            #lgnd_lpred = 'Knorm%={0:.2f}, R2={1:.3f}'.format(100*k_slope/y.max(), r2_score)
            lgnd_lpred = 'Ksl*100={0:.4f}, R2={1:.3f}'.format(k_slope2, r2_score)
            ax.plot(ind, lpred, label=lgnd_lpred, color='tab:orange')

        if draw_line0:      
            ax.plot(ind, np.zeros(len(x)))
            
        fig.autofmt_xdate()
    else:
        if scatter:
            ax.scatter(x, y, label=lgnd)
        else:
            ax.plot(x, y, label=lgnd)
        ax.locator_params(axis='x', nbins=x_intervals)        
        if draw_LRline:
            k_slope, r2_score, lpred = LR(x, y)
            k_slope2 =  100 * k_slope / (y.max() - y.min())
            #k_slope = y.mean() / k_slope / len(x)
            #lgnd_lpred = 'Knorm%={0:.2f}, R2={1:.3f}'.format(100*k_slope/y.max(), r2_score)
            lgnd_lpred = 'Ksl*100={0:.4f}, R2={1:.3f}'.format(k_slope2, r2_score)
            ax.plot(x, lpred, label=lgnd_lpred, color='tab:orange')
        #ax.plot(x, y, label=lgnd)
    
    ax.legend(loc="best")

    if draw_grid:
        ax.grid()
        
    if len(figax) == 0:
        plt.show()
    
    #else:
    #    fig = figax[0]
    #    ax = figax[1]
    #plt.show() 
    #return (fig, ax) #может понадобиться


def plot_norm_d(df, tics, g_size_d = (10,6), grid=True):
    fig, ax = plt.subplots(figsize=g_size_d)
    #сетку включаем здесь иначе будет вкл/выкл
    if grid: 
        ax.grid() 
    for x in tics:
        plot_d(df['date'], df[x]/df[x].max(), figax = (fig, ax), draw_grid=False, draw_LRline=False, lgnd=x)
        
    plt.show()
    
def plot_spred_d(df, tic1, tics, legend='spred(y=ax+b)', grid=True, g_size_d = (5,3)):
    if type(tics) is str:
        tics = [tics]
    for tic2 in tics:
        spred = LR_spred(df[tic1], df[tic2])
        tit = tic1+'/'+tic2
        plot_d(df['date'], spred, title=tit, lgnd =legend,
               x_intervals=6, g_size=g_size_d, 
               draw_grid=grid, draw_LRline=False, draw_line0=True, scatter=False) 
    
def plot_LR_tics_d(df, tics, grid=False, g_size_d = (5,3), num_interval=6):
    if type(tics) is str:
        tics = [tics]
    for tic in tics:
        plot_d(df['date'], df[tic],
               title=tic, lgnd =tic,
               x_intervals=num_interval, g_size=g_size_d, 
               draw_grid=grid, draw_LRline=True, draw_line0=False, scatter=False)
    
def plot_LR_rel2tic_d(df, tic1, tics, grid=False, g_size_d = (5,3)):
    
    if type(tics) is str:
        tics = [tics]
    
    for tic2 in tics:
        plot_d(df['date'], df[tic1]/df[tic2],
               title=tic1+'/'+tic2, lgnd =tic1+'/'+tic2,
               x_intervals=6, g_size=g_size_d, 
               draw_grid=grid, draw_LRline=True, draw_line0=False, scatter=False)                
        
        
        
        
        
        
        
 