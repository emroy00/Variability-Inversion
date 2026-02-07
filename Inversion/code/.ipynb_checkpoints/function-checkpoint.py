#!/usr/bin/env python
'''
--------- Roy et al. (2025) function script ----------

Jan 29, 2025
Eric M. Roy 
emroy@mit.edu

Overview:

All functions required to reproduce results in Roy et
al. (2025)

------------------------------------------------------
'''
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import os

import scipy.stats
from scipy.stats import norm
from scipy.stats import uniform

def HgConversion(conc):
    '''
    Converts modeled Hg concentration (mol mol-1 dry air) to ng m-3
    
    Input: Hg field or vector
    Output: Hg field or vector (depends on input)
    '''
    
    mw_hg = 200.59 #g/mol
    std_p = 101325 #Pa
    std_T = 273.15 #K
    R = 8.314 #m3 Pa K-1 mol-1
    ng = 1e9 #conversion from grams to nanograms
    ngm3 = conc*mw_hg*std_p*ng/(R*std_T)
    return ngm3

def read_gc(sitename,coords,run='run0045',compiled=False,surf=False):
    '''
    Opens preprocessed GC timeseries csv for single location, as well as accompanying regional concentrations.
    '''
    
    modpath = '/net/fs03/d0/emroy/GCrundir/'
    
    if sitename=='Harvard':
        timezone='America/New_York' 
        
        if compiled==False:
            ds = xr.open_mfdataset(modpath+run+'/OutputDir/GEOSChem.SpeciesConcHF.2019*.nc4').isel(lev=0,ilev=0)
        if compiled==True:
            ds = xr.open_dataset(modpath+run+'/OutputDir/GEOSChem.SpeciesConcHF.Hg0.2019.nc4').isel(lev=0)
        
        ds['SpeciesConcVV_Hg0'] = HgConversion(ds['SpeciesConcVV_Hg0'])
        da = ds.sel(lon=coords[1],lat=coords[0],method='nearest')['SpeciesConcVV_Hg0']
        
        
        df_t = da.to_dataframe().drop(['lat', 'lon','lev'], axis=1)
        df = df_t.tz_localize(tz='UTC').tz_convert('America/New_York').tz_localize(None)
        
        #df = df.tz_localize(tz='UTC')
        #df = df.tz_convert(timezone) #convert time to timezone
        #df_convert = df.tz_convert(None) #move to naive so you can merge with observations
        return ds,df
    
    if sitename=='Chacaltaya':

        eroy_path = '../../Model_data/CHC/GEOSChem/'

        Base   = xr.open_dataset(eroy_path+'Base/GEOSChem.SpeciesConcBaseHeigt_levels.nc4').isel(height=0)
        NoAsgm = xr.open_dataset(eroy_path+'NoAsgm/CHC_5240_Hg.nc4').isel(height=0)
        MDD_2  = xr.open_dataset(eroy_path+'MDD_2/GEOSChem.SpeciesConc.14_15output_levels.nc4').isel(height=0)
        Apr_2  = xr.open_dataset(eroy_path+'Apr_2/GEOSChem.SpeciesConc.14_15output_levels.nc4').isel(height=0)
        Aqp_2  = xr.open_dataset(eroy_path+'Aqp_2/GEOSChem.SpeciesConc.14_15output_levels.nc4').isel(height=0)
        Npun_2 = xr.open_dataset(eroy_path+'Npun_2/GEOSChem.SpeciesConc.14_15output_levels.nc4').isel(height=0)
        Spun_2 = xr.open_dataset(eroy_path+'Spun_2/GEOSChem.SpeciesConc.14_15output_levels.nc4').isel(height=0)
        
        #converting to ng m-3 std conditions
        Base['SpeciesConc_Hg0'] = HgConversion(Base['SpeciesConc_Hg0'])
        NoAsgm['SpeciesConc_Hg0'] = HgConversion(NoAsgm['SpeciesConc_Hg0'])
        MDD_2['SpeciesConc_Hg0'] = HgConversion(MDD_2['SpeciesConc_Hg0'])
        Apr_2['SpeciesConc_Hg0'] = HgConversion(Apr_2['SpeciesConc_Hg0'])
        Aqp_2['SpeciesConc_Hg0'] = HgConversion(Aqp_2['SpeciesConc_Hg0'])
        Npun_2['SpeciesConc_Hg0'] = HgConversion(Npun_2['SpeciesConc_Hg0'])
        Spun_2['SpeciesConc_Hg0'] = HgConversion(Spun_2['SpeciesConc_Hg0'])
        
        #extracting receptor
        Base_a = Base['SpeciesConc_Hg0'].sel(lon=coords[1],lat=coords[0],method='nearest').to_dataframe().drop(['lat', 'lon','height'],axis=1).tz_localize(tz='UTC').tz_convert('America/La_Paz').tz_localize(None)
        NoAsgm_a = NoAsgm['SpeciesConc_Hg0'].sel(lon=coords[1],lat=coords[0],method='nearest').to_dataframe().drop(['lat', 'lon','height'],axis=1).tz_localize(tz='UTC').tz_convert('America/La_Paz').tz_localize(None)
        MDD_a = MDD_2['SpeciesConc_Hg0'].sel(lon=coords[1],lat=coords[0],method='nearest').to_dataframe().drop(['lat', 'lon','height'],axis=1).tz_localize(tz='UTC').tz_convert('America/La_Paz').tz_localize(None)
        Apr_a = Apr_2['SpeciesConc_Hg0'].sel(lon=coords[1],lat=coords[0],method='nearest').to_dataframe().drop(['lat', 'lon','height'],axis=1).tz_localize(tz='UTC').tz_convert('America/La_Paz').tz_localize(None)
        Aqp_a = Aqp_2['SpeciesConc_Hg0'].sel(lon=coords[1],lat=coords[0],method='nearest').to_dataframe().drop(['lat', 'lon','height'],axis=1).tz_localize(tz='UTC').tz_convert('America/La_Paz').tz_localize(None)
        Npun_a = Npun_2['SpeciesConc_Hg0'].sel(lon=coords[1],lat=coords[0],method='nearest').to_dataframe().drop(['lat', 'lon','height'],axis=1).tz_localize(tz='UTC').tz_convert('America/La_Paz').tz_localize(None)
        Spun_a = Spun_2['SpeciesConc_Hg0'].sel(lon=coords[1],lat=coords[0],method='nearest').to_dataframe().drop(['lat', 'lon','height'],axis=1).tz_localize(tz='UTC').tz_convert('America/La_Paz').tz_localize(None)

        GC_peru = Base_a['SpeciesConc_Hg0'].to_frame()
        GC_peru = GC_peru.rename(columns={'SpeciesConc_Hg0':'base'})
        GC_peru['noasgm'] = NoAsgm_a['SpeciesConc_Hg0']
        GC_peru['mdd'] = MDD_a['SpeciesConc_Hg0']
        GC_peru['apr'] = Apr_a['SpeciesConc_Hg0']
        GC_peru['aqp'] = Aqp_a['SpeciesConc_Hg0']
        GC_peru['npun'] = Npun_a['SpeciesConc_Hg0']
        GC_peru['spun'] = Spun_a['SpeciesConc_Hg0']
        
        return Base,GC_peru
    
    if sitename=='GunnPoint':
        
        path = '../../Model_data/ATARS/GEOSChem/'
        ts = 'GPtimeseries_'
        
        df = pd.read_csv(path+run+'/'+ts+run+'.csv')
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        df = df.tz_localize(tz='UTC').tz_convert('Etc/GMT+10').tz_localize(None)
        
        if surf == True:
            ts = 'GEOSChem.SpeciesConcGPA.allsurf.nc4'
            ds = xr.open_dataset(path+run+'/'+ts,engine='netcdf4').shift(time=-10) #shifting from UTC to EDT
            return ds,df
        else:
            return df
        
def read_obs(sitename,test=False):
    '''
    Reads observations
    '''    
    obs_path = '../../Measurement_data/'
    
    if sitename=='Harvard':        
        obs = pd.read_csv(obs_path+'hf373-01-30min-gem-flux.csv')
        obs['datetime'] = pd.to_datetime(obs['datetime'])
        obs = obs.set_index('datetime')
        obs = obs[['airGEM','ac.GEMflux.outlier.interp']]        
        return obs
        
    if sitename=='Chacaltaya':
        Ch14 = pd.read_csv(obs_path+'L1_TGM_CHC_2014.csv',delimiter=';',decimal=",",skiprows=35)
        Ch15 = pd.read_csv(obs_path+'L1_TGM_CHC_2015.csv',delimiter=';',decimal=",",skiprows=35)

        Ch_dat = pd.concat([Ch14, Ch15])
        Ch_dat['datetime_utc']=pd.to_datetime(Ch_dat['Date/time (UTC)'])
        Ch_dat = Ch_dat.set_index('datetime_utc')
        Ch_dat = Ch_dat.sort_index()
        Ch_dat = Ch_dat[Ch_dat.index>'2014-11-01']
        Ch_dat = Ch_dat[Ch_dat.index<'2015-06-30']
        Ch_dat = Ch_dat['TGM_valid'].astype(float)
        Ch_dat = Ch_dat.tz_localize(tz='UTC').tz_convert('America/La_Paz').tz_localize(None) #convert to local time
        Ch_dat = Ch_dat.rename_axis('time')
        return Ch_dat
    
    if sitename=='GunnPoint':
        return pd.read_excel(obs_path+'ATARS_2014_2015.xlsx')
    
    if sitename=='GunnPoint_process':
        obs_path = '/home/emroy/GP/AUBB/data/'
        if test == False:
            GP = pd.read_excel(obs_path+'TekxGP_5m_14_17_20221006.xlsx',sheet_name='gp')
            GP = GP.set_index('date')
            GP = GP.sort_index()
            GP_h = GP.resample('1H').mean()
            GP_h = GP_h.tz_localize(tz='UTC').tz_convert('Etc/GMT+10').tz_localize(None) #convert to local time
            return GP_h
        if test == True:
            GP = pd.read_excel(obs_path+'TekxGP_5m_14_17_20221006_condensed.xlsx')
            return GP

def obs_filter(df,var='airGEM',sd=4):
    '''
    Filters observations that are greater than or less than "sd" standard deviations from the mean.
    Default is 4.
    '''
    mean = df[var].mean()
    stdev = df[var].std()
    df = df[df[var]<(df[var].mean()+stdev*sd)]
    df = df[df[var]>(df[var].mean()-stdev*sd)]
    return df
    
def deposition_conv(ds,coords,var='DryDep_Hg0',receptor_only=True):
    '''
    convert deposition from molec cm-2 s-1 to ng m-2 hr-1
    '''
    val = ds.copy()
    val[var] = ds[var].copy()*100*100 #cm2 to m2 
    val[var] = val[var]/(6.022e23)*200.59*1e9 #convert from molec m-2 s-1 to ng m-2 s-1
    if receptor_only == True:
        return (val[var]*60*60).sel(lon=coords[1],lat=coords[0],method='nearest') #convert from ng m-2 s-1 to ng m-2 hr-1
    if receptor_only == False:
        return val #units are ng m-2 s-1

def read_asgm_ems(d_start='2014-11-01',d_end='2015-06-30'):
    eroy_path = '../../Model_data/CHC/GEOSChem/' #'/net/fs03/d0/emroy/Shakes/gc_data/'
    emissions_data = pd.read_csv(eroy_path+'EmissionsData/emissions_GMA_ASGM.csv')

    #calculate emissions for period of interest, assuming constant emission rate
    d_start=pd.to_datetime(d_start)
    d_end = pd.to_datetime(d_end)
    y_frac = (d_end-d_start).days/365
    
    s_puno_hg=emissions_data['s_pun'].values*y_frac
    n_puno_hg=emissions_data['n_pun'].values*y_frac
    mdd_hg = emissions_data['mdd'].values*y_frac
    aqp_hg = emissions_data['aqp'].values*y_frac
    apr_hg = emissions_data['apr'].values*y_frac

    scalars ={'spun':[s_puno_hg,s_puno_hg*2],'npun':[n_puno_hg,n_puno_hg*2],'mdd':[mdd_hg,mdd_hg*2],'aqp':[aqp_hg,aqp_hg*2], 'apr':[apr_hg,apr_hg/2]}
    regions=['spun','npun','mdd','aqp','apr']
    region_names={'spun':'South Puno','npun':'North Puno','mdd':'Madre de Dios','aqp':'Arequipa','apr':'Apurimac'}
    return scalars,regions,region_names

def signal_asgm(base,sens,scalars,region='spun'):
    if region == 'apr':
        #apr order needs to be flipped since sensitivity is half base instead of double
        sig = (base-sens)/(scalars[region][0]-scalars[region][1])
        return sig*scalars[region][1] #+intercept  
    else:    
        sig = (base-sens)/(scalars[region][0]-scalars[region][1])
        intercept = sens - (scalars[region][1]*sig)
        return sig*scalars[region][0] #+intercept    

def signal_asgm_stilt(run,scalar):
    #apr order needs to be flipped since sensitivity is half base instead of double
    sig = run.to_dataframe()/scalar.values
    return sig 
    
def statistic_calc(series,statistic): 
    '''
    calculates specified statistic
    '''
    if statistic=='mean':
        stat = series.mean()
    elif statistic=='median':
        stat = series.median()
    elif statistic=='IQR':
        stat = series.quantile(q=0.75)-series.quantile(q=0.25)
    elif statistic=='75th':
        stat = series.quantile(q=0.875)-series.quantile(q=0.125)
    elif statistic=='90th':
        stat = series.quantile(q=0.95)-series.quantile(q=0.05)
    elif statistic=='95th':
        stat = series.quantile(q=0.975)-series.quantile(q=0.025)
    return stat

def ts_detrend(series,period=90):
    '''
    Returns a detrended dataset
    '''
    return series - series.rolling(period,min_periods=1,center=True).mean()

def bootstrap_4_statistic(ts_nonan,ts_nonan_rolling,stdev,statistic='mean',iteration=10000,detrend_period=90):
    
    '''
    Calculates error in terms of statistic of interest.
    Key inputs are the timeseries, standard deviation from literature-based errors
    '''
    
    #calculate val for timeseries, ignoring uncertainty.
    obs_detrend = ts_nonan-ts_nonan_rolling
    obs_stat = statistic_calc(obs_detrend,statistic=statistic)

    r_stats = np.array([])
    for i in list(range(0,iteration)):
        r = scipy.stats.norm.rvs(loc=obs_detrend, scale=stdev, size=len(obs_detrend))
        r = pd.Series(r)
        r_detrend = ts_detrend(r,period=detrend_period)
        r_stat = statistic_calc(r_detrend,statistic=statistic)
        
        r_stats = np.append(r_stats,r_stat)
        
    return obs_stat,r_stats

def proposal_distribution(theta_m1,proposal_stdev,bounds,dist_type='normal'):
    #return a random number centered around the previous state vector
    
    if dist_type=='normal':
        proposal = norm.rvs(loc=theta_m1,scale=proposal_stdev)
        while np.any(proposal<=bounds[0]) or np.any(proposal>bounds[1]):
            proposal = norm.rvs(loc=theta_m1,scale=proposal_stdev)
    
    if dist_type=='uniform':
        proposal = uniform.rvs(scale=30)
    return proposal

def prob_y_x(model,obs,stdev):
    #return pdf surrounding each observation, providing the likelihood of given model parameterization
    return norm.pdf(model,loc=obs,scale=stdev) #not multiplying by uniform() distribution, irrelevant since always equal to 1

def model_dd(ConstantTerm,DDsig,dep_HF,stat='stat'):
    #calculating the 95th pctile range for a given set of parameters
    conc = ConstantTerm-(dep_HF*DDsig)
    if stat == '95th':
        q975,q025 = np.percentile(conc,[97.5,2.5])
        return q975-q025
    if stat == '90th':
        q95,q5 = np.percentile(conc,[95,5])
        return q95-q5
    if stat == '75th':
        q875,q125 = np.percentile(conc,[87.5,12.5])
        return q875-q125
    if stat == 'IQR':
        q75,q25 = np.percentile(conc,[75,25])
        return q75-q25

def model_asgm(ConstantTerm,sig,theta,stat='stat'):
    
    ems_mdd,ems_apr,ems_aqp,ems_npun,ems_spun = theta
    sig_mdd,sig_apr,sig_aqp,sig_npun,sig_spun = sig
    
    #calculating the 95th pctile range for a given set of parameters
    conc = ConstantTerm+(ems_mdd*sig_mdd)+(ems_apr*sig_apr)+(ems_aqp*sig_aqp)+(ems_npun*sig_npun)+(ems_spun*sig_spun)
    conc = conc.dropna()
    
    if stat == '95th':
        q975,q025 = np.percentile(conc,[97.5,2.5])
        return q975-q025
    if stat == '90th':
        q95,q5 = np.percentile(conc,[95,5])
        return q95-q5
    if stat == '75th':
        q875,q125 = np.percentile(conc,[87.5,12.5])
        return q875-q125
    if stat == 'IQR':
        q75,q25 = np.percentile(conc,[75,25])
        return q75-q25


def model_bb(ConstantTerm,BBsig,bb_GP,stat='stat'):
    #calculating the 95th pctile range for a given set of parameters
    conc = ConstantTerm+(bb_GP*BBsig)

    if stat == '95th':
        q975,q025 = np.percentile(conc,[97.5,2.5])
        return q975-q025
    if stat == '90th':
        q95,q5 = np.percentile(conc,[95,5])
        return q95-q5
    if stat == '75th':
        q875,q125 = np.percentile(conc,[87.5,12.5])
        return q875-q125
    if stat == 'IQR':
        q75,q25 = np.percentile(conc,[75,25])
        return q75-q25


def metropolis_hastings(ConstantTerm,signal,prior,obs_stat,std,proposal_std=10,dist_type='normal',MCMC_iter=10000,bounds=[4,35],sitename='HarvardForest',stat='stat'):
    
    if sitename=='HarvardForest':
        i_mcmc = np.arange(0,MCMC_iter)
        theta = prior
        predict = model_dd(ConstantTerm,signal,theta,stat=stat)
        rhs_bayes = prob_y_x(predict,obs_stat,std)

        theta_list = np.array([theta])
        rhs_bayes_list = np.array([rhs_bayes])

        for i in i_mcmc:
            #need to figure out how to select dep_HF - some sort of random walk? iterative step size? AH. This is the markov chain step
            #defining dep_HF

            theta_new = proposal_distribution(theta,proposal_std,bounds,dist_type=dist_type)    
            predict_new = model_dd(ConstantTerm,signal,theta_new,stat=stat)
            rhs_bayes_new = prob_y_x(predict_new,obs_stat,std)

            r = rhs_bayes_new/rhs_bayes

            if r>=1:
                theta = theta_new
                theta_list = np.append(theta_list,theta)

                rhs_bayes = rhs_bayes_new
                rhs_bayes_list = np.append(rhs_bayes_list,rhs_bayes)

            else:
                u = uniform.rvs() #draw random number between 0 and 1
                if u<r:
                    theta = theta_new
                    theta_list = np.append(theta_list,theta)

                    rhs_bayes = rhs_bayes_new
                    rhs_bayes_list = np.append(rhs_bayes_list,rhs_bayes)
                else:
                    theta_list = np.append(theta_list,theta)
                    rhs_bayes_list = np.append(rhs_bayes_list,rhs_bayes)

        return theta_list,rhs_bayes_list

    if sitename=='Chacaltaya':
        i_mcmc = np.arange(0,MCMC_iter)
        theta = prior
        predict = model_asgm(ConstantTerm,signal,theta,stat=stat)
        rhs_bayes = prob_y_x(predict,obs_stat,std)

        theta_list = np.array(theta)
        rhs_bayes_list = np.array(rhs_bayes)

        for i in i_mcmc:
            #need to figure out how to select dep_HF - some sort of random walk? iterative step size? AH. This is the markov chain step
            #defining dep_HF

            theta_new = proposal_distribution(theta,proposal_std,bounds,dist_type=dist_type)    
            predict_new = model_asgm(ConstantTerm,signal,theta_new,stat=stat)
            rhs_bayes_new = prob_y_x(predict_new,obs_stat,std)

            r = rhs_bayes_new/rhs_bayes

            if r>=1:
                theta = theta_new
                theta_list = np.hstack((theta_list,theta))

                rhs_bayes = rhs_bayes_new
                rhs_bayes_list = np.hstack((rhs_bayes_list,rhs_bayes))

            else:
                u = uniform.rvs() #draw random number between 0 and 1
                if u<r:
                    theta = theta_new
                    theta_list = np.hstack((theta_list,theta))

                    rhs_bayes = rhs_bayes_new
                    rhs_bayes_list = np.hstack((rhs_bayes_list,rhs_bayes))
                else:
                    theta_list = np.hstack((theta_list,theta))
                    rhs_bayes_list = np.hstack((rhs_bayes_list,rhs_bayes))

        return theta_list,rhs_bayes_list

    if sitename=='GunnPoint':
        i_mcmc = np.arange(0,MCMC_iter)
        theta = prior
        predict = model_bb(ConstantTerm,signal,theta,stat=stat)
        rhs_bayes = prob_y_x(predict,obs_stat,std)

        theta_list = np.array([theta])
        rhs_bayes_list = np.array([rhs_bayes])

        for i in i_mcmc:
            #need to figure out how to select dep_HF - some sort of random walk? iterative step size? AH. This is the markov chain step
            #defining dep_HF

            theta_new = proposal_distribution(theta,proposal_std,bounds,dist_type=dist_type)    
            predict_new = model_bb(ConstantTerm,signal,theta_new,stat=stat)
            rhs_bayes_new = prob_y_x(predict_new,obs_stat,std)

            r = rhs_bayes_new/rhs_bayes

            if r>=1:
                theta = theta_new
                theta_list = np.append(theta_list,theta)

                rhs_bayes = rhs_bayes_new
                rhs_bayes_list = np.append(rhs_bayes_list,rhs_bayes)

            else:
                u = uniform.rvs() #draw random number between 0 and 1
                if u<r:
                    theta = theta_new
                    theta_list = np.append(theta_list,theta)

                    rhs_bayes = rhs_bayes_new
                    rhs_bayes_list = np.append(rhs_bayes_list,rhs_bayes)
                else:
                    theta_list = np.append(theta_list,theta)
                    rhs_bayes_list = np.append(rhs_bayes_list,rhs_bayes)

        return theta_list,rhs_bayes_list

    
def inventory_regrid(inventory,path='insert_path_here',loc='Harvard',run='HF_0p25_z250_gfs',year='2014',dummyfile='201907150000_-72.173_42.537_5_foot.nc',t_res='H',specific_path=False):
    '''
    Converts from native inventory resolution to resolution of STILT output.
    '''
    
    if specific_path == False:
        path = path+loc+'/'+run+'/'+year+'/out/footprints/'        
    
    grid = xr.open_dataset(path+dummyfile) #this is the path to the dummy stilt field
    regrid = xe.Regridder(inventory.isel(time=0),grid,"conservative") #regrids compiled inventory to resolution of STILT field
    inv_rg = regrid(inventory,keep_attrs=True) #inv_rg is the regridded flux inventory. Temporal resolution defined by t_res.

    if t_res == 'D':
        inv_rg = inv_rg.resample(time='1D').bfill()
    if t_res == 'H':
        inv_rg = inv_rg.resample(time='1H').bfill()
    
    return inv_rg

def footprint_convert(footprint):
    std_p = 101325 #Pa
    std_T = 273.15 #K
    R = 8.314 #m3 Pa K-1 mol-1
    footprint['foot'] = footprint['foot']*std_p/(R*std_T)
    footprint['foot'].attrs = {'units':'ng m-3 (ng m-2 s-1)-1','standard_name':'foot','long_name':'Footprint','_FillValue': False}
    return footprint


def gridbox_contribution_STILT(inventory,path='insert_path_here',vstr='DryDep_Hg0',loc='Harvard',run='HF_0p25_z250_gfs',year='2014',receptor_filt=False,df=True,export=False,rec_height='5',inventory_name='gc_base',specific_path=False,stilt_hours=24):
    '''
    This function reads in arbitrary STILT simulation and convolves it with surface flux file
    '''
    
    #need to make sure these match stilt file names!
    if loc == 'Harvard':
        coordinates = [-72.173,42.537]
    if loc == 'Chacaltaya':
        coordinates = [-68.12,-16.2] 
    if loc == 'GunnPoint':
        coordinates = [131.038,-12.262] 
        
    if specific_path==False:
        path = path+loc+'/'+run+'/'+year+'/out/'
        flist = os.listdir(path+'footprints/')
    if specific_path==True:
        flist = os.listdir(path)
    
    for f in flist:
        if f.endswith('_'+str(coordinates[0])+'_'+str(coordinates[1])+'_'+rec_height+'_foot.nc'):
            if specific_path==False:
                ds = xr.open_dataset(path+'footprints/'+f)
            if specific_path==True:
                ds = xr.open_dataset(path+f)
            ds = footprint_convert(ds)
            time = pd.to_datetime(f[0:12]) #get the timestamp from the filename

            #inventory = HF_hour
            if len(inventory.time.values)>1:
                ds_grid = ds*inventory.sel(time=ds.time.values)[vstr]
            else:
                ds_grid = ds*inventory.isel(time=0)[vstr]
                
            ds_grid = ds_grid.rename({'foot':vstr})
            contribution = ds_grid.isel(time=slice(-stilt_hours,-1)).sum(dim='time')
            contribution = contribution.expand_dims(dim={"time":[time]}, axis=2)

            if f == flist[0]:
                ds_contribution = contribution
            else:
                ds_contribution = xr.concat([ds_contribution,contribution],dim='time')

        if receptor_filt == True:
            mask = (
                  (ds_contribution.lat==ds_contribution.sel(lat=coordinates[1],method='nearest').lat)
                  & (ds_contribution.lon==ds_contribution.sel(lon=coordinates[0],method='nearest').lon)
                  )
            ds_contribution = xr.where(mask, 0, ds_contribution)

    if df == True:
        df_signal = df_contribution['foot'].sum('lat').sum('lon').to_dataframe()
        df_signal = df_signal.rename(columns={'foot':'ngm-3'})
        if export == True:
            df_signal.to_csv(path+'concentration_signal/'+run+'_'+inventory_name+'.csv')            
        return ds_contribution,df_signal

    else:    
        return ds_contribution

def unit_convert_asgm(ds,ds_area,var='emi_hg_0_asgm',units='kg_days-1',days=365,Mg=False):
    '''
    Script converts to units of ug m-2 yr-1 or kg yr-1 based on variable name
    '''
    if var == 'emi_hg_0_asgm' or 'emi_hg_g' or 'fluxes': #converts from kg m-2 s-1 to kg m-2 yr-1
        da = ds[var]*60*60*24*days 
        if Mg==True:
            da = da/1000
            
    if units=='kg_days-1': #converts from kg m-2 days-1 to kg days-1
        return da*ds_area['cell_area']
    if units=='kgm-2days-1': 
        return da


def asgm_regions(ds,EDGAR_8p1=False,summation=True):
    
    if EDGAR_8p1==False:
        spuno = ds.sel(lat=slice(-17,-15),lon=slice(-71.25+360,-68.75+360))
        npuno = ds.sel(lat=slice(-15,-13),lon=slice(-71.25+360,-68.75+360))
        mdd =   ds.sel(lat=slice(-13,-11),lon=slice(-71.25+360,-68.75+360))
        are =   ds.sel(lat=slice(-17,-15),lon=slice(-73.75+360,-71.25+360))
        apu =   ds.sel(lat=slice(-15,-13),lon=slice(-73.75+360,-71.25+360))    
    if EDGAR_8p1==True:
        spuno = ds.sel(lat=slice(-17,-15),lon=slice(-71.25,-68.75))
        npuno = ds.sel(lat=slice(-15,-13),lon=slice(-71.25,-68.75))
        mdd =   ds.sel(lat=slice(-13,-11),lon=slice(-71.25,-68.75))
        are =   ds.sel(lat=slice(-17,-15),lon=slice(-73.75,-71.25))
        apu =   ds.sel(lat=slice(-15,-13),lon=slice(-73.75,-71.25)) 
    if summation==True:
        return spuno.sum('lat').sum('lon'), npuno.sum('lat').sum('lon'), mdd.sum('lat').sum('lon'), are.sum('lat').sum('lon'), apu.sum('lat').sum('lon')
    if summation==False:
        return spuno, npuno, mdd, are, apu


def GFED4s_emission_dataset(yearlist,monthlist,inpath,outpath,hourly=True):
    '''
    Goal of this function is to preprocess input files to be in hourly (daily for FINN) resolution in terms of dry
    matter burned. The original emission_dataset function was designed to do far more conversions and therefore is
    more convoluted than is needed for this application.
    '''
    
    for y in yearlist:
        for m in monthlist:
            yyyymm = str(y)+str(m).zfill(2)
            mfile = 'GFED4_gen.025x025.'+yyyymm+'.nc'
            dfile = 'GFED4_dailyfrac_gen.025x025.'+yyyymm+'.nc'
            hfile = 'GFED4_3hrfrac_gen.025x025.'+yyyymm+'.nc'

            #reading in relevant files
            ds_m = xr.open_dataset(inpath+str(y)+'/'+mfile).isel(time=0)
            ds_d = xr.open_dataset(inpath+str(y)+'/'+dfile)
            ds_h = xr.open_dataset(inpath+str(y)+'/'+hfile)
            
            for d in list(range(0,len(ds_d.time))):            
                datetime = np.array([])
                
                if hourly == True:
                    for t in list(range(0,len(ds_h.time))):
                        date = np.full(8, np.atleast_1d(ds_d.isel(time=d).time.dt.date.values)[0])
                        time = ds_h.time.dt.time.values
                        dt = pd.Timestamp.combine(date[t],time[t])
                        datetime = np.append(datetime,dt)
                    dtindex = pd.to_datetime(datetime)

                    frac = ds_d.isel(time=d)['GFED_FRACDAY']*ds_h['GFED_FRAC3HR'].to_dataset(name='frac')
                    frac = frac.assign_coords(time=dtindex)
                else:
                    frac = ds_d.isel(time=d)['GFED_FRACDAY'].to_dataset(name='frac')
                    
                if d == 0:
                    ds_frac = frac
                else:
                    ds_frac = xr.concat([ds_frac,frac],dim='time')
            
            ds = ds_frac['frac']*ds_m
            #ds = ds.coarsen(lon=2,lat=2).mean() #convert from 0.25x0.25 to 0.5x0.5 This isnt a great way of doing this. Using xesmf instead in a later step
            if hourly == True:
                ds.to_netcdf(path=outpath+'GFED4s_hourly_'+yyyymm+'.nc4',format='NETCDF4')
            else:
                ds.to_netcdf(path=outpath+'GFED4s_'+yyyymm+'.nc4',format='NETCDF4')
                
def output2CO(ds='GFED4s'):
    if ds == 'GFED4s':
        co_efac = [6.3e-2,9.3e-2,8.8e-2,1.27e-1,2.1e-1,1.02e-1]
        ltype = ['SAVA','DEFO','TEMP','BORF','PEAT','AGRI']
        EF = pd.DataFrame(co_efac, index=ltype, columns=["Emission_Factors_GC"]) #units of g CO g-1 DM
    if ds == 'FINN':
        EF = pd.read_csv('../../Model_data/ATARS/Priors/FINN/FINN_EFratios_CO2.csv',skiprows=[0,1])
        EF = EF[['GenVegCode','Description','CO2/CO2','CO2/CO']]
        
    if ds == 'FINN1.6':
        EF = pd.read_csv('../../Model_data/ATARS/Priors/FINN_EFratios_CO2_NEW11222017.csv')
        EF = EF.rename(columns={'Unnamed: 0':'GenVegCode','Unnamed: 1':'landtype'})
        EF = EF[['GenVegCode','landtype','CO2/CO2','CO2/CO']]
    return EF
