Create_Inputs = True
Run_MCMC = True

#Site locations [lat,lon]
HarvardForest = [42.53415, -72.17147]

#Required packages
import numpy as np
import pandas as pd
import xarray as xr
import function as f

import scipy.stats
from scipy.stats import norm
from scipy.stats import uniform


# ------- user inputs -------
preprocess_inputs = True #set to True if raw GEOS-Chem, STILT outputs are available. False uses preprocessed outputs.

year = '2019'
d_start = year+'-06-01'          #start date
d_end = year+'-09-30'            #end date (inclusive)
resample = '1D'                 #averaging interval
obs_name = 'airGEM'             #measured concentration column name
stat = 'IQR'                   #95th percentile range
rolling_period = 60             #days
MCMC_iter = 100000
bounds = [0,35]                 #range of fluxes over which MCMC is allowed to explore
proposal_std = 10               #stdev of Metropolis-Hastings algorithm (can be adjusted to improve efficiency)
receptor_filt = False
model_out_path = '../../Model_data/HarvardForest/'
df_path = '../posteriors/'

#errors (based on Song et al., 2015)
IC_err = 10 #%
Pr_err = 2 #%

#important for model-measurement errors
x_grid = 2.5
y_grid = 2

x_n_grid = 0.625
y_n_grid = 0.5

SamplingFrequency_agg = False

# --- prior estimates (litterfall) ---

#Using obrist et al. (2021) litterfall value
Total_litterfall = 8.1


# --- Read observations, read concentration signals and respective fluxes from std and nested GC runs ---

obs = f.read_obs('Harvard')

if preprocess_inputs == True:

    ds_dd,df_dd = f.read_gc('Harvard',HarvardForest,run='run0045')                             #2x2.5 run. Std.
    ds_nodd,df_nodd = f.read_gc('Harvard',HarvardForest,run='run0046')                         #2x2.5 run. No dry deposition in NE

    ds_dd_n,df_dd_n = f.read_gc('Harvard',HarvardForest,run='run0051',compiled=True)           #0.5x0.625 run. Std.
    ds_nodd_n,df_nodd_n = f.read_gc('Harvard',HarvardForest,run='run0051_nodd',compiled=True)  #0.5x0.625 run. No dry deposition in NE.

    #exporting to netcdf, csv files
    ds_dd.to_netcdf(model_out_path+'GC_2x2p5_dd.nc')
    ds_nodd.to_netcdf(model_out_path+'GC_2x2p5_nodd.nc')
    ds_dd_n.to_netcdf(model_out_path+'GC_0p5x0p625_dd.nc')
    ds_nodd_n.to_netcdf(model_out_path+'GC_0p5x0p625_nodd.nc')
    
    df_dd.to_csv(model_out_path+'GC_2x2p5_dd.csv')
    df_nodd.to_csv(model_out_path+'GC_2x2p5_nodd.csv')
    df_dd_n.to_csv(model_out_path+'GC_0p5x0p625_dd.csv')
    df_nodd_n.to_csv(model_out_path+'GC_0p5x0p625_nodd.csv')

    #calculating deposition fields
    dd = f.deposition_conv(ds_dd,HarvardForest)                                                #dd flux from base run, ng m-2 hr-1
    nodd = f.deposition_conv(ds_nodd,HarvardForest)                                            #dd flux from nodd run, ng m-2 hr-1

    dd_n = f.deposition_conv(ds_dd_n,HarvardForest)                                            #dd flux from nested base run, ng m-2 hr-1
    nodd_n = f.deposition_conv(ds_nodd_n,HarvardForest)                                        #dd flux from nested nodd run, ng m-2 hr-1

    
    # --- Read and process STILT runs ---

    #1. gridbox_contribution will calculate the contribution of each gridbox at each timestamp to the signal. 
    #2. sum over lat and lon to get a timeseries
    #3. add this to baseline 
    #4. estimate model-measurement error separately.

    # --------------- GFS (0.25 x 0.25) ----------------------------

    path_stilt = '/net/fs03/d0/emroy/STILT/STILTrundir/Royetal_ACP/'
    loc = 'Harvard'
    run = 'GFS_0p25'
    year = '2019'

    ds_dd_ngm2s = f.deposition_conv(ds_dd_n,HarvardForest,receptor_only=False)
    inventory_5m = f.inventory_regrid(ds_dd_ngm2s,path=path_stilt,loc=loc,run=run,year=year,dummyfile='201907150000_-72.173_42.537_5_foot.nc',t_res='H')
    gfs_5m_contrib = f.gridbox_contribution_STILT(inventory_5m,path=path_stilt,vstr='DryDep_Hg0',loc=loc,run=run,year=year,df=False,receptor_filt=receptor_filt)
    gfs_5m_df = gfs_5m_contrib.sum(('lat','lon')).to_dataframe()
    gfs_5m_df = gfs_5m_df.rename(columns={'DryDep_Hg0':'gfs_5m_ngm-3'})
    gfs_5m_df.to_csv(model_out_path+'ST_gfs.csv')


    # ------------- MERRA2 (0.5 x 0.5) -------------------------------

    #path_stilt = '/net/fs03/d0/emroy/STILT/STILTrundir/HF_0p5_z250_MERRA2/out/footprints/'
    path_stilt = '/net/fs03/d0/emroy/STILT/STILTrundir/Royetal_ACP/'
    loc = 'Harvard'
    run = 'MERRA2_0p5'
    year = '2019'

    inventory = f.inventory_regrid(ds_dd_ngm2s,path=path_stilt,loc=loc,run=run,year=year,dummyfile='201907150000_-72.173_42.537_30_foot.nc',t_res='H',specific_path=False)
    contrib = f.gridbox_contribution_STILT(inventory,path=path_stilt,vstr='DryDep_Hg0',rec_height='30',loc=loc,run=run,year=year,df=False,receptor_filt=receptor_filt,specific_path=False)
    df = contrib.sum(('lat','lon')).to_dataframe()
    merra_df = df.rename(columns={'DryDep_Hg0':'merra_ngm-3'})
    merra_df.to_csv(model_out_path+'ST_merra.csv')

    # -------------- NAM (~12km x 12km) --------------------------------

    path_stilt = '/net/fs03/d0/emroy/STILT/STILTrundir/Royetal_ACP/'
    loc = 'Harvard'
    run = 'nam12'
    year = '2019'

    inventory = f.inventory_regrid(ds_dd_ngm2s,path=path_stilt,loc=loc,run=run,year=year,dummyfile='201907150000_-72.173_42.537_30_foot.nc',t_res='H')
    contrib = f.gridbox_contribution_STILT(inventory,path=path_stilt,vstr='DryDep_Hg0',rec_height='30',loc=loc,run=run,year=year,df=False,receptor_filt=receptor_filt) 
    df = contrib.sum(('lat','lon')).to_dataframe()
    nam_df = df.rename(columns={'DryDep_Hg0':'nam_ngm-3'})
    nam_df.to_csv(model_out_path+'ST_nam.csv')

if preprocess_inputs == False:
    ds_dd = xr.open_dataset(model_out_path+'GC_2x2p5_dd.nc')
    ds_nodd = xr.open_dataset(model_out_path+'GC_2x2p5_nodd.nc')
    ds_dd_n = xr.open_dataset(model_out_path+'GC_0p5x0p625_dd.nc')
    ds_nodd_n = xr.open_dataset(model_out_path+'GC_0p5x0p625_nodd.nc')
    
    df_dd = pd.read_csv(model_out_path+'GC_2x2p5_dd.csv')
    df_nodd = pd.read_csv(model_out_path+'GC_2x2p5_nodd.csv')
    df_dd_n = pd.read_csv(model_out_path+'GC_0p5x0p625_dd.csv')
    df_nodd_n = pd.read_csv(model_out_path+'GC_0p5x0p625_nodd.csv')
    
    #calculating deposition fields
    dd = f.deposition_conv(ds_dd,HarvardForest)                                                #dd flux from base run, ng m-2 hr-1
    nodd = f.deposition_conv(ds_nodd,HarvardForest)                                            #dd flux from nodd run, ng m-2 hr-1

    dd_n = f.deposition_conv(ds_dd_n,HarvardForest)                                            #dd flux from nested base run, ng m-2 hr-1
    nodd_n = f.deposition_conv(ds_nodd_n,HarvardForest)                                        #dd flux from nested nodd run, ng m-2 hr-1
    
    #now reading STILT
    gfs_5m_df = pd.read_csv(model_out_path+'ST_gfs.csv')
    merra_df = pd.read_csv(model_out_path+'ST_merra.csv')
    nam_df = pd.read_csv(model_out_path+'ST_nam.csv')


# -------- Combine observations and model arrays into single dataframe ----------

obs_filt = obs.resample('1H').mean() #f.obs_filter(obs,var='airGEM',sd=4).resample('1H').mean()

#merge observations and gc, subset for desired period
df_dd = df_dd.rename(columns={'SpeciesConcVV_Hg0':'dd_ngm-3'})     #renaming concentration
df_dd['nodd_ngm-3'] = df_nodd['SpeciesConcVV_Hg0']                 #adding concentration from dd off run
df_dd['dd_n_ngm-3'] = df_dd_n['SpeciesConcVV_Hg0']                 #adding concentration from nested run
df_dd['nodd_n_ngm-3'] = df_nodd_n['SpeciesConcVV_Hg0']             #adding concentration from nested dd off run

df_dd['dd_ngm-2hr-1'] = dd                                         #adding dry deposition in standard simulation
df_dd['dd_n_ngm-2hr-1'] = dd_n                                     #adding dry deposition from nested simulation
df_dd['nodd_ngm-2hr-1'] = nodd                                     #adding dry deposition in dd off run (this is zero)

#merging with stilt and defining stilt concentration
df_dd = pd.merge(df_dd,gfs_5m_df['gfs_5m_ngm-3'],left_index=True,right_index=True,how='outer')
df_dd = pd.merge(df_dd,merra_df['merra_ngm-3'],left_index=True,right_index=True,how='outer')
df_dd = pd.merge(df_dd,nam_df['nam_ngm-3'],left_index=True,right_index=True,how='outer')

obs_gc = pd.merge(obs_filt,df_dd,left_index=True,right_index=True) #merging obs and df, only keeping timesteps where both are available
obs_gc = obs_gc.resample(resample).mean()
obs_gc_roll = obs_gc.rolling(rolling_period,min_periods=1,center=True).mean()

#subsampling dataframe to cover active deposition season
obs_gc = obs_gc.loc[d_start:d_end]
obs_gc = obs_gc.resample(resample).mean()

obs_gc_nonan = obs_gc.dropna(subset=['airGEM'])
obs_gc_nonan_r = obs_gc_roll.loc[obs_gc_nonan.index]
obs_gc_dt = obs_gc_nonan - obs_gc_nonan_r                          #Detrended observations and model

#make sure surface data cover same timerange, resample
ds_dd_merged = ds_dd['SpeciesConcVV_Hg0'].resample(time=resample).mean().sel(time=obs_gc.index)
ds_nodd_merged = ds_nodd['SpeciesConcVV_Hg0'].resample(time=resample).mean().sel(time=obs_gc.index)

ds_dd_n_merged = ds_dd_n['SpeciesConcVV_Hg0'].resample(time=resample).mean().sel(time=obs_gc.index)
ds_nodd_n_merged = ds_nodd_n['SpeciesConcVV_Hg0'].resample(time=resample).mean().sel(time=obs_gc.index)


#defining priors
prior_dd = (obs_gc['dd_ngm-2hr-1']*24).sum()/1000 #converting to cumulative sum in units of ug m-2
prior_n_dd = (obs_gc['dd_n_ngm-2hr-1']*24).sum()/1000 #converting to cumulative sum in units of ug m-2
meas_dd = (obs_gc['ac.GEMflux.outlier.interp']*24).sum()/1000*-1 #negative factor to ensure consistent directionality

print('GEOS-Chem (2x2.5) Prior is '+str(round(prior_dd,3))+' ug m$^{-2}$')
print('GEOS-Chem (0.5x0.625) Prior is '+str(round(prior_n_dd,3))+' ug m$^{-2}$')
print('Observed deposition is '+str(round(meas_dd,3))+' ug m$^{-2}$')
print('Litterfall deposition (Obrist et al. 2021) is '+str(round(Total_litterfall,3))+' ug m$^{-2}$')


# -------- Calculating errors -----------

std = obs_gc_nonan[obs_name]*(IC_err/100) #calculates sigma (standard deviation)
standard_IC,random_IC = f.bootstrap_4_statistic(obs_gc_nonan[obs_name],obs_gc_nonan_r[obs_name],std,statistic=stat)

#precision error
std = obs_gc_nonan[obs_name]*(Pr_err/100) #calculates sigma (standard deviation)
standard_Pr,random_Pr = f.bootstrap_4_statistic(obs_gc_nonan[obs_name],obs_gc_nonan_r[obs_name],std,statistic=stat)

#sampling frequency error
dailystd = obs[obs_name].resample('1D').std()
dailycnt = obs[obs_name].resample('1D').count()
SF_err = dailystd/np.sqrt(dailycnt) #updated calculation based on Rigby et al., 2012 and Chen and Prinn (2006)

SF_err_median = SF_err.median() #Using median, as mean would be a biased representation of the data

if SamplingFrequency_agg == True:
    std = obs_gc_nonan[obs_name]*SF_err_median

if SamplingFrequency_agg == False:
    std = (obs_gc_nonan[obs_name]*SF_err).dropna()

standard_SF,random_SF = f.bootstrap_4_statistic(obs_gc_nonan[obs_name],obs_gc_nonan_r[obs_name],std,statistic=stat)

#Measurement-model representation error (standard resolution)

lon = np.atleast_1d(ds_dd_merged.sel(lon=HarvardForest[1],lat=HarvardForest[0],method='nearest').lon.values)[0]
lat = np.atleast_1d(ds_dd_merged.sel(lon=HarvardForest[1],lat=HarvardForest[0],method='nearest').lat.values)[0]

ds_dd_merged_t = ds_dd_merged.sel(lat=slice(lat-y_grid,lat+y_grid),lon=slice(lon-x_grid,lon+x_grid))
surf_subset_detrend = ds_dd_merged_t - ds_dd_merged_t.rolling(time=rolling_period,center=True,min_periods=1).mean()

if stat == '95th':
    q025 = surf_subset_detrend.chunk({"time": -1}).quantile(q=.025,dim='time')
    q975 = surf_subset_detrend.chunk({"time": -1}).quantile(q=.975,dim='time')

    q_range = q975-q025
    
if stat == '90th':
    q05 = surf_subset_detrend.chunk({"time": -1}).quantile(q=.05,dim='time')
    q95 = surf_subset_detrend.chunk({"time": -1}).quantile(q=.95,dim='time')

    q_range = q95-q05

if stat == '75th':
    q125 = surf_subset_detrend.chunk({"time": -1}).quantile(q=.125,dim='time')
    q875 = surf_subset_detrend.chunk({"time": -1}).quantile(q=.875,dim='time')

    q_range = q875-q125    

if stat == 'IQR':
    q25 = surf_subset_detrend.chunk({"time": -1}).quantile(q=.25,dim='time')
    q75 = surf_subset_detrend.chunk({"time": -1}).quantile(q=.75,dim='time')

    q_range = q75-q25    

MR_err = q_range.values.std()

std = obs_gc_nonan[obs_name]*MR_err
standard_MR,random_MR = f.bootstrap_4_statistic(obs_gc_nonan[obs_name],obs_gc_nonan_r[obs_name],std,statistic=stat)

tot_err = np.sqrt((random_IC.std()**2)+(random_Pr.std()**2)+(random_SF.std()**2)+(random_MR.std()**2))
errors = [random_IC.std(),random_Pr.std(),random_SF.std(),random_MR.std(),tot_err]
np.save(df_path+'Errors_s_HF_'+stat+'_'+year+'.npy',errors)

#Measurement-model representation error (nested resolution)
lon = np.atleast_1d(ds_dd_n_merged.sel(lon=HarvardForest[1],lat=HarvardForest[0],method='nearest').lon.values)[0]
lat = np.atleast_1d(ds_dd_n_merged.sel(lon=HarvardForest[1],lat=HarvardForest[0],method='nearest').lat.values)[0]

ds_dd_n_merged_t = ds_dd_n_merged.sel(lat=slice(lat-y_n_grid,lat+y_n_grid),lon=slice(lon-x_n_grid,lon+x_n_grid))
surf_subset_n_detrend = ds_dd_n_merged_t - ds_dd_n_merged_t.rolling(time=rolling_period,center=True,min_periods=1).mean()

if stat == '95th':
    q025 = surf_subset_n_detrend.chunk({"time": -1}).quantile(q=.025,dim='time')
    q975 = surf_subset_n_detrend.chunk({"time": -1}).quantile(q=.975,dim='time')

    q_range = q975-q025

if stat == '90th':
    q05 = surf_subset_n_detrend.chunk({"time": -1}).quantile(q=.05,dim='time')
    q95 = surf_subset_n_detrend.chunk({"time": -1}).quantile(q=.95,dim='time')

    q_range = q95-q05

if stat == '75th':
    q125 = surf_subset_n_detrend.chunk({"time": -1}).quantile(q=.125,dim='time')
    q875 = surf_subset_n_detrend.chunk({"time": -1}).quantile(q=.875,dim='time')

    q_range = q875-q125

if stat == 'IQR':
    q25 = surf_subset_n_detrend.chunk({"time": -1}).quantile(q=.25,dim='time')
    q75 = surf_subset_n_detrend.chunk({"time": -1}).quantile(q=.75,dim='time')

    q_range = q75-q25

MR_err_n = q_range.values.std()

std_n = obs_gc_nonan[obs_name]*MR_err_n
standard_MR_n,random_MR_n = f.bootstrap_4_statistic(obs_gc_nonan[obs_name],obs_gc_nonan_r[obs_name],std_n,statistic=stat)

tot_err_n = np.sqrt((random_IC.std()**2)+(random_Pr.std()**2)+(random_SF.std()**2)+(random_MR_n.std()**2))
errors_n = [random_IC.std(),random_Pr.std(),random_SF.std(),random_MR_n.std(),tot_err_n]
np.save(df_path+'Errors_n_HF_'+stat+'_'+year+'.npy',errors_n)



# ------- MCMC (GEOS-Chem 2° x 2.5°) ---------

signal = (obs_gc_dt['nodd_ngm-3']-obs_gc_dt['dd_ngm-3'])/(prior_dd)
ConstantTerm = obs_gc_dt['nodd_ngm-3']
obs_stat = f.statistic_calc(obs_gc_dt['airGEM'],stat)

gc_base,test1 = f.metropolis_hastings(ConstantTerm,signal,prior_dd,obs_stat,tot_err,proposal_std=proposal_std,dist_type='normal',bounds=bounds,MCMC_iter=MCMC_iter,stat=stat)

# ------- MCMC (GEOS-Chem 0.5° x 0.625°) ---------
signal = (obs_gc_dt['nodd_n_ngm-3']-obs_gc_dt['dd_n_ngm-3'])/(prior_n_dd)
ConstantTerm = obs_gc_dt['nodd_n_ngm-3']
obs_stat = f.statistic_calc(obs_gc_dt['airGEM'],stat)

gc_base_n,test1 = f.metropolis_hastings(ConstantTerm,signal,prior_n_dd,obs_stat,tot_err_n,proposal_std=proposal_std,dist_type='normal',bounds=bounds,MCMC_iter=MCMC_iter,stat=stat)


#first, handling GFS, which only becomes available on 6/13/2019
obs_gc_gfs = obs_gc_dt.dropna()

signal_5m = (obs_gc_gfs['nodd_n_ngm-3']-obs_gc_gfs['gfs_5m_ngm-3'])/(prior_n_dd)
ConstantTerm = obs_gc_gfs['nodd_n_ngm-3']
obs_stat = f.statistic_calc(obs_gc_gfs['airGEM'],stat)

stilt_5m,test = f.metropolis_hastings(ConstantTerm,signal_5m,prior_n_dd,obs_stat,tot_err,proposal_std=proposal_std,dist_type='normal',bounds=bounds,MCMC_iter=MCMC_iter,stat=stat)

#then, handling other inventories that represent more complete coverage.
signal_merra = (obs_gc_dt['nodd_n_ngm-3']-obs_gc_dt['merra_ngm-3'])/(prior_n_dd)
signal_nam = (obs_gc_dt['nodd_n_ngm-3']-obs_gc_dt['nam_ngm-3'])/(prior_n_dd)
ConstantTerm = obs_gc_dt['nodd_n_ngm-3']
obs_stat = f.statistic_calc(obs_gc_dt['airGEM'],stat)

stilt_merra,test = f.metropolis_hastings(ConstantTerm,signal_merra,prior_n_dd,obs_stat,tot_err_n,proposal_std=proposal_std,dist_type='normal',bounds=bounds,MCMC_iter=MCMC_iter,stat=stat)
stilt_nam,test = f.metropolis_hastings(ConstantTerm,signal_nam,prior_n_dd,obs_stat,tot_err_n,proposal_std=proposal_std,dist_type='normal',bounds=bounds,MCMC_iter=MCMC_iter,stat=stat)

# ------ Compile a dataframe for each posterior distribution of fluxes ----------
posterior_df = pd.DataFrame({'gc':gc_base,'gc_n':gc_base_n,'gfs_5m':stilt_5m,'merra_250m':stilt_merra,'nam_250m':stilt_nam})

# ------ Export posterior distributions to a csv file that can be read by plotting script -----------
posterior_df.to_csv(df_path+'HarvardForest_posterior_'+stat+'.csv')
