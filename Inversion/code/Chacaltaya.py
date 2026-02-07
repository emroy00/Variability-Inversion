Create_Inputs = True
Run_MCMC = True

#Site locations [lat,lon]
Chacaltaya = [-16.35352,-68.13150]

#Required packages
import numpy as np
import pandas as pd
import xarray as xr
import function as f

import scipy.stats
from scipy.stats import norm
from scipy.stats import uniform


# ------- user inputs -------
d_start =        '2014-11-01' #start date
d_end =          '2015-06-30' #'2015-06-30' #end date (I think this is inclusive, need to confirm)
resample =       '1D'         #averaging interval
obs_name =       'TGM_valid'  #measured concentration column name
stat =           '95th'       #95th: 95th pctile range; 90th: 90th pctile range; 75th: 75th pctile range; IQR: interquartile range
rolling_period = 60           #days
MCMC_iter =      100000
bounds =         [0,70] 
proposal_std =   10
df_path =        '../posteriors/' 
Convolve =       False       # If convolve is true, stilt sensitivities will be convolved with emissions and exported to /home/emroy/GP/ACP/scripts/final/data/STILT/Results
model_out_path = '../../Model_data/CHC/'
stilt_hours =    24

#errors (based on Song et al., 2015)
IC_err = 10 #%
Pr_err = 2 #%

#important for model-measurement errors
x_grid = 2.5
y_grid = 2

SamplingFrequency_agg = False


# --- Read observations, read concentration signals and respective fluxes from std and nested GC runs ---

obs = pd.DataFrame(f.read_obs('Chacaltaya'))
obs = obs.loc[d_start:d_end]
ds,df = f.read_gc('Chacaltaya',Chacaltaya)
scalars,regions,region_names = f.read_asgm_ems(d_start=d_start,d_end=d_end)
df = df.resample('1D').mean()

#merging gc and obs
obs_gc = pd.merge(df,obs.resample('1D').mean(),left_index=True,right_index=True)
obs_gc_roll = obs_gc.rolling(rolling_period,min_periods=1,center=True).mean()
obs_gc_nonan = obs_gc.dropna(subset=[obs_name])
obs_gc_nonan_r = obs_gc_roll.loc[obs_gc_nonan.index]
obs_gc_dt = obs_gc_nonan - obs_gc_nonan_r                          #Detrended observations and model

#converting scalars using number of days considered for this test

spun_sig = f.signal_asgm(obs_gc_dt['base'],obs_gc_dt['spun'],scalars,region='spun')
npun_sig = f.signal_asgm(obs_gc_dt['base'],obs_gc_dt['npun'],scalars,region='npun')
mdd_sig = f.signal_asgm(obs_gc_dt['base'],obs_gc_dt['mdd'],scalars,region='mdd')
aqp_sig = f.signal_asgm(obs_gc_dt['base'],obs_gc_dt['aqp'],scalars,region='aqp')
apr_sig = f.signal_asgm(obs_gc_dt['base'],obs_gc_dt['apr'],scalars,region='apr')
obs_gc_dt['AllRegions'] = spun_sig+npun_sig+mdd_sig+aqp_sig+apr_sig


# --- Read emission datasets for STILT ---

GMA = xr.open_dataset('../../Model_data/CHC/Priors/GMA_emissions_Hg.0.25x0.25.2015.nc')
ED10 = xr.open_dataset('../../Model_data/CHC/Priors/EDGAR_gold_A_2010_Hg.nc')

#gridbox area for respective runs
GMA_area = xr.open_dataset('../../Model_data/CHC/Priors/gma_gridarea.nc')
ED4p1_area = xr.open_dataset('../../Model_data/CHC/Priors/Edgarv4p1_gridarea.nc')

# --- Calculate STILT concentration signals using prior estimates of emissions ---
# Note: This takes a long time. Use "Convolve" switch to instead use previously calculated signals

# --- MERRA2 ---

if Convolve == True:

    #define run and dummy field
    path_stilt = '/net/fs03/d0/emroy/STILT/STILTrundir/Royetal_ACP/'
    loc = 'Chacaltaya'
    run = 'MERRA2_0p5'
    year = '2014_2015_all'

    inventory_gma_merra = f.inventory_regrid(GMA,path=path_stilt,loc=loc,run=run,year=year,dummyfile='201503310600_-68.12_-16.2_2000_foot.nc',t_res='H')*1e12
    inventory_ed10_merra = f.inventory_regrid(ED10,path=path_stilt,loc=loc,run=run,year=year,dummyfile='201503310600_-68.12_-16.2_2000_foot.nc',t_res='H')*1e12

    gma_merra_contrib = f.gridbox_contribution_STILT(inventory_gma_merra,path=path_stilt,vstr='emi_hg_0_asgm',rec_height='2000',loc=loc,run=run,year=year,df=False,receptor_filt=True,stilt_hours=stilt_hours)  #.sum(('lat','lon'))['foot']
    ed10_merra_contrib = f.gridbox_contribution_STILT(inventory_ed10_merra,path=path_stilt,vstr='emi_hg_g',rec_height='2000',loc=loc,run=run,year=year,df=False,receptor_filt=True,stilt_hours=stilt_hours)  #.sum(('lat','lon'))['foot']
    
    #exporting results to model_out_path
    gma_merra_contrib.to_netcdf(model_out_path+'gma_merra.nc4')
    ed10_merra_contrib.to_netcdf(model_out_path+'ed10_merra.nc4')
    
# --- GDAS ---

if Convolve == True:

    #define run and dummy field
    path_stilt = '/net/fs03/d0/emroy/STILT/STILTrundir/Royetal_ACP/'
    loc = 'Chacaltaya'
    run = 'gdas_0p5'
    year = '2014_2015_all'

    inventory_gma_gdas = f.inventory_regrid(GMA,path=path_stilt,loc=loc,run=run,year=year,dummyfile='201503310600_-68.12_-16.2_2000_foot.nc',t_res='H')*1e12
    inventory_ed10_gdas = f.inventory_regrid(ED10,path=path_stilt,loc=loc,run=run,year=year,dummyfile='201503310600_-68.12_-16.2_2000_foot.nc',t_res='H')*1e12

    gma_gdas_contrib = f.gridbox_contribution_STILT(inventory_gma_gdas,path=path_stilt,vstr='emi_hg_0_asgm',rec_height='2000',loc=loc,run=run,year=year,df=False,receptor_filt=False,stilt_hours=stilt_hours)  #.sum(('lat','lon'))['foot']
    ed10_gdas_contrib = f.gridbox_contribution_STILT(inventory_ed10_gdas,path=path_stilt,vstr='emi_hg_g',rec_height='2000',loc=loc,run=run,year=year,df=False,receptor_filt=False,stilt_hours=stilt_hours)  #.sum(('lat','lon'))['foot']

    gma_gdas_contrib.to_netcdf(model_out_path+'gma_gdas.nc4')
    ed10_gdas_contrib.to_netcdf(model_out_path+'ed10_gdas.nc4')

# --- Below, instead reading precalculated STILT concentration signals ---
    
if Convolve == False:
    gma_merra_contrib = xr.open_dataset(model_out_path+'STILT/gma_merra.nc4')
    ed10_merra_contrib = xr.open_dataset(model_out_path+'STILT/ed10_merra.nc4')   
    gma_gdas_contrib = xr.open_dataset(model_out_path+'STILT/gma_gdas.nc4')
    ed10_gdas_contrib = xr.open_dataset(model_out_path+'STILT/ed10_gdas.nc4')


# --- Calculating prior emissions for the number of days considered in obs ---

days = (pd.to_datetime(d_end)-pd.to_datetime(d_start)).days #gma_gdas_contrib.sortby('time').resample(time='1D').mean().time.size

gma_spuno,gma_npuno,gma_mdd,gma_are,gma_apu = f.asgm_regions(f.unit_convert_asgm(GMA,GMA_area,days=days,Mg=True))
ed10_spuno,ed10_npuno,ed10_mdd,ed10_are,ed10_apu = f.asgm_regions(f.unit_convert_asgm(ED10,ED4p1_area,var='emi_hg_g',days=days,Mg=True))

# --- Calculating STILT concentration signal for each respective region ---

gma_merra_spuno,gma_merra_npuno,gma_merra_mdd,gma_merra_are,gma_merra_apu = f.asgm_regions(gma_merra_contrib['emi_hg_0_asgm'].sortby('time'),EDGAR_8p1=True)
gma_gdas_spuno,gma_gdas_npuno,gma_gdas_mdd,gma_gdas_are,gma_gdas_apu = f.asgm_regions(gma_gdas_contrib['emi_hg_0_asgm'].sortby('time'),EDGAR_8p1=True)

ed10_merra_spuno,ed10_merra_npuno,ed10_merra_mdd,ed10_merra_are,ed10_merra_apu = f.asgm_regions(ed10_merra_contrib['emi_hg_g'].sortby('time'),EDGAR_8p1=True)
ed10_gdas_spuno,ed10_gdas_npuno,ed10_gdas_mdd,ed10_gdas_are,ed10_gdas_apu = f.asgm_regions(ed10_gdas_contrib['emi_hg_g'].sortby('time'),EDGAR_8p1=True)

# --- Defining STILT signals ---

spun_sig_merra_gma = f.signal_asgm_stilt(gma_merra_spuno,gma_spuno).resample('1D').mean()['emi_hg_0_asgm']
npun_sig_merra_gma = f.signal_asgm_stilt(gma_merra_npuno,gma_npuno).resample('1D').mean()['emi_hg_0_asgm']
mdd_sig_merra_gma = f.signal_asgm_stilt(gma_merra_mdd,gma_mdd).resample('1D').mean()['emi_hg_0_asgm']
are_sig_merra_gma = f.signal_asgm_stilt(gma_merra_are,gma_are).resample('1D').mean()['emi_hg_0_asgm']
apu_sig_merra_gma = f.signal_asgm_stilt(gma_merra_apu,gma_apu).resample('1D').mean()['emi_hg_0_asgm']

spun_sig_gdas_gma = f.signal_asgm_stilt(gma_gdas_spuno,gma_spuno).resample('1D').mean()['emi_hg_0_asgm']
npun_sig_gdas_gma = f.signal_asgm_stilt(gma_gdas_npuno,gma_npuno).resample('1D').mean()['emi_hg_0_asgm']
mdd_sig_gdas_gma = f.signal_asgm_stilt(gma_gdas_mdd,gma_mdd).resample('1D').mean()['emi_hg_0_asgm']
are_sig_gdas_gma = f.signal_asgm_stilt(gma_gdas_are,gma_are).resample('1D').mean()['emi_hg_0_asgm']
apu_sig_gdas_gma = f.signal_asgm_stilt(gma_gdas_apu,gma_apu).resample('1D').mean()['emi_hg_0_asgm']

spun_sig_merra_ed10 = f.signal_asgm_stilt(ed10_merra_spuno,ed10_spuno).resample('1D').mean()['emi_hg_g']
npun_sig_merra_ed10 = f.signal_asgm_stilt(ed10_merra_npuno,ed10_npuno).resample('1D').mean()['emi_hg_g']
mdd_sig_merra_ed10 = f.signal_asgm_stilt(ed10_merra_mdd,ed10_mdd).resample('1D').mean()['emi_hg_g']
are_sig_merra_ed10 = f.signal_asgm_stilt(ed10_merra_are,ed10_are).resample('1D').mean()['emi_hg_g']
apu_sig_merra_ed10 = f.signal_asgm_stilt(ed10_merra_apu,ed10_apu).resample('1D').mean()['emi_hg_g']

spun_sig_gdas_ed10 = f.signal_asgm_stilt(ed10_gdas_spuno,ed10_spuno).resample('1D').mean()['emi_hg_g']
npun_sig_gdas_ed10 = f.signal_asgm_stilt(ed10_gdas_npuno,ed10_npuno).resample('1D').mean()['emi_hg_g']
mdd_sig_gdas_ed10 = f.signal_asgm_stilt(ed10_gdas_mdd,ed10_mdd).resample('1D').mean()['emi_hg_g']
are_sig_gdas_ed10 = f.signal_asgm_stilt(ed10_gdas_are,ed10_are).resample('1D').mean()['emi_hg_g']
apu_sig_gdas_ed10 = f.signal_asgm_stilt(ed10_gdas_apu,ed10_apu).resample('1D').mean()['emi_hg_g']

# --- Calculating errors ---

###error calculation###

#intercomparison error
std = obs_gc_nonan[obs_name]*(IC_err/100) #calculates sigma (standard deviation)
standard95_IC,random95_IC = f.bootstrap_4_statistic(obs_gc_nonan[obs_name],obs_gc_nonan_r[obs_name],std,statistic=stat)

#precision error
std = obs_gc_nonan[obs_name]*(Pr_err/100) #calculates sigma (standard deviation)
standard95_Pr,random95_Pr = f.bootstrap_4_statistic(obs_gc_nonan[obs_name],obs_gc_nonan_r[obs_name],std,statistic=stat)

#sampling frequency error
dailystd = obs[obs_name].resample('1D').std()
dailycnt = obs[obs_name].resample('1D').count()
SF_err = dailystd/np.sqrt(dailycnt) #updated calculation based on Rigby et al., 2012 and Chen and Prinn (2006)

SF_err_median = SF_err.median() #Using median, as mean would be a biased representation of the data

if SamplingFrequency_agg == True:
    std = obs_gc_nonan[obs_name]*SF_err_median

if SamplingFrequency_agg == False:
    std = (obs_gc_nonan[obs_name]*SF_err).dropna()

standard95_SF,random95_SF = f.bootstrap_4_statistic(obs_gc_nonan[obs_name],obs_gc_nonan_r[obs_name],std,statistic=stat)


#Measurement-model representation error

lon = np.atleast_1d(ds.sel(lon=Chacaltaya[1],lat=Chacaltaya[0],method='nearest').lon.values)[0]
lat = np.atleast_1d(ds.sel(lon=Chacaltaya[1],lat=Chacaltaya[0],method='nearest').lat.values)[0]

ds_t = ds.sel(lat=slice(lat-y_grid,lat+y_grid),lon=slice(lon-x_grid,lon+x_grid))['SpeciesConc_Hg0']
surf_subset_detrend = ds_t - ds_t.rolling(time=rolling_period,center=True,min_periods=1).mean()

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
standard95_MR,random95_MR = f.bootstrap_4_statistic(obs_gc_nonan[obs_name],obs_gc_nonan_r[obs_name],std,statistic=stat)

tot_err = np.sqrt((random95_IC.std()**2)+(random95_Pr.std()**2)+(random95_SF.std()**2)+(random95_MR.std()**2))

errors = [random95_IC.std(),random95_Pr.std(),random95_SF.std(),random95_MR.std(),tot_err]


#observed variability (detrended)
obs_stat = f.statistic_calc(obs_gc_dt[obs_name],stat)


# --- MCMC (GEOS-Chem 2° x 2.5°) ---

#Constant term
ConstantTerm = obs_gc_dt['base']-obs_gc_dt['AllRegions'] #defines signal from all other sources (asgm from other regions, other processes)

prior = [scalars['mdd'][0],scalars['apr'][0],scalars['aqp'][0],scalars['npun'][0],scalars['spun'][0]]
signal = [mdd_sig,apr_sig,aqp_sig,npun_sig,spun_sig]

gc_base,gc_base_1 = f.metropolis_hastings(ConstantTerm,signal,prior,obs_stat,tot_err,proposal_std=proposal_std,dist_type='normal',bounds=bounds,sitename='Chacaltaya',MCMC_iter=MCMC_iter,stat=stat)

#defining prior df
prior_df = pd.DataFrame({'gc':[scalars['mdd'][0][0],scalars['apr'][0][0],scalars['aqp'][0][0],scalars['npun'][0][0],scalars['spun'][0][0]]})

# --- MCMC (STILT-MERRA2-GMA 0.5° x 0.5°) ---

prior = [gma_mdd.values,gma_apu.values,gma_are.values,gma_npuno.values,gma_spuno.values]
signal = [mdd_sig_merra_gma,apu_sig_merra_gma,are_sig_merra_gma,npun_sig_merra_gma,spun_sig_merra_gma]

stilt_merra_gma,stilt_merra_gma_1 = f.metropolis_hastings(ConstantTerm,signal,prior,obs_stat,tot_err,proposal_std=proposal_std,dist_type='normal',bounds=bounds,sitename='Chacaltaya',MCMC_iter=MCMC_iter,stat=stat)

prior_df['gma'] = [gma_mdd.values[0],gma_apu.values[0],gma_are.values[0],gma_npuno.values[0],gma_spuno.values[0]]

# --- MCMC (STILT-GDAS-GMA 0.5° x 0.5°) ---

prior = [gma_mdd.values,gma_apu.values,gma_are.values,gma_npuno.values,gma_spuno.values]
signal = [mdd_sig_gdas_gma,apu_sig_gdas_gma,are_sig_gdas_gma,npun_sig_gdas_gma,spun_sig_gdas_gma]

stilt_gdas_gma,stilt_gdas_gma_1 = f.metropolis_hastings(ConstantTerm,signal,prior,obs_stat,tot_err,proposal_std=proposal_std,dist_type='normal',bounds=bounds,sitename='Chacaltaya',MCMC_iter=MCMC_iter,stat=stat)


# --- MCMC (STILT-MERRA2-EDGAR2010 0.5° x 0.5°) ---

prior = [np.array([ed10_mdd.values.tolist()]),np.array([ed10_apu.values.tolist()]),np.array([ed10_are.values.tolist()]),np.array([ed10_npuno.values.tolist()]),np.array([ed10_spuno.values.tolist()])]
signal = [mdd_sig_merra_ed10.fillna(0),apu_sig_merra_ed10.fillna(0),are_sig_merra_ed10.fillna(0),npun_sig_merra_ed10.fillna(0),spun_sig_merra_ed10.fillna(0)]

stilt_merra_ed10,stilt_merra_ed10_1 = f.metropolis_hastings(ConstantTerm,signal,prior,obs_stat,tot_err,proposal_std=proposal_std,dist_type='normal',bounds=bounds,sitename='Chacaltaya',MCMC_iter=MCMC_iter,stat=stat)

prior_df['ed10'] = [ed10_mdd.values,ed10_apu.values,ed10_are.values,ed10_npuno.values,ed10_spuno.values]

# --- MCMC (STILT-GDAS-EDGAR2010 0.5° x 0.5°) ---

prior = [np.array([ed10_mdd.values.tolist()]),np.array([ed10_apu.values.tolist()]),np.array([ed10_are.values.tolist()]),np.array([ed10_npuno.values.tolist()]),np.array([ed10_spuno.values.tolist()])]
signal = [mdd_sig_gdas_ed10,apu_sig_gdas_ed10,are_sig_gdas_ed10,npun_sig_gdas_ed10,spun_sig_gdas_ed10]

stilt_gdas_ed10,stilt_gdas_ed10_1 = f.metropolis_hastings(ConstantTerm,signal,prior,obs_stat,tot_err,proposal_std=proposal_std,dist_type='normal',bounds=bounds,sitename='Chacaltaya',MCMC_iter=MCMC_iter,stat=stat)


# --- Exporting posterior distributions for each model as dataframes ---

# --- GC Base ---
posterior_gc_base = pd.DataFrame({'mdd':gc_base[0],
                                  'apu':gc_base[1],
                                  'are':gc_base[2],
                                  'npun':gc_base[3],
                                  'spun':gc_base[4]})

posterior_gc_base.to_csv(df_path+'Chacaltaya_gcbase_'+stat+'.csv')

# --- MERRA2 GMA ---
posterior_stilt_merra_gma = pd.DataFrame({'mdd':stilt_merra_gma[0],
                                  'apu':stilt_merra_gma[1],
                                  'are':stilt_merra_gma[2],
                                  'npun':stilt_merra_gma[3],
                                  'spun':stilt_merra_gma[4]})

posterior_stilt_merra_gma.to_csv(df_path+'Chacaltaya_merragma_'+stat+'.csv')

# --- GDAS GMA ---
posterior_stilt_gdas_gma = pd.DataFrame({'mdd':stilt_gdas_gma[0],
                                  'apu':stilt_gdas_gma[1],
                                  'are':stilt_gdas_gma[2],
                                  'npun':stilt_gdas_gma[3],
                                  'spun':stilt_gdas_gma[4]})

posterior_stilt_gdas_gma.to_csv(df_path+'Chacaltaya_gdasgma_'+stat+'.csv')

# --- MERRA2 ED10 ---
posterior_stilt_merra_ed10 = pd.DataFrame({'mdd':stilt_merra_ed10[0],
                                  'apu':stilt_merra_ed10[1],
                                  'are':stilt_merra_ed10[2],
                                  'npun':stilt_merra_ed10[3],
                                  'spun':stilt_merra_ed10[4]})

posterior_stilt_merra_ed10.to_csv(df_path+'Chacaltaya_merraed10_'+stat+'.csv')

# --- GDAS ED10 ---
posterior_stilt_gdas_ed10 = pd.DataFrame({'mdd':stilt_gdas_ed10[0],
                                  'apu':stilt_gdas_ed10[1],
                                  'are':stilt_gdas_ed10[2],
                                  'npun':stilt_gdas_ed10[3],
                                  'spun':stilt_gdas_ed10[4]})

posterior_stilt_gdas_ed10.to_csv(df_path+'Chacaltaya_gdased10_'+stat+'.csv')

regions = ['mdd','apu','are','npun','spun']

prior_df['regions'] = regions
prior_df.to_csv(df_path+'Chacaltaya_priors.csv')
print('prior has now been exported')

