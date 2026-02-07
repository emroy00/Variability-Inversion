Create_Inputs = True
Run_MCMC = True

#Site locations [lat,lon]
GunnPoint = [-12.24912,131.04459]

#Required packages
import numpy as np
import pandas as pd
import xarray as xr
import function as f

import scipy.stats
from scipy.stats import norm
from scipy.stats import uniform


#user inputs
preprocess_inputs = True #set to True if raw GEOS-Chem, STILT outputs are available. False uses preprocessed outputs.
precalculated_SF = True

year =           2015
rolling_period = 60
bounds =         [0,30]
MCMC_iter =      100000

d_start =        str(year)+'-06-01'   #start date
d_end =          str(year)+'-10-01'   #end date (I think this is inclusive, need to confirm)

resample =       '1D'
obs_name =       'conc5_avg'
stat =           'IQR'
df_path =        '../posteriors/'
model_out_path = '../../Model_data/ATARS/STILT/'


#errors (based on Song et al., 2015)
IC_err = 10 #%
Pr_err = 2 #%

#important for model-measurement errors
x_grid = 2.5
y_grid = 2

x_n_grid = 0.625
y_n_grid = 0.5

SamplingFrequency_agg = True
GFED_preprocess=False

obs = f.read_obs('GunnPoint',test=True)
obs = obs.set_index('date')

#outliers on September 9th and 10th right before instrument failure. Removing from dataset.
if year == 2015:
    obs = obs[obs.index<'2015-09-09']

#Base res
ds_bb,df_bb = f.read_gc('GunnPoint',GunnPoint,run='run0018',surf=True)
ds_nobb,df_nobb = f.read_gc('GunnPoint',GunnPoint,run='run0019',surf=True)
ds_nont,df_nont = f.read_gc('GunnPoint',GunnPoint,run='run0038',surf=True)

#Nested res
ds_bb_n,df_bb_n = f.read_gc('GunnPoint',GunnPoint,run='run0022',surf=True)
ds_nobb_n,df_nobb_n = f.read_gc('GunnPoint',GunnPoint,run='run0023',surf=True)
ds_nont_n,df_nont_n = f.read_gc('GunnPoint',GunnPoint,run='run0039',surf=True)

if GFED_preprocess==True:
    #creating daily GFED4s inputs from raw GEOS-Chem Data
    #outpath file 
    inpath = '../../Model_data/ATARS/Priors/GFED4s/'
    outpath = '../../Model_data/ATARS/Priors/GFED4s/processed/'

    f.GFED4s_emission_dataset([year],np.arange(0,12)+1,inpath,outpath,hourly=False)
    
#read in GFED, FINN
GFED4s = xr.open_mfdataset('../../Model_data/ATARS/Priors/GFED4s/processed/GFED4s*.nc4').drop_vars('DM_TOTL')
FINN = xr.open_dataset('../../Model_data/ATARS/Priors/FINN/FINN_daily_'+str(year)+'_0.25x0.25.compressed.nc')
#FINN = xr.open_dataset('/net/fs03/d0/emroy/STILT/FINN1p6/FINNv1.6_'+str(year)+'_GEOSChem.daily.nc')

#scaling factors
GFED_ef = f.output2CO('GFED4s')
FINN1p5_ef = f.output2CO('FINN')
#FINN1p6_ef = f.output2CO('FINN1.6')

#converting GFED
ltype = ['SAVA','DEFO','TEMP','BORF','PEAT','AGRI']
for i in ltype:
    GFED4s['CO_'+i] = GFED4s['DM_'+i]*GFED_ef.loc[i]['Emission_Factors_GC']
    GFED4s = GFED4s.drop_vars('DM_'+i)
    #units are in kg CO m-2 s-1, these were lost in the conversion step in GFED4s_emission_dataset()

ltype = [1,2,3,4,5,9]
lname = ['Savanna','WoodySavanna','Tropical','Temperate','Boreal','Crops']
j=0
for i in ltype:
    FINN['CO_'+lname[j]] = FINN['fire_vegtype'+str(i)]/FINN1p5_ef[FINN1p5_ef['GenVegCode']==i]['CO2/CO'].values[0]
    FINN = FINN.drop_vars('fire_vegtype'+str(i))
    j = j+1

FINN = FINN.drop_vars(['lon_bounds','lat_bounds'])

#calculating total emissions from each landtype
GFED4s['CO_TOT'] = GFED4s['CO_SAVA']+GFED4s['CO_DEFO']+GFED4s['CO_TEMP']+GFED4s['CO_BORF']+GFED4s['CO_PEAT']+GFED4s['CO_AGRI']
FINN['CO_TOT'] = FINN['CO_Savanna']+FINN['CO_WoodySavanna']+FINN['CO_Tropical']+FINN['CO_Temperate']+FINN['CO_Boreal']+FINN['CO_Crops']

#area required for calculation of priors
gfed_area = xr.open_dataset('../../Model_data/ATARS/Priors/GFED4s/processed/GFED4s_gridarea.nc')

#need to add preprocess switch like I did for GEOS-Chem

if preprocess_inputs == True:

    # --- GDAS 0.5° x 0.5°, GFED4s ---

    path_stilt = '/net/fs03/d0/emroy/STILT/STILTrundir/Royetal_ACP/'
    loc = 'GunnPoint'
    run = 'gdas_0p5'

    inventory = f.inventory_regrid(GFED4s,path=path_stilt,loc=loc,run=run,year=str(year),dummyfile=str(year)+'06150000_131.038_-12.262_5_foot.nc',t_res='H',specific_path=False)
    contrib = f.gridbox_contribution_STILT(inventory,path=path_stilt,vstr='CO_TOT',rec_height='5',loc=loc,run=run,year=str(year),df=False,receptor_filt=False,specific_path=False)
    df = contrib.sum(('lat','lon')).to_dataframe()*7.17e-7*1e12
    gdas_gfed_df = df.rename(columns={'CO_TOT':'gdas_gfed_ngm-3'}).resample('1H').bfill()
    gdas_gfed_df.to_csv(model_out_path+'ST_gdas_'+str(year)+'_gfed.csv')


    # --- GDAS 0.5° x 0.5°, FINN1.5 ---

    path_stilt = '/net/fs03/d0/emroy/STILT/STILTrundir/Royetal_ACP/'
    loc = 'GunnPoint'
    run = 'gdas_0p5'

    inventory = f.inventory_regrid(FINN,path=path_stilt,loc=loc,run=run,year=str(year),dummyfile=str(year)+'06150000_131.038_-12.262_5_foot.nc',t_res='H',specific_path=False)
    contrib = f.gridbox_contribution_STILT(inventory,path=path_stilt,vstr='CO_TOT',rec_height='5',loc=loc,run=run,year=str(year),df=False,receptor_filt=False,specific_path=False)
    df = contrib.sum(('lat','lon')).to_dataframe()*7.17e-7*1e12
    gdas_finn_df = df.rename(columns={'CO_TOT':'gdas_finn_ngm-3'}).resample('1H').bfill()
    gdas_finn_df.to_csv(model_out_path+'ST_gdas_'+str(year)+'_finn.csv')


    # --- MERRA2 0.5° x 0.5°, GFED4s ---

    path_stilt = '/net/fs03/d0/emroy/STILT/STILTrundir/Royetal_ACP/'
    loc = 'GunnPoint'
    run = 'MERRA2_0p5'

    inventory = f.inventory_regrid(GFED4s,path=path_stilt,loc=loc,run=run,year=str(year),dummyfile=str(year)+'06150000_131.038_-12.262_5_foot.nc',t_res='H',specific_path=False)
    contrib = f.gridbox_contribution_STILT(inventory,path=path_stilt,vstr='CO_TOT',rec_height='5',loc=loc,run=run,year=str(year),df=False,receptor_filt=False,specific_path=False)
    df = contrib.sum(('lat','lon')).to_dataframe()*7.17e-7*1e12
    merra_gfed_df = df.rename(columns={'CO_TOT':'merra_gfed_ngm-3'}).resample('1H').bfill()
    merra_gfed_df.to_csv(model_out_path+'ST_merra_'+str(year)+'_gfed.csv')


    # --- MERRA2 0.5° x 0.5°, FINN1.5 ---

    path_stilt = '/net/fs03/d0/emroy/STILT/STILTrundir/Royetal_ACP/'
    loc = 'GunnPoint'
    run = 'MERRA2_0p5'

    inventory = f.inventory_regrid(FINN,path=path_stilt,loc=loc,run=run,year=str(year),dummyfile=str(year)+'06150000_131.038_-12.262_5_foot.nc',t_res='H',specific_path=False)
    contrib = f.gridbox_contribution_STILT(inventory,path=path_stilt,vstr='CO_TOT',rec_height='5',loc=loc,run=run,year=str(year),df=False,receptor_filt=False,specific_path=False) 
    df = contrib.sum(('lat','lon')).to_dataframe()*7.17e-7*1e12
    merra_finn_df = df.rename(columns={'CO_TOT':'merra_finn_ngm-3'}).resample('1H').bfill()
    merra_finn_df.to_csv(model_out_path+'ST_merra_'+str(year)+'_finn.csv')

if preprocess_inputs == False:
    gdas_gfed_df = pd.read_csv(model_out_path+'ST_gdas_'+str(year)+'_gfed.csv')
    gdas_finn_df = pd.read_csv(model_out_path+'ST_gdas_'+str(year)+'_finn.csv')
    merra_gfed_df = pd.read_csv(model_out_path+'ST_merra_'+str(year)+'_gfed.csv')
    merra_finn_df = pd.read_csv(model_out_path+'ST_merra_'+str(year)+'_finn.csv')
    

# --- merge observations and gc, subset for desired period ---
df_bb = df_bb.rename(columns={'conc_ngm3':'bb_ngm-3'})     #renaming concentration
df_bb['nobb_ngm-3'] = df_nobb['conc_ngm3']                 #adding concentration from bb off run
df_bb['nont_ngm-3'] = df_nont['conc_ngm3']                 #adding concentration from bb off run

df_bb['bb_n_ngm-3'] = df_bb_n['conc_ngm3']                   #adding concentration from bb off run
df_bb['nobb_n_ngm-3'] = df_nobb_n['conc_ngm3']               #adding concentration from bb off run
df_bb['nont_n_ngm-3'] = df_nont_n['conc_ngm3']               #adding concentration from bb off run

df_bb['gdas_gfed_ngm-3'] = gdas_gfed_df['gdas_gfed_ngm-3']
df_bb['gdas_finn_ngm-3'] = gdas_finn_df['gdas_finn_ngm-3'] 
df_bb['merra_gfed_ngm-3'] = merra_gfed_df['merra_gfed_ngm-3']
df_bb['merra_finn_ngm-3'] = merra_finn_df['merra_finn_ngm-3']

df_bb = df_bb[df_bb.index>=d_start]
df_bb = df_bb[df_bb.index<=d_end]
df_bb = df_bb.resample('1D').mean()                      #added to account for daily observations

#merge with obs
df_bb = pd.merge(df_bb,obs['conc5_avg'],left_index=True,right_index=True).dropna()

#resample to specified frequency
df_bb = df_bb.resample(resample).mean()
df_bb_roll = df_bb.rolling(rolling_period,min_periods=1,center=True).mean()

#drop na values and detrend
df_bb_nonan = df_bb.dropna()
df_bb_nonan_r = df_bb_roll.loc[df_bb_nonan.index]
df_bb_dt = df_bb_nonan - df_bb_nonan_r                          #Detrended observations and model


# --- error calculation --- 

#intercomparison error
std = df_bb_nonan[obs_name]*(IC_err/100) #calculates sigma (standard deviation)
standard95_IC,random95_IC = f.bootstrap_4_statistic(df_bb_nonan[obs_name],df_bb_nonan_r[obs_name],std,statistic=stat)

#precision error
std = df_bb_nonan[obs_name]*(Pr_err/100) #calculates sigma (standard deviation)
standard95_Pr,random95_Pr = f.bootstrap_4_statistic(df_bb_nonan[obs_name],df_bb_nonan_r[obs_name],std,statistic=stat)

if precalculated_SF == False:
    #sampling frequency error
    dailystd = obs[obs_name].resample('1D').std()
    dailycnt = obs[obs_name].resample('1D').count()
    SF_err = dailystd/np.sqrt(dailycnt) #updated calculation based on Rigby et al., 2012 and Chen and Prinn (2006)

    SF_err_median = SF_err.median() #Using median, as mean would be a biased representation of the data

    if SamplingFrequency_agg == True:
        std = df_bb_nonan[obs_name]*SF_err_median

    if SamplingFrequency_agg == False:
        std = (df_bb_nonan[obs_name]*SF_err).dropna()

if precalculated_SF == True:
    csv = pd.read_csv('../../Measurement_data/SamplingFrequency.csv').set_index('year')
    std = csv[csv.index==year]['SamplingFrequencyError'].values[0]
    
standard95_SF,random95_SF = f.bootstrap_4_statistic(df_bb_nonan[obs_name],df_bb_nonan_r[obs_name],std,statistic=stat)

#Measurement-model representation error (standard resolution)
lon = np.atleast_1d(ds_bb.sel(lon=GunnPoint[1],lat=GunnPoint[0],method='nearest').lon.values)[0]
lat = np.atleast_1d(ds_bb.sel(lon=GunnPoint[1],lat=GunnPoint[0],method='nearest').lat.values)[0]

ds_bb_t = ds_bb.sel(lat=slice(lat-y_grid,lat+y_grid),lon=slice(lon-x_grid,lon+x_grid))
surf_subset_detrend = ds_bb_t - ds_bb_t.rolling(time=rolling_period,center=True,min_periods=1).mean()

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

MR_err = q_range['SpeciesConcVV_Hg0'].values.std()

std = df_bb_nonan[obs_name]*MR_err
standard95_MR,random95_MR = f.bootstrap_4_statistic(df_bb_nonan[obs_name],df_bb_nonan_r[obs_name],std,statistic=stat)

tot_err = np.sqrt((random95_IC.std()**2)+(random95_Pr.std()**2)+(random95_SF.std()**2)+(random95_MR.std()**2))
errors = [random95_IC.std(),random95_Pr.std(),random95_SF.std(),random95_MR.std(),tot_err]
np.save(df_path+'Errors_s_GP_'+stat+'_'+str(year)+'.npy',errors)

#Measurement-model representation error (nested resolution)
lon = np.atleast_1d(ds_bb_n.sel(lon=GunnPoint[1],lat=GunnPoint[0],method='nearest').lon.values)[0]
lat = np.atleast_1d(ds_bb_n.sel(lon=GunnPoint[1],lat=GunnPoint[0],method='nearest').lat.values)[0]

ds_bb_n_t = ds_bb_n.sel(lat=slice(lat-y_n_grid,lat+y_n_grid),lon=slice(lon-x_n_grid,lon+x_n_grid))
surf_subset_n_detrend = ds_bb_n_t - ds_bb_n_t.rolling(time=rolling_period,center=True,min_periods=1).mean()

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

MR_err_n = q_range['SpeciesConcVV_Hg0'].values.std()

std_n = df_bb_nonan[obs_name]*MR_err_n
standard95_MR_n,random95_MR_n = f.bootstrap_4_statistic(df_bb_nonan[obs_name],df_bb_nonan_r[obs_name],std_n,statistic=stat)

tot_err_n = np.sqrt((random95_IC.std()**2)+(random95_Pr.std()**2)+(random95_SF.std()**2)+(random95_MR_n.std()**2))
errors_n = [random95_IC.std(),random95_Pr.std(),random95_SF.std(),random95_MR_n.std(),tot_err_n]
np.save(df_path+'Errors_n_GP_'+stat+'_'+str(year)+'.npy',errors)


# --- Calculate priors ---

GFED4s_sum = (GFED4s['CO_TOT']*60*60*24*gfed_area['cell_area']/1000*7.17e-7).sel(time=slice(d_start,d_end)).sum('time').sel(lat=slice(-26,-10),lon=slice(129,138)).sum().values*1
FINN_sum = (FINN['CO_TOT']*60*60*24*FINN['AREA']/1000*7.17e-7).sel(time=slice(d_start,d_end)).sum('time').sel(lat=slice(-26,-10),lon=slice(129,138)).sum().values*1


# --- Target ---
obs_stat = f.statistic_calc(df_bb_dt[obs_name],stat)

# --- MCMC (GEOS-Chem 2° x 2.5°) ---

ConstantTerm = df_bb_dt['nont_ngm-3'] #defines signal from all other sources except biomass burning in AU's NT
prior = [GFED4s_sum]
signal = (df_bb_dt['bb_ngm-3']-df_bb_dt['nont_ngm-3'])/prior

gc_base,gc_base_1 = f.metropolis_hastings(ConstantTerm,signal,prior,obs_stat,tot_err,proposal_std=10,dist_type='normal',bounds=bounds,sitename='GunnPoint',MCMC_iter=MCMC_iter,stat=stat)


# --- MCMC (GEOS-Chem 0.5° x 0.625°) ---

ConstantTerm = df_bb_dt['nont_n_ngm-3'] #defines signal from all other sources except biomass burning in AU's NT
prior = [GFED4s_sum]
signal = (df_bb_dt['bb_n_ngm-3']-df_bb_dt['nont_n_ngm-3'])/prior

gc_nest,gc_nest_1 = f.metropolis_hastings(ConstantTerm,signal,prior,obs_stat,tot_err_n,proposal_std=10,dist_type='normal',bounds=bounds,sitename='GunnPoint',MCMC_iter=MCMC_iter,stat=stat)


# --- MCMC (STILT-MERRA2-GFED4s 0.5° x 0.5°) ---

ConstantTerm = df_bb_dt['nont_n_ngm-3'] #defines signal from all other sources except biomass burning in AU's NT
prior = [GFED4s_sum]
signal = (df_bb_dt['merra_gfed_ngm-3'])/prior

merra_gfed,merra_gfed_1 = f.metropolis_hastings(ConstantTerm,signal,prior,obs_stat,tot_err_n,proposal_std=10,dist_type='normal',bounds=bounds,sitename='GunnPoint',MCMC_iter=MCMC_iter,stat=stat)


# --- MCMC (STILT-GDAS-GFED4s 0.5° x 0.5°) ---

ConstantTerm = df_bb_dt['nont_n_ngm-3'] #defines signal from all other sources except biomass burning in AU's NT
prior = [GFED4s_sum]
signal = (df_bb_dt['gdas_gfed_ngm-3'])/prior

gdas_gfed,gdas_gfed_1 = f.metropolis_hastings(ConstantTerm,signal,prior,obs_stat,tot_err_n,proposal_std=10,dist_type='normal',bounds=bounds,sitename='GunnPoint',MCMC_iter=MCMC_iter,stat=stat)


# --- MCMC (STILT-MERRA2-FINN 0.5° x 0.5°) ---

ConstantTerm = df_bb_dt['nont_n_ngm-3'] #defines signal from all other sources except biomass burning in AU's NT
prior = [FINN_sum]
signal = (df_bb_dt['merra_finn_ngm-3'])/prior

merra_finn,merra_finn_1 = f.metropolis_hastings(ConstantTerm,signal,prior,obs_stat,tot_err_n,proposal_std=10,dist_type='normal',bounds=bounds,sitename='GunnPoint',MCMC_iter=MCMC_iter,stat=stat)


# --- MCMC (STILT-GDAS-FINN 0.5° x 0.5°) ---

ConstantTerm = df_bb_dt['nont_n_ngm-3'] #defines signal from all other sources except biomass burning in AU's NT
prior = [FINN_sum]
signal = (df_bb_dt['gdas_finn_ngm-3'])/prior

gdas_finn,gdas_finn_1 = f.metropolis_hastings(ConstantTerm,signal,prior,obs_stat,tot_err_n,proposal_std=10,dist_type='normal',bounds=bounds,sitename='GunnPoint',MCMC_iter=MCMC_iter,stat=stat)


# --- Exporting posterior distributions for each model as dataframes ---
posterior_df = pd.DataFrame({'gc':gc_base,
                             'gc_n':gc_nest,
                             'merra2_gfed':merra_gfed,
                             'gdas_gfed':gdas_gfed,
                             'merra2_finn':merra_finn,
                             'gdas_finn':gdas_finn})

posterior_df.to_csv(df_path+'GunnPoint_posterior_'+str(year)+'_'+stat+'.csv')

#defining prior df
prior_df = pd.DataFrame({'GFED4s':GFED4s_sum,'FINN1p5':FINN_sum},index=[0])
prior_df.to_csv(df_path+'GunnPoint_priors_'+str(year)+'.csv')

