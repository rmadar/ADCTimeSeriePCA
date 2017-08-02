import numpy             as np
import pandas            as pd
import matplotlib        as mpl
import matplotlib.pyplot as plt
from   sklearn           import decomposition
from   timeit            import default_timer


## ====================================================================
##
## Author: Romain Madar (romain.madar@gmail.com)
## Date  : 01/08/17
##
## Example of Principal Conponent Analysis manipulation using sklearn
## This code analyze time serie of ADC counts and try to reduce low
## frequency correlation using PCA. The full dataset of N samples is
## arranged in N0 vector having each Ndim dimension on which the PCA
## will be performed.
##
## ====================================================================



#-----------------
# 0. Plot settings
#-----------------
mpl.rcParams['figure.facecolor'] = 'w'
mpl.rcParams['legend.frameon'  ] = False
mpl.rcParams['legend.fontsize' ] = 'x-large'
mpl.rcParams['legend.loc'      ] = 'best'
mpl.rcParams['xtick.labelsize' ] = 16
mpl.rcParams['ytick.labelsize' ] = 16
mpl.rcParams['axes.titlesize'  ] = 18
mpl.rcParams['axes.labelsize'  ] = 18
mpl.rcParams['lines.linewidth' ] = 2.5
mpl.rcParams['patch.edgecolor' ] = 'none'
mpl.rcParams['patch.linewidth' ] = 2.5



#--------------------------------------
# 1. Shape the time serie into N points
#    of Ndim-dimension vector
#--------------------------------------
Ndim=1000
fTrain=0.5

def getRawData(whiteNoise=False):
    
    df      = pd.read_csv('data.txt', names=['ADC'])  # Using pandas is 10 times faster than an explicit file reading 
    adc_raw = np.array( df['ADC'] )                   # Get array containing 9 millions values
    N_Ndim  = int( adc_raw.shape[0]/Ndim )            # Compute number of N Ndim-multiplet
    adc_raw = adc_raw[:Ndim*N_Ndim]                   # Keep only the first N*Ndim values (since Nrest<Ndim)
    data    = np.reshape(adc_raw, (N_Ndim,Ndim) )     # Reshape to arrange values in a N rows of Ndim columns
    return data
    

# For timing the different steps
t0 = default_timer()


#---------------------------------------------
# 2. Get the data and initizalize the PCA tool
#---------------------------------------------
print '\n--> Data being loaded ...'
data_raw = getRawData()
data_0   = data_raw - np.mean(data_raw)
pca      = decomposition.PCA(n_components=Ndim)
data_dim = data_raw.shape
t1 = default_timer()
print '   * done in {:.2f}s: {} events of {}-dimension multiplet\n'.format( t1-t0 ,data_dim[0] , data_dim[1] )


#---------------------------------------
# 3. Define training and testing samples
#---------------------------------------
print '--> Data sample is splitted into training ({:2.0f}%) and testing ({:2.0f}%) samples ...'.format( fTrain*100, (1-fTrain)*100 )
data_raw_training = data_raw[:int((len(data_raw)+1)*fTrain)]
data_raw_testing  = data_raw[int(len(data_raw)*fTrain+1):]
data_0_training   = data_0  [:int((len(data_0)+1)*fTrain)]
data_0_testing    = data_0  [int(len(data_0)*fTrain+1):]
t2 = default_timer()
print '   * done in {:.2f}s\n'.format( t2-t1 )



#-----------------
# 4. Fit the model
#-----------------
print '--> PCA matrix is being diagonalized using training data ...'
pca.fit(data_0_training)
t3 = default_timer()
print '   * done in {:.2f}s\n'.format( t3-t2 )


#--------------------------------------------------------------
# 5. Use the model, ie work in initial and decorrelated spaces
#--------------------------------------------------------------
data_decor_train = pca.transform(data_0_training)
data_decor       = pca.transform(data_0_testing)
# Inverse transformation:
# data_back = pca.inverse_transform(data_decor)


#-------------------------------------
# 6. Define your variables of interest
#-------------------------------------
mean_initial     = np.mean( data_raw        , axis=1 )
mean_decor       = np.mean( data_decor      , axis=1 ) + np.mean(data_raw)
mean_decor_train = np.mean( data_decor_train, axis=1 ) + np.mean(data_raw)


#-------------------
# 7. Plot everything
#-------------------
plt.figure(figsize=(20,7))

plt.subplot(131)
plt.title('PC$ = \Sigma_{i} \: \\alpha_{i} n_{i}$')
i=range(0,6)
for i in range(0,6):
    plt.plot(pca.components_[i],label='PC{}'.format(i))
plt.ylim(-0.07,0.15)
plt.xlabel('$i^{th}$ sample ($=$ time [$\mu$s])')
plt.ylabel('$\\alpha_i$')
plt.legend()

plt.subplot(132)
plt.title('Explained Variance Ratio')
plt.loglog( pca.explained_variance_ratio_ )
plt.xlabel('$j^{th}$ principal component')
plt.ylabel('$\sigma_{j} / \sigma_{tot}$')

plt.subplot(133)
plt.title('Sample Distribution')
bins = np.linspace(450,670,220)
plt.hist(mean_initial    , bins, color='b' , histtype='step', alpha=0.6, label='Raw Samples'   )
plt.hist(mean_decor_train, bins, color='g' , histtype='step', alpha=0.6, label='PCA [training]')
plt.hist(mean_decor      , bins, color='r' , histtype='step', alpha=0.6, label='PCA [testing]' )
plt.xlabel('Mean of {:2d} samples'.format(Ndim))
plt.ylabel('Entry')
plt.legend()


plt.tight_layout()
plt.savefig('PCA_rResult.png')
plt.show()


