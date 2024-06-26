if [[ $1 == 'all' || $1 == 'test' ]]
then
    echo 'Running the script to regenerate all modelling data'
    echo '     [...]'
fi


###########################################################
########### stimulation on top of background (Fig. 4) ####
##########################################################
# demo sim
if [[ $1 == 'all' || $1 == 'demo-input-output' ]]
then
    python stim_on_background.py -c Basket --nCluster 10 25 40\
                    --bgStimFreq 3e-3 --bgFreqInhFactor 0.75 --nStimRepeat 10\
                    --test_uniform --suffix Demo --with_presynaptic_spikes
    python stim_on_background.py -c Martinotti --nCluster 8 14 20\
                    --bgStimFreq 1e-3 --bgFreqInhFactor 8 --nStimRepeat 10\
                    --test_NMDA --suffix Demo --with_presynaptic_spikes
fi
# full sim
if [[ $1 == 'all' || $1 == 'full-input-output-curve' ]]
then
    # Martinotti Cell
    python stim_on_background.py -c Martinotti\
                             --nCluster 0 2 4 6 8 10 12 14 16 18 20\
                             --bgStimFreq 1e-3\
                             --bgFreqInhFactor 8\
                             --nStimRepeat 100\
                             --test_NMDA --suffix Full --ISI 400
    # Basket Cell
    python stim_on_background.py -c Basket\
                                 --nCluster 0 5 10 15 20 25 30 35 40 45 50\
                                 --bgStimFreq 3e-3\
                                 --bgFreqInhFactor 0.75\
                                 --nStimRepeat 100\
                                 --test_uniform --suffix Full --ISI 400 
fi


#########################################
########### clustered input (Fig. 4) ####
#########################################
if [[ $1 == 'all' || $1 == 'clustered-input' ]]
then
    # SST Model
    python clustered_input_stim.py -c Martinotti --test_NMDA --sparsening 5 7 10 
    python clustered_input_stim.py -c Martinotti --test_NMDA --sparsening 5 7 10 --fix_missing
    # PV Cell
    python clustered_input_stim.py -c Basket --test_uniform --sparsening 5 7 10
    python clustered_input_stim.py -c Basket --test_uniform --sparsening 5 7 10 --fix_missing
fi


################################
######## step rate (Fig. 5) ####
################################
#
if [[ $1 == 'all' || $1 == 'demo-step' ]]
then
    ## Basket Cell
    python step_stim.py --test_with_repeats -c Basket\
                            --with_presynaptic_spikes\
                            --stimFreq 10e-3\
                            --bgFreqInhFactor 1\
                            --iBranch 1 --nSpikeSeed 102
    # Martinotti Cell
    python step_stim.py --test_with_repeats -c Martinotti\
                            --with_NMDA\
                            --with_presynaptic_spikes\
                            --stimFreq 4e-4\
                            --bgFreqInhFactor 1\
                            --iBranch 1 --nSpikeSeed 102
fi

##########################################################
##### time-varying rate Stochastic Inputs (Fig. 5) #######
##########################################################
#
if [[ $1 == 'all' || $1 == 'demo-tvRate' ]]
then
    ## Basket Cell
    python tvRate_sim.py --test_with_repeats\
                        -c Basket\
                        --with_presynaptic_spikes\
                        --stimFreq 6e-3\
                        --bgFreqInhFactor 1\
                        --iBranch 1\
                        --nSpikeSeed 56
    # Martinotti Cell
    python tvRate_sim.py --test_with_repeats\
                         -c Martinotti\
                         --with_NMDA --with_presynaptic_spikes\
                         --stimFreq 1.65e-4\
                         --bgFreqInhFactor 1\
                         --iBranch 1\
                         --nSpikeSeed 56
    # Martinotti Cell, no NMDA
    python tvRate_sim.py --test_with_repeats\
                         -c Martinotti\
                         --with_presynaptic_spikes\
                         --stimFreq 1e-3\
                         --bgFreqInhFactor 1\
                         --iBranch 1\
                         --nSpikeSeed 56\
                         --suffix noNMDA 
fi

if [[ $1 == 'all' || $1 == 'tvRate' ]]
then
    ## Basket Cell
    python tvRate_sim.py -c Basket\
                        --with_presynaptic_spikes\
                        --stimFreq 6e-3\
                        --bgFreqInhFactor 1\
                        --iBranch 1\
                         --nStochProc 4 --no_Vm\
                        --nSpikeSeed 56
    # Martinotti Cell
    python tvRate_sim.py -c Martinotti\
                         --with_NMDA\
                         --with_presynaptic_spikes\
                         --stimFreq 1.65e-4\
                         --bgFreqInhFactor 1\
                         --iBranch 1\
                         --nStochProc 4 --no_Vm\
                         --nSpikeSeed 56
    # Martinotti Cell, no NMDA
    python tvRate_sim.py -c Martinotti\
                         --with_presynaptic_spikes\
                         --stimFreq 1e-3\
                         --bgFreqInhFactor 1\
                         --iBranch 1\
                         --nSpikeSeed 56\
                         --nStochProc 4 --no_Vm\
                         --suffix noNMDA 
fi






##########################################################
########### dendro-somatic propagation ###################
##########################################################
#if [[ $1 == 'all' || $1 == 'dendro-somatic-propag' ]]
#then
    ## Martinotti Cell
    #python propag.py -c Martinotti --nCluster 1 --test_active --suffix Single
    #python propag.py -c Martinotti --nCluster 4 --test_active --suffix Multi
    ### Basket Cell
    #python propag.py -c Basket --nCluster 1 --test_active --suffix Single
    #python propag.py -c Basket --nCluster 4 --test_active --suffix Multi
#fi

##########################################################
########### intensity-timing simulations #################
##########################################################
#if [[ $1 == 'all' || $1 == 'width-demo' ]]
#then
    ## # --- narrow width --- # #
    ## Basket cell
    #python intensity_timing_sim.py --test --with_presynaptic_spikes --freq 1e-2 --width 25 --bgStimFreq 2e-3 --bgFreqInhFactor 1 -c Basket --iBranch 1 --filename ../../data/detailed_model/narrow-width-Basket.npy &
    ## Martinotti cell
    #python intensity_timing_sim.py --test --with_presynaptic_spikes --freq 1e-2 --width 25 --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --with_NMDA --filename ../../data/detailed_model/narrow-width-Martinotti.npy --iBranch 1 & 
    ## # --- large width --- # #
    ## Basket cell
    #python intensity_timing_sim.py --test --with_presynaptic_spikes --freq 1e-2 --width 100 --bgStimFreq 2e-3 --bgFreqInhFactor 1 -c Basket --iBranch 1 --filename ../../data/detailed_model/narrow-width-Basket.npy &
    ## Martinotti cell
    #python intensity_timing_sim.py --test --with_presynaptic_spikes --freq 1e-2 --width 100 --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --with_NMDA --filename ../../data/detailed_model/narrow-width-Martinotti.npy --iBranch 1 & 
#fi

#if [[ $1 == 'all' || $1 == 'intensity-demo' ]]
#then
    ## # --- demo data --- # #
    ## Basket cell
    #for w in 6.25 12.5 25 50 100
    #do
        #python intensity_timing_sim.py --test --with_presynaptic_spikes --freq 1e-2 --width $w --bgStimFreq 2e-3 --bgFreqInhFactor 1 -c Basket --from_uniform --iBranch 1 --filename ../../data/detailed_model/IT-Basket-w${w}ms-f10mHz.npy &
    #done
    ## Martinotti cell
    #for w in 6.25 12.5 25 50 100
    #do
        #python intensity_timing_sim.py --test --with_presynaptic_spikes --freq 1e-2 --width $w --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --with_NMDA --filename ../../data/detailed_model/IT-Martinotti-w${w}ms-f10mHz.npy --iBranch 1 & 
    #done
#fi

#if [[ $1 == 'all' || $1 == 'timing-demo' ]]
#then
    ## # --- demo data --- # #
    ## Martinotti cell
    #for factor in 1 2 4 8 16
    #do
        #w=$(printf %.2f $(echo "6.25 * $factor" | bc -l))
        #f=$(printf %.2f $(echo "10.00 / $factor" | bc -l))
        #python intensity_timing_sim.py --test --with_presynaptic_spikes --freq ${f}e-3 --width $w --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --with_NMDA --filename ../../data/detailed_model/IT-Martinotti-w${w}ms-f${f}mHz.npy --iBranch 1 --nStimRepeat 10 & 
        #python intensity_timing_sim.py --test --with_presynaptic_spikes --freq ${f}e-3 --width $w --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --filename ../../data/detailed_model/IT-Martinotti-NO-NMDA-w${w}ms-f${f}mHz.npy --iBranch 1 --nStimRepeat 10 & 
    #done
#fi

#if [[ $1 == 'all' || $1 == 'intensity-passive' ]]
#then
    #python intensity_timing_sim.py --freq 1e-2 --width 6.25 12.5 25 50 100 --bgStimFreq 2e-3 --bgFreqInhFactor 1 -c Basket --from_uniform --iBranch 1 --nStimRepeat 50 --ISI 800 --passive --suffix PassiveExample & 
    #python intensity_timing_sim.py --freq 1e-2 --width 6.25 12.5 25 50 100 --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --with_NMDA --iBranch 1 --nStimRepeat 50 --ISI 800 --passive --suffix PassiveExample
#fi

#if [[ $1 == 'all' || $1 == 'intensity-active' ]]
#then
    #python intensity_timing_sim.py --freq 1e-2 --width 6.25 12.5 25 50 100 --bgStimFreq 2e-3 --bgFreqInhFactor 1 -c Basket --from_uniform --iBranch 1 --nStimRepeat 50 --ISI 500 --suffix ActiveExample & 
    #python intensity_timing_sim.py --freq 1e-2 --width 6.25 12.5 25 50 100 --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --with_NMDA --iBranch 1 --nStimRepeat 50 --ISI 500 --suffix ActiveExample
#fi

#if [[ $1 == 'all' || $1 == 'window-dep-full' ]]
#then
    ## Basket
    #python intensity_timing_sim.py --freq 1e-2 --width 6.25 12.5 25 50 100 --bgStimFreq 2e-3 --bgFreqInhFactor 1 -c Basket --from_uniform --nStimRepeat 30 --nBranch 6 --suffix Full #--fix_missing_only
    #for i in 1 2 3 4 5 
    #do
        #python intensity_timing_sim.py --freq 1e-2 --width 6.25 12.5 25 50 100 --bgStimFreq 2e-3 --bgFreqInhFactor 1 -c Basket --from_uniform --nStimRepeat 30 --nBranch 6 --suffix Full --fix_missing_only
    #done
    ## Martinotti
    #python intensity_timing_sim.py --freq 1e-2 --width 6.25 12.5 25 50 100 --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --with_NMDA --nStimRepeat 30 --nBranch 6 --suffix Full #--fix_missing_only
    #for i in 1 2 3 4 5 
    #do
        #python intensity_timing_sim.py --freq 1e-2 --width 6.25 12.5 25 50 100 --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --with_NMDA --nStimRepeat 30 --nBranch 6 --suffix Full --fix_missing_only
    #done
#fi

#if [[ $1 == 'all' || $1 == 'broadening-demo' ]]
#then
    #python intensity_timing_sim.py --freq 2e-2 --width 6.25 --broadening 1 2 4 8 16 --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --iBranch 1 --nStimRepeat 50 --ISI 500 --suffix BroadeningExampleNoNMDA
    #python intensity_timing_sim.py --freq 1e-2 --width 6.25 --broadening 1 2 4 8 16 --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --with_NMDA --iBranch 1 --nStimRepeat 50 --ISI 500 --suffix BroadeningExampleWithNMDA
#fi

#if [[ $1 == 'all' || $1 == 'broadening-full' ]]
#then
    ## no nmda
    #python intensity_timing_sim.py --freq 2e-2 --width 6.25 --broadening 1 2 4 8 16 --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --nBranch 6 --nStimRepeat 50 --ISI 500 --suffix BroadeningFullNoNMDA
    ## with nmda
    #python intensity_timing_sim.py --freq 1e-2 --width 6.25 --broadening 1 2 4 8 16 --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --with_NMDA --nBranch 6 --nStimRepeat 50 --ISI 500 --suffix BroadeningFullWithNMDA
#fi


# # --- full data --- # #
# Basket cell
# python intensity_timing_sim.py --bgStimFreq 2e-3 --bgFreqInhFactor 1 -c Basket --nStimRepeat 10 --suffix Dual #--fix_missing_only
# python intensity_timing_sim.py --bgStimFreq 2e-3 --bgFreqInhFactor 1 -c Basket --nStimRepeat 10 --suffix Dual --fix_missing_only
# Martinotti cell
# python intensity_timing_sim.py --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --test_NMDA --nStimRepeat 10 --suffix Dual # --fix_missing_only
# python intensity_timing_sim.py --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --test_NMDA --nStimRepeat 10 --suffix Dual --fix_missing_only

# #########################################################
# ########## clustered input simulations ##################
# #########################################################
# Martinotti Cell
# python clustered_input_stim.py -c Matrinotti --test_uniform --sparsening 4 5 6 7 8 9 10 11 --passive
# Basket Cell
# python clustered_input_stim.py -c Basket --test_uniform --sparsening 3 4 5 6 7 8 9 10 --passive

# #########################################################
# ########## other test simulations  ######################
# #########################################################
# Martinotti Cell
# python stim_on_background.py -c Martinotti --nCluster 40 --bgStimFreq 5e-4 --bgFreqInhFactor 1 --nStimRepeat 100 --test_NMDA --suffix TimeCourse40 --ISI 300
# python stim_on_background.py -c Martinotti --nCluster 45 --bgStimFreq 5e-4 --bgFreqInhFactor 1 --nStimRepeat 100 --test_NMDA --suffix TimeCourse45 --ISI 300
# python stim_on_background.py -c Martinotti --nCluster 50 --bgStimFreq 5e-4 --bgFreqInhFactor 1 --nStimRepeat 100 --test_NMDA --suffix TimeCourse50 --ISI 300
# Basket Cell
# python stim_on_background.py -c Basket --nCluster 40 --bgStimFreq 2e-3 --bgFreqInhFactor 1 --nStimRepeat 100 --test_uniform --suffix TimeCourse40 --ISI 400 
# python stim_on_background.py -c Basket --nCluster 45 --bgStimFreq 2e-3 --bgFreqInhFactor 1 --nStimRepeat 100 --test_uniform --suffix TimeCourse45 --ISI 400 
# python stim_on_background.py -c Basket --nCluster 50 --bgStimFreq 2e-3 --bgFreqInhFactor 1 --nStimRepeat 100 --test_uniform --suffix TimeCourse50 --ISI 400 
# python clustered_input_stim.py -c Basket --test_uniform --sparsening 3 4 5 6 7 8 9 10
