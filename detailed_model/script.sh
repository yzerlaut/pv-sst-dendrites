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
    python clustered_input_stim.py -c Martinotti --test_NMDA --sparsening 2 4 6 8
    python clustered_input_stim.py -c Martinotti --test_NMDA --sparsening 2 4 6 8 --fix_missing
    # PV Cell
    python clustered_input_stim.py -c Basket --test_uniform --sparsening 2 4 6 8
    python clustered_input_stim.py -c Basket --test_uniform --sparsening 2 4 6 8 --fix_missing
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
                            --iBranch 1 --nSpikeSeed 12 
    # Martinotti Cell
    python step_stim.py --test_with_repeats -c Martinotti\
                            --with_NMDA\
                            --with_presynaptic_spikes\
                            --stimFreq 4e-4\
                            --bgFreqInhFactor 1\
                            --iBranch 1 --nSpikeSeed 12 
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
                         --fix_missing_only\
                        --nSpikeSeed 56
    # Martinotti Cell
    python tvRate_sim.py -c Martinotti\
                         --with_NMDA\
                         --with_presynaptic_spikes\
                         --stimFreq 1.65e-4\
                         --bgFreqInhFactor 1\
                         --iBranch 1\
                         --nStochProc 4 --no_Vm\
                         --fix_missing_only\
                         --nSpikeSeed 56
    # Martinotti Cell, no NMDA
    python tvRate_sim.py -c Martinotti\
                         --with_presynaptic_spikes\
                         --stimFreq 1e-3\
                         --bgFreqInhFactor 1\
                         --iBranch 1\
                         --nSpikeSeed 56\
                         --nStochProc 4 --no_Vm\
                         --fix_missing_only\
                         --suffix noNMDA 
fi


##########################################################
##### time-varying rate Stochastic Inputs (Fig. 5) #######
##########################################################
#
if [[ $1 == 'all' || $1 == 'demo-natMovie' ]]
then
    ## Basket Cell
    python natMovie_sim.py --test_with_repeats\
                            -c Basket\
                            --with_presynaptic_spikes\
                            --stimFreq 1.2e-3\
                            --bgFreqInhFactor 1.00\
                            --iBranch 1\
                            --tstop 0.\
                            --dt 0.05\
                            --nSpikeSeed 16
    # Martinotti Cell
    python natMovie_sim.py --test_with_repeats\
                             -c Martinotti\
                             --with_NMDA --with_presynaptic_spikes\
                             --stimFreq 5e-5\
                             --tstop 0.\
                             --dt 0.05\
                             --bgFreqInhFactor 1\
                             --iBranch 1\
                             --nSpikeSeed 16
    # Martinotti Cell, no NMDA
    #python tvRate_sim.py --test_with_repeats\
                         #-c Martinotti\
                         #--with_presynaptic_spikes\
                         #--stimFreq 1e-3\
                         #--bgFreqInhFactor 1\
                         #--iBranch 1\
                         #--nSpikeSeed 56\
                         #--suffix noNMDA 
fi
