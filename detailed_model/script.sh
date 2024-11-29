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
    #python clustered_input_stim.py -c Martinotti --test_NMDA --sparsening 2 4 6 8
    #python clustered_input_stim.py -c Martinotti --test_NMDA --sparsening 2 4 6 8 --fix_missing
    #python clustered_input_stim.py -c Martinotti --test_NMDA\
                                                 #--sparsening 12\
                                                 #--nReleaseSeed 24\
                                                 #--p_release 0.25\
                                                 #--Nmax_release 2\
                                                 #--suffix Stochastic
    # PV Cell
    #python clustered_input_stim.py -c Basket --test_uniform --sparsening 2 4 6 8
    #python clustered_input_stim.py -c Basket --test_uniform --sparsening 2 4 6 8 --fix_missing
    python clustered_input_stim.py -c Basket --test_uniform\
                                                 --sparsening 8\
                                                 --nReleaseSeed 16\
                                                 --p_release 0.5\
                                                 --Nmax_release 2\
                                                 --suffix Stochastic --fix_missing
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
#####     Natural Movie Input Dynamics (Fig. x)    #######
##########################################################
#
if [[ $1 == 'all' || $1 == 'demo-natMovie' ]]
then
    nSpikeSeed=56
    dt=0.05
    ## Basket Cell
    python natMovie_sim.py --test_with_repeats\
                            -c Basket\
                            --with_presynaptic_spikes\
                            --iBranch 1\
                            --tstop 40000.\
                            --Inh_fraction 0.05\
                            --synapse_subsampling 2\
                            --with_STP\
                            --dt $dt\
                            --nSpikeSeed $nSpikeSeed
    # -----------------------
    # Martinotti Cell
    python natMovie_sim.py --test_with_repeats\
                             -c Martinotti\
                             --with_presynaptic_spikes\
                             --Inh_fraction 0.15\
                             --tstop 40000.\
                             --iBranch 1\
                             --synapse_subsampling 12\
                             --nSpikeSeed $nSpikeSeed\
                             --dt $dt\
                             --with_NMDA\
                             --with_STP
    # -----------------------
    # Martinotti Cell -- NO STP
    python natMovie_sim.py --test_with_repeats\
                             -c Martinotti\
                             --with_presynaptic_spikes\
                             --Inh_fraction 0.15\
                             --tstop 40000.\
                             --iBranch 1\
                             --synapse_subsampling 12\
                             --nSpikeSeed $nSpikeSeed\
                             --dt $dt\
                             --with_NMDA\
                             --suffix noSTP
    # -----------------------
    # Martinotti Cell, no NMDA
    python natMovie_sim.py --test_with_repeats\
                             -c Martinotti\
                             --with_presynaptic_spikes\
                             --Inh_fraction 0.1\
                             --tstop 40000\
                             --iBranch 1\
                             --synapse_subsampling 12\
                             --nSpikeSeed $nSpikeSeed\
                             --dt $dt\
                             --with_STP\
                             --suffix noNMDA
fi

if [[ $1 == 'all' || $1 == 'full-natMovie' ]]
then
    nSpikeSeed=40
    dt=0.1
    tstop=20000
    ## Basket Cell
    """
    python natMovie_sim.py -c Basket\
                            --Inh_fraction 0.05\
                            --synapse_subsampling 2\
                            --with_STP\
                            --no_Vm\
                            --dt $dt\
                            --tstop $tstop\
                            --nSpikeSeed $nSpikeSeed\
                            --suffix Full
    """
    # -----------------------
    # Martinotti Cell
    python natMovie_sim.py -c Martinotti\
                             --Inh_fraction 0.15\
                             --synapse_subsampling 12\
                             --with_NMDA\
                             --with_STP\
                             --no_Vm\
                             --dt $dt\
                             --tstop $tstop\
                             --nSpikeSeed $nSpikeSeed\
                             --suffix Full
    # -----------------------
    # Basket Cell -- NO STP
    # -----------------------
    # Martinotti Cell -- NO STP
    # -----------------------
    # Martinotti Cell, no NMDA
    # -----------------------
    # Martinotti Cell, no NMDA - no STP
    # -----------------------
fi

if [[ $1 == 'all' || $1 == 'input-range-natMovie' ]]
then
    nSpikeSeed=5
    # -----------------------
    ## Basket Cell
    python natMovie_sim.py -c Basket\
                             --tstop 20000\
                             --Inh_fraction 0.02 0.04 0.06 0.08\
                             --synapse_subsampling 1 2 4 8\
                            --nSpikeSeed $nSpikeSeed\
                             --with_STP\
                            --no_Vm\
                            --suffix InputRange
    # -----------------------
    # Martinotti Cell
    python natMovie_sim.py -c Martinotti\
                             --tstop 20000\
                             --Inh_fraction 0.1 0.15 0.2 0.25\
                             --synapse_subsampling 6 8 10 12\
                             --nSpikeSeed $nSpikeSeed\
                             --with_STP\
                             --with_NMDA\
                             --no_Vm\
                             --suffix InputRange
    # -----------------------
    # -----------------------
    ## Basket Cell -- no STP
    python natMovie_sim.py -c Basket\
                             --tstop 20000\
                             --Inh_fraction 0.04 0.06 0.08\
                             --synapse_subsampling 1 2 4\
                            --nSpikeSeed $nSpikeSeed\
                            --no_Vm\
                            --suffix InputRange_noSTP
    # -----------------------
    # Martinotti Cell -- no NMDA
    python natMovie_sim.py -c Martinotti\
                             --tstop 20000\
                             --Inh_fraction 0.05 0.1 0.15\
                             --synapse_subsampling 4 8 12\
                             --nSpikeSeed $nSpikeSeed\
                             --with_STP\
                             --no_Vm\
                             --suffix InputRange_noNMDA
    # Martinotti Cell -- no STP
    python natMovie_sim.py -c Martinotti\
                             --tstop 20000\
                             --Inh_fraction 0.05 0.1 0.15\
                             --synapse_subsampling 4 8 12\
                             --nSpikeSeed $nSpikeSeed\
                             --with_NMDA\
                             --no_Vm\
                             --suffix InputRange_noSTP
    # Martinotti Cell -- no STP -- no NMDA
    python natMovie_sim.py -c Martinotti\
                             --tstop 20000\
                             --Inh_fraction 0.05 0.1 0.15\
                             --synapse_subsampling 2 4 8 12\
                             --nSpikeSeed $nSpikeSeed\
                             --no_Vm\
                             --suffix InputRange_noNMDAnoSTP
fi

