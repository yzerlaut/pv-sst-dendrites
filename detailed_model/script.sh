
START=$(date +%s)

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
if [[ $1 == 'all' || $1 == 'demo-step-1' ]]
then
    ### ----- SIMULATIONS WITHOUT STP ----- ###
    nSeed=32
    widths=(50 50 200)
    ampFs=(4 2 2)
    for i in 1 2 3
    do
        ## Basket Cell
        python step_stim.py --test_with_repeats -c Basket\
                                --with_presynaptic_spikes\
                                --stimFreq 3\
                                --Inh_fraction 0.15 --synapse_subsampling 2\
                                --stepAmpFactor ${ampFs[$i-1]}\
                                --stepWidth ${widths[$i-1]}\
                                --iBranch 1\
                                --nSpikeSeed $nSeed\
                                --suffix noSTP-Step$i
        : '
        # Martinotti Cell - with NMDA
        python step_stim.py --test_with_repeats -c Martinotti\
                                --with_NMDA\
                                --with_presynaptic_spikes\
                                --Inh_fraction 0.15 --synapse_subsampling 2\
                                --stimFreq 2\
                                --stepAmpFactor ${ampFs[$i-1]}\
                                --stepWidth ${widths[$i-1]}\
                                --iBranch 5\
                                --nSpikeSeed $nSeed\
                                --suffix noSTP-Step$i
        # Martinotti Cell - no NMDA
        python step_stim.py --test_with_repeats -c Martinotti\
                                --with_presynaptic_spikes\
                                --Inh_fraction 0.15 --synapse_subsampling 2\
                                --stimFreq 2\
                                --AMPAboost 4\
                                --stepAmpFactor ${ampFs[$i-1]}\
                                --stepWidth ${widths[$i-1]}\
                                --iBranch 5\
                                --nSpikeSeed $nSeed\
                                --suffix noSTPnoNMDA-Step$i
        ' 
    done
    ### ----- SIMULATIONS WITH STP ----- ###
fi
if [[ $1 == 'all' || $1 == 'demo-step-2' ]]
then
    ### ----- SIMULATIONS WITH STP ----- ###
    nSeed=40
    widths=(100 100 1000)
    ampFs=(4 2 2)
    for i in 1 2 3
    do
        ## Basket Cell
        python step_stim.py --test_with_repeats -c Basket\
                                --with_presynaptic_spikes\
                                --stimFreq 4\
                                --stepAmpFactor ${ampFs[$i-1]}\
                                --stepWidth ${widths[$i-1]}\
                                --iBranch 1\
                                --nSpikeSeed $nSeed\
                                --suffix withSTP-Step$i
        # Martinotti Cell
        python step_stim.py --test_with_repeats -c Martinotti\
                                --with_NMDA\
                                --with_presynaptic_spikes\
                                --synapse_subsampling 2\
                                --stimFreq 1\
                                --stepAmpFactor ${ampFs[$i-1]}\
                                --stepWidth ${widths[$i-1]}\
                                --iBranch 5\
                                --nSpikeSeed $nSeed\
                                --suffix withSTP-Step$i
    done
fi

if [[ $1 == 'all' || $1 == 'full-step' ]]
then
    nSeed=4
    dt=0.05
    widths=(50 200 1000)
    nSeeds=(20 5 2)
    for i in 1 2 3
    do
        python step_stim.py --no_Vm -c Martinotti\
                                --with_NMDA --with_STP\
                                --Inh_fraction 0.15 --synapse_subsampling 2\
                                --stimFreq 1.\
                                --stepAmpFactor 2 3 4\
                                --interstim 400\
                                --stepWidth ${widths[$i-1]}\
                                --dt $dt --nSpikeSeed ${nSeeds[$i-1]}\
                                --suffix vSteps${i}Full
        python step_stim.py --no_Vm -c Martinotti\
                                --with_NMDA --with_STP\
                                --Inh_fraction 0.15 --synapse_subsampling 2\
                                --stimFreq 1.\
                                --stepAmpFactor 2 3 4\
                                --stepWidth ${widths[$i-1]}\
                                --interstim 500\
                                --dt $dt --nSpikeSeed ${nSeeds[$i-1]}\
                                --suffix vSteps${i}Full
        python step_stim.py --no_Vm -c Martinotti\
                                --with_NMDA\
                                --Inh_fraction 0.15 --synapse_subsampling 2\
                                --stimFreq 1.\
                                --stepAmpFactor 2 3 4\
                                --stepWidth ${widths[$i-1]}\
                                --interstim 500\
                                --dt $dt --nSpikeSeed ${nSeeds[$i-1]}\
                                --suffix vSteps${i}NoSTP
        python step_stim.py --no_Vm -c Martinotti\
                                --Inh_fraction 0.15 --synapse_subsampling 2\
                                --stimFreq 1.\
                                --stepAmpFactor 2 3 4\
                                --stepWidth ${widths[$i-1]}\
                                --interstim 500\
                                --AMPAboost 4\
                                --dt $dt --nSpikeSeed ${nSeeds[$i-1]}\
                                --suffix vSteps${i}NoNMDA
        python step_stim.py --no_Vm -c Basket\
                                --with_STP\
                                --Inh_fraction 0.1 --synapse_subsampling 2\
                                --stimFreq 3.\
                                --stepAmpFactor 2 3 4\
                                --stepWidth ${widths[$i-1]}\
                                --interstim 500\
                                --dt $dt --nSpikeSeed ${nSeeds[$i-1]}\
                                --suffix vSteps${i}Full
        python step_stim.py --no_Vm -c Basket\
                                --Inh_fraction 0.1 --synapse_subsampling 2\
                                --stimFreq 3.\
                                --stepAmpFactor 2 3 4\
                                --stepWidth ${widths[$i-1]}\
                                --interstim 500\
                                --dt $dt --nSpikeSeed ${nSeeds[$i-1]}\
                                --suffix vSteps${i}NoSTP
    done
fi


if [[ $1 == 'all' || $1 == 'step-ampa-calib' ]]
then
    nSeed=12
    # Martinotti Cell
    python step_stim.py --test_with_repeats -c Martinotti\
                            --AMPAboost 2 3 4 5 6 7\
                            --with_presynaptic_spikes\
                            --synapse_subsampling 2\
                            --stimFreq 1\
                            --stepAmpFactor 2\
                            --stepWidth 50 200 400\
                            --iBranch 5\
                            --interstim 300\
                            --nSpikeSeed $nSeed\
                            --suffix AMPAcalib
fi

if [[ $1 == 'all' || $1 == 'step-range-PV-noSTP' ]]
then
    nSeed=10
    ## Basket Cell
    python step_stim.py --test_with_repeats -c Basket\
      --no_Vm --nSpikeSeed $nSeed --synapse_subsampling 1 2 4\
      --stepWidth 200 --interstim 300 --suffix InputRangeNoSTP\
      --Inh_fraction 0.1 0.15 0.2 --stimFreq 4 5 6 7 8 --stepAmpFactor 2 4
fi

if [[ $1 == 'all' || $1 == 'step-range' ]]
then
    : '
    # Martinotti Cell
    python step_stim.py --test_with_repeats -c Martinotti\
                            --with_presynaptic_spikes\
                            --with_NMDA --with_STP\
                            --Inh_fraction 0.15 --synapse_subsampling 2\
                            --stimFreq 1.\
                            --stepAmpFactor 3\
                            --stepWidth 1000 --interstim 1000\
                            --iBranch 5 --nSpikeSeed $nSeed\
                            --suffix longFull
    python step_stim.py --test_with_repeats -c Martinotti\
                            --with_presynaptic_spikes\
                            --with_NMDA\
                            --Inh_fraction 0.15 --synapse_subsampling 2\
                            --stimFreq 1.\
                            --stepAmpFactor 3\
                            --stepWidth 1000 --interstim 1000\
                            --iBranch 5 --nSpikeSeed $nSeed\
                            --suffix longNoSTP
    python step_stim.py --test_with_repeats -c Martinotti\
                            --with_presynaptic_spikes\
                            --with_STP\
                            --Inh_fraction 0.15\
                            --synapse_subsampling 2\
                            --stimFreq 1.\
                            --stepAmpFactor 3\
                            --stepWidth 1000 --interstim 1000\
                            --iBranch 5 --nSpikeSeed $nSeed\
                            --suffix longNoNMDA
    python step_stim.py --test_with_repeats -c Martinotti\
                            --with_presynaptic_spikes\
                            --Inh_fraction 0.15\
                            --synapse_subsampling 2\
                            --stimFreq 1.\
                            --stepAmpFactor 3\
                            --stepWidth 1000 --interstim 1000\
                            --iBranch 5 --nSpikeSeed $nSeed\
                            --suffix longNoSTPNoNMDA
    python step_stim.py --test_with_repeats -c Martinotti\
                            --with_NMDA --with_STP\
                            --with_presynaptic_spikes\
                            --bgStimFreq 1\
                            --stimFreq 3\
                            --suffix withSTP\
                            --iBranch 5 --nSpikeSeed $nSeed
    python step_stim.py --test_with_repeats -c Martinotti\
                            --with_presynaptic_spikes\
                            --bgStimFreq 5\
                            --stimFreq 15\
                            --suffix noNMDA\
                            --iBranch 5 --nSpikeSeed $nSeed
    '
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
    nSpikeSeed=64
    dt=0.025
    tstop=15000
    ## Basket Cell
    python natMovie_sim.py --test_with_repeats -c Basket\
                            --with_presynaptic_spikes\
                            --iBranch 1\
                            --Inh_fraction 0.05 --synapse_subsampling 2\
                            --with_STP\
                            --dt $dt --tstop $tstop --nSpikeSeed $nSpikeSeed
    ## Basket Cell - no STP
    python natMovie_sim.py --test_with_repeats -c Basket\
                            --with_presynaptic_spikes\
                            --iBranch 1\
                            --Inh_fraction 0.05 --synapse_subsampling 2\
                            --dt $dt --tstop $tstop --nSpikeSeed $nSpikeSeed\
                            --suffix noSTP
    # -----------------------
    # Martinotti Cell
    python natMovie_sim.py --test_with_repeats -c Martinotti\
                            --with_presynaptic_spikes\
                            --iBranch 3\
                            --Inh_fraction 0.15 --synapse_subsampling 12\
                            --dt $dt --tstop $tstop --nSpikeSeed $nSpikeSeed\
                            --with_NMDA\
                            --with_STP
    # -----------------------
    # Martinotti Cell -- NO STP
    python natMovie_sim.py --test_with_repeats -c Martinotti\
                            --with_presynaptic_spikes\
                            --iBranch 3\
                            --Inh_fraction 0.15 --synapse_subsampling 12\
                            --dt $dt --tstop $tstop --nSpikeSeed $nSpikeSeed\
                            --with_NMDA\
                            --suffix noSTP
    # -----------------------
    # Martinotti Cell, no NMDA
    python natMovie_sim.py --test_with_repeats -c Martinotti\
                            --with_presynaptic_spikes\
                            --iBranch 3\
                            --Inh_fraction 0.1 --synapse_subsampling 4\
                            --dt $dt --tstop $tstop --nSpikeSeed $nSpikeSeed\
                            --with_STP\
                            --suffix noNMDA
    # -----------------------
    # Martinotti Cell, no STP no NMDA
    python natMovie_sim.py --test_with_repeats -c Martinotti\
                            --with_presynaptic_spikes\
                            --iBranch 3\
                            --Inh_fraction 0.1 --synapse_subsampling 4\
                            --dt $dt --tstop $tstop --nSpikeSeed $nSpikeSeed\
                             --suffix noSTPnoNMDA
fi

if [[ $1 == 'all' || $1 == 'full-natMovie' ]]
then
    nSpikeSeed=40
    dt=0.05
    tstop=50000
    ## Basket Cell
    python natMovie_sim.py -c Basket --no_Vm\
                            --Inh_fraction 0.05 --synapse_subsampling 2\
                            --with_STP\
                            --dt $dt --tstop $tstop --nSpikeSeed $nSpikeSeed\
                            --suffix Full
    # -----------------------
    # Basket Cell -- NO STP
    python natMovie_sim.py -c Basket --no_Vm\
                            --Inh_fraction 0.05 --synapse_subsampling 2\
                            --dt $dt --tstop $tstop --nSpikeSeed $nSpikeSeed\
                            --suffix FullnoSTP
    # -----------------------
    # Martinotti Cell
    python natMovie_sim.py -c Martinotti --no_Vm\
                            --Inh_fraction 0.15 --synapse_subsampling 12\
                            --dt $dt --tstop $tstop --nSpikeSeed $nSpikeSeed\
                            --with_NMDA\
                            --with_STP\
                            --suffix Full
    # -----------------------
    # Martinotti Cell -- NO STP
    python natMovie_sim.py -c Martinotti --no_Vm\
                            --Inh_fraction 0.15 --synapse_subsampling 12\
                            --dt $dt --tstop $tstop --nSpikeSeed $nSpikeSeed\
                            --with_NMDA\
                            --suffix FullnoSTP
    # -----------------------
    # Martinotti Cell, no NMDA
    python natMovie_sim.py -c Martinotti --no_Vm\
                            --Inh_fraction 0.1 --synapse_subsampling 4\
                            --dt $dt --tstop $tstop --nSpikeSeed $nSpikeSeed\
                            --with_STP\
                            --suffix FullnoNMDA
    # -----------------------
    # Martinotti Cell, no NMDA - no STP
    python natMovie_sim.py -c Martinotti --no_Vm\
                            --Inh_fraction 0.1 --synapse_subsampling 4\
                            --dt $dt --tstop $tstop --nSpikeSeed $nSpikeSeed\
                            --suffix FullnoSTPnoNMDA
    # -----------------------
fi

if [[ $1 == 'all' || $1 == 'input-range-natMovie' ]]
then
    nSpikeSeed=5
    tstop=10000
    dt=0.05
    Inh_range='0.05 0.1 0.15 0.2'
    SS_range='2 4 8 12'
    # -----------------------
    ## Basket Cell
    python natMovie_sim.py -c Basket\
                            --tstop $tstop --dt $dt\
                            --Inh_fraction $Inh_range\
                            --synapse_subsampling $SS_range\
                            --nSpikeSeed $nSpikeSeed\
                            --with_STP\
                            --no_Vm\
                            --suffix InputRange
    # -----------------------
    # Martinotti Cell
    python natMovie_sim.py -c Martinotti\
                            --tstop $tstop --dt $dt\
                            --Inh_fraction $Inh_range\
                            --synapse_subsampling $SS_range\
                            --nSpikeSeed $nSpikeSeed\
                            --with_STP\
                            --with_NMDA\
                            --no_Vm\
                            --suffix InputRange
    # -----------------------
    # -----------------------
    ## Basket Cell -- no STP
    python natMovie_sim.py -c Basket\
                            --tstop $tstop --dt $dt\
                            --Inh_fraction $Inh_range\
                            --synapse_subsampling $SS_range\
                            --nSpikeSeed $nSpikeSeed\
                            --no_Vm\
                            --suffix InputRange_noSTP
    # -----------------------
    # Martinotti Cell -- no NMDA
    python natMovie_sim.py -c Martinotti\
                            --tstop $tstop --dt $dt\
                            --Inh_fraction $Inh_range\
                            --synapse_subsampling $SS_range\
                            --nSpikeSeed $nSpikeSeed\
                            --with_STP\
                            --no_Vm\
                            --suffix InputRange_noNMDA
    # Martinotti Cell -- no STP
    python natMovie_sim.py -c Martinotti\
                            --tstop $tstop --dt $dt\
                            --Inh_fraction $Inh_range\
                            --synapse_subsampling $SS_range\
                            --nSpikeSeed $nSpikeSeed\
                            --with_NMDA\
                            --no_Vm\
                            --suffix InputRange_noSTP
    # Martinotti Cell -- no STP -- no NMDA
    python natMovie_sim.py -c Martinotti\
                            --tstop $tstop --dt $dt\
                            --Inh_fraction $Inh_range\
                            --synapse_subsampling $SS_range\
                            --nSpikeSeed $nSpikeSeed\
                            --no_Vm\
                            --suffix InputRange_noNMDAnoSTP
fi


#########################################
######## grating stim. rate (Fig. 8) ####
#########################################
#
if [[ $1 == 'all' || $1 == 'demo-grating' ]]
then
    nSeed=40
    : '
    ## Basket Cell
    python grating_stim.py --test_with_repeats -c Basket\
                            --with_presynaptic_spikes \
                            --bgStimFreq 8 --stimFreq 24\
                            --with_STP --suffix withSTP\
                            --iBranch 1 --nSpikeSeed $nSeed
    python grating_stim.py --test_with_repeats -c Basket\
                            --with_presynaptic_spikes\
                            --bgStimFreq 12 --stimFreq 36\
                            --iBranch 1 --nSpikeSeed $nSeed
    # Martinotti Cell
    python grating_stim.py --test_with_repeats -c Martinotti\
                            --with_presynaptic_spikes\
                            --bgStimFreq 1.5 --stimFreq 4.5\
                            --with_NMDA --iBranch 5 --nSpikeSeed $nSeed
    python grating_stim.py --test_with_repeats -c Martinotti\
                            --with_presynaptic_spikes\
                            --with_NMDA --with_STP --suffix withSTP\
                            --bgStimFreq 1 --stimFreq 3\
                            --iBranch 5 --nSpikeSeed $nSeed
    python grating_stim.py --test_with_repeats -c Martinotti\
                            --with_presynaptic_spikes\
                             --iBranch 5 --bgStimFreq 20 --stimFreq 60\
                            --suffix noNMDA --nSpikeSeed $nSeed
    '
    python grating_stim.py --test_with_repeats -c Martinotti\
                            --with_presynaptic_spikes\
                            --iBranch 5 --bgStimFreq 3 --stimFreq 9\
                            --with_STP --suffix noNMDAwithSTP --nSpikeSeed $nSeed
fi

if [[ $1 == 'all' || $1 == 'full-grating' ]]
then
    nSeed=40
    : '
    ## Basket Cell
    python grating_stim.py --no_Vm -c Basket\
                            --bgStimFreq 8 --stimFreq 24 --nSpikeSeed $nSeed
    python grating_stim.py --no_Vm -c Basket\
                            --bgStimFreq 12 --stimFreq 36\
                            --with_STP --suffix withSTP --nSpikeSeed $nSeed
    # Martinotti Cell
    python grating_stim.py --no_Vm -c Martinotti\
                            --with_NMDA --bgStimFreq 1.5 --stimFreq 4.5 --nSpikeSeed $nSeed
    python grating_stim.py --no_Vm -c Martinotti\
                            --bgStimFreq 1 --stimFreq 3\
                            --with_NMDA --with_STP --suffix withSTP --nSpikeSeed $nSeed
    python grating_stim.py --no_Vm -c Martinotti\
                            --bgStimFreq 20 --stimFreq 60\
                            --suffix noNMDA --nSpikeSeed $nSeed
    '
    python grating_stim.py --no_Vm -c Martinotti\
                            --with_STP --bgStimFreq 5 --stimFreq 15\
                            --suffix noNMDAwithSTP --nSpikeSeed $nSeed
fi


END=$(date +%s)
DIFF=$(( $END - $START ))
H=$(($DIFF/3600))
M=$(($(($DIFF%3600))/60))
S=$(($DIFF%60))
echo ""
echo ""
echo "     ------> Simulations took:" ${H}h:${M}m:${S}s
echo ""
