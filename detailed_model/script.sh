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
    nSeed=200
    cells=("Martinotti" "Martinotti" "Martinotti" "Basket")
    args=("--with_NMDA" "" "" "")
    suffix=("noSTP" "noSTPnoNMDA" "noSTPnoNMDA0" "noSTP")
    branch=(0 0 0 1)
    boost=(0 4.5 1.0 0)
    freqs=(1.5 1.5 1.5 8)
    for c in 1 2 3 4
    do
        widths=(50 50 200)
        ampFs=(3.5 2 2)
        for i in 1 2 3
        do
            python step_stim.py\
                --test_with_repeats\
                --with_presynaptic_spikes\
                -c ${cells[$c-1]} ${args[$c-1]}\
                --Inh_fraction 0.2\
                --synapse_subsampling 1\
                --stimFreq ${freqs[$c-1]}\
                --AMPAboost ${boost[$c-1]}\
                --stepAmpFactor ${ampFs[$i-1]}\
                --stepWidth ${widths[$i-1]}\
                --nSpikeSeed $nSeed\
                --iBranch ${branch[$c-1]}\
                --suffix ${suffix[$c-1]}$i
        done
    done
fi

if [[ $1 == 'all' || $1 == 'demo-step-2' ]]
then
    ### ----- SIMULATIONS WITH STP ----- ###
    nSeed=120
    cells=("Martinotti" "Martinotti" "Basket")
    args=("--with_NMDA --with_STP" "--with_STP" "--with_STP")
    suffix=("wiSTP" "wiSTPnoNMDA" "wiSTP")
    branch=(0 0 1)
    freqs=(1.2 1.2 8)
    for c in 1 2 3
    do
        widths=(200 200 1000)
        ampFs=(4 2 2)
        for i in 1 2 3
        do
            python step_stim.py\
                --test_with_repeats\
                --with_presynaptic_spikes\
                -c ${cells[$c-1]} ${args[$c-1]}\
                --Inh_fraction 0.2\
                --synapse_subsampling 1\
                --stimFreq ${freqs[$c-1]}\
                --AMPAboost 4.5\
                --stepAmpFactor ${ampFs[$i-1]}\
                --stepWidth ${widths[$i-1]}\
                --nSpikeSeed $nSeed\
                --interstim $(((2000-${widths[$i-1]})/2))\
                --iBranch ${branch[$c-1]}\
                --suffix ${suffix[$c-1]}$i
        done
    done
fi

if [[ $1 == 'all' || $1 == 'full-step' ]]
then
    cells=("Martinotti" "Martinotti" "Martinotti" "Martinotti"
           "Basket" "Basket")
    args=("--with_NMDA --with_STP" "--with_STP" "--with_NMDA" ""
          "--with_STP" "")
    suffix=("Full" "noNMDA" "noSTP" "noNMDAnoSTP" 
            "Full" "noSTP")
    freqs=(1.4 1.2 1.5 1.5 
           8.2 8.2)
    for c in 1 2 3 4 5 6
    do
        widths=(50 100 200 1000)
        #nSeeds=(20 8 4 4) # for debugging
        nSeeds=(160 80 20 20)
        for i in 1 2 3 4
        do
            python step_stim.py\
                --no_Vm\
                -c ${cells[$c-1]} ${args[$c-1]}\
                --Inh_fraction 0.2\
                --synapse_subsampling 1\
                --stimFreq ${freqs[$c-1]}\
                --AMPAboost 4.5\
                --stepAmpFactor 2 3 4\
                --stepWidth ${widths[$i-1]}\
                --nSpikeSeed ${nSeeds[$i-1]}\
                --suffix vSteps${suffix[$c-1]}$i
        done
    done
fi


if [[ $1 == 'all' || $1 == 'step-current-calib' ]]
then
    nSeed=4
    # Martinotti Cell
    python step_stim.py\
                -c Martinotti --no_Vm\
                --currentDrive 0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18\
                --stimFreq 1.5\
                --stepAmpFactor 2\
                --stepWidth 50\
                --interstim 300\
                --nSpikeSeed $nSeed\
                --suffix currentCalibnoSTP
    python step_stim.py\
                -c Martinotti --no_Vm\
                --with_STP\
                --currentDrive 0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18\
                --stimFreq 1.4\
                --stepAmpFactor 2\
                --stepWidth 50\
                --interstim 300\
                --nSpikeSeed $nSeed\
                --suffix currentCalibwiSTP
fi

if [[ $1 == 'all' || $1 == 'step-range-SST-noSTP' ]]
then
    nSeed=8
    ## Basket Cell
    python step_stim.py\
        -c Martinotti\
        --with_NMDA\
        --synapse_subsampling 1\
        --nSpikeSeed $nSeed\
        --stepWidth 200\
        --interstim 500\
        --dt 0.05\
        --suffix InputRangeNoSTP\
        --Inh_fraction 0.2\
        --stimFreq 0.8 1 1.2 1.4 1.6 1.8 2\
        --stepAmpFactor 2 4
fi

if [[ $1 == 'all' || $1 == 'step-range-PV-noSTP' ]]
then
    nSeed=8
    ## Basket Cell
    python step_stim.py\
        -c Basket\
        --synapse_subsampling 1\
        --nSpikeSeed $nSeed\
        --stepWidth 200\
        --interstim 500\
        --dt 0.05\
        --suffix InputRangeNoSTP\
        --Inh_fraction 0.2\
        --stimFreq 6 6.5 7 7.5 8 8.5 9\
        --stepAmpFactor 2 4
fi

if [[ $1 == 'all' || $1 == 'step-range' ]]
then
    nSeed=20
    args=("--with_NMDA --with_STP" "--with_STP" "--with_NMDA" "")
    suffix=("Full" "noNMDA" "noSTP" "noNMDAnoSTP")
    for c in 1 2 3 4
    do
        python step_stim.py\
            --no_Vm\
            -c Martinotti ${args[$c-1]}\
            --Inh_fraction 0.2\
            --synapse_subsampling 1\
            --stimFreq 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8\
            --AMPAboost 4.5\
            --stepAmpFactor 4\
            --stepWidth 50\
            --nSpikeSeed $nSeed\
            --suffix sRange${suffix[$c-1]}
    done
    cells=("Basket" "Basket")
    args=("--with_STP" "")
    suffix=("Full" "noSTP")
    for c in 1 2
    do
        python step_stim.py\
            --no_Vm\
            -c Basket ${args[$c-1]}\
            --Inh_fraction 0.2\
            --synapse_subsampling 1\
            --stimFreq 7.0 7.2 7.4 7.6 7.8 8.0 8.2 8.4 8.6 8.8 9.0\
            --stepAmpFactor 4\
            --stepWidth 50\
            --nSpikeSeed $nSeed\
            --suffix sRange${suffix[$c-1]}
    done
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
    nSeed=120
    cells=("Martinotti" "Martinotti" "Martinotti" "Basket" "Basket")
    args=("--with_NMDA --with_STP" "--with_NMDA" "" "--with_STP" "")
    suffix=("Full" "noNMDA" "noNMDAnoSTP" "Full" "noSTP")
    branch=(0 0 0 1 1)
    freqs=(1.2 1.2 1.2 9 9)
    for c in 1 2 4
    do
        python grating_stim.py --test_with_repeats\
            -c ${cells[$c-1]} ${args[$c-1]}\
            --with_presynaptic_spikes\
            --Inh_fraction 0.2\
            --synapse_subsampling 1\
            --stimFreq ${freqs[$c-1]}\
            --stepAmpFactor 4\
            --iBranch ${branch[$c-1]}\
            --nSpikeSeed $nSeed\
            --suffix ${suffix[$c-1]}$i
    done
fi

if [[ $1 == 'all' || $1 == 'full-grating' ]]
then
    nSeed=96
    cells=("Martinotti" "Martinotti" "Martinotti" "Martinotti" "Basket" "Basket")
    args=("--with_NMDA --with_STP" "--with_STP" "--with_NMDA" "" "--with_STP" "")
    suffix=("Full" "noNMDA" "noSTP" "noNMDAnoSTP" "Full" "noSTP")
    freqs=(1.2 1.2 1.2 1.2 9.0 9.0)
    for c in 1 2 5
    do
        python grating_stim.py\
            -c ${cells[$c-1]} ${args[$c-1]}\
            --with_presynaptic_spikes\
            --Inh_fraction 0.2\
            --synapse_subsampling 1\
            --stimFreq ${freqs[$c-1]}\
            --stepAmpFactor 4\
            --nSpikeSeed $nSeed\
            --suffix ${suffix[$c-1]}$i
    done
fi

if [[ $1 == 'all' || $1 == 'grating-ampa-ratio' ]]
then
    nSeed=12
    python grating_stim.py --test_with_repeats\
        -c Martinotti --with_STP\
        --with_presynaptic_spikes\
        --Inh_fraction 0.2\
        --synapse_subsampling 1\
        --stimFreq 2.0\
        --AMPAboost 1.5 2 2.5 3 3.5\
        --stepAmpFactor 4\
        --iBranch 0\
        --nSpikeSeed $nSeed\
        --suffix AmpaRationoNMDA
fi

if [[ $1 == 'all' || $1 == 'grating-range' ]]
then
    nSeed=4
    args=("--with_NMDA --with_STP" "--with_STP" "--with_NMDA" "")
    suffix=("Full" "noNMDA" "noSTP" "noNMDAnoSTP")
    for c in 1 2
    do
        python grating_stim.py\
            -c Martinotti ${args[$c-1]}\
            --with_presynaptic_spikes\
            --Inh_fraction 0.2\
            --synapse_subsampling 1\
            --stimFreq 1.0 1.1 1.2 1.3 1.4 1.5\
            --stepAmpFactor 4\
            --nSpikeSeed $nSeed\
            --suffix ${suffix[$c-1]}${i}Range
    done
    nSeed=4
    args=("--with_STP" "")
    suffix=("Full" "noSTP")
    for c in 1
    do
        python grating_stim.py\
            -c Basket ${args[$c-1]}\
            --Inh_fraction 0.2\
            --synapse_subsampling 1\
            --stimFreq 7 8 9 10 11 12 13 14\
            --stepAmpFactor 4\
            --nSpikeSeed $nSeed\
            --suffix ${suffix[$c-1]}${i}Range
    done
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
