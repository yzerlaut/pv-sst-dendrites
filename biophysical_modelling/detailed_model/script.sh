##########################################################
########### dendro-somatic propagation ###################
##########################################################
# Martinotti Cell
#python propag.py -c Martinotti --nCluster 1 --test_active --suffix Single
#python propag.py -c Martinotti --nCluster 4 --test_active --suffix Multi
## Basket Cell
#python propag.py -c Basket --nCluster 1 --test_active --suffix Single
#python propag.py -c Basket --nCluster 4 --test_active --suffix Multi
###########################################################
########### stimulation on top of background #############
##########################################################
# Martinotti Cell
#python stim_on_background.py -c Martinotti --nCluster 0 5 10 15 20 25 30 35 40 45 --bgStimFreq 5e-4 --bgFreqInhFactor 1 --nStimRepeat 100 --test_NMDA --suffix Full --ISI 400
#python stim_on_background.py -c Martinotti --nCluster 0 5 10 15 20 25 30 35 --bgStimFreq 5e-4 --bgFreqInhFactor 1 --nStimRepeat 100 --test_uniform --suffix TestUniform --ISI 400 --with_NMDA
# Basket Cell
#python stim_on_background.py -c Basket --nCluster 0 5 10 15 20 25 30 35 40 45 50 55 --bgStimFreq 2e-3 --bgFreqInhFactor 1 --nStimRepeat 100 --test_uniform --suffix Full --ISI 400 
##########################################################
########### clustered input simulations ##################
##########################################################
# # --- demo data --- # #
# Basket cell
#for w in 6.25 12.5 25 50 100
#do
    #python intensity_timing_sim.py --test --with_presynaptic_spikes --freq 1e-2 --width $w --bgStimFreq 2e-3 --bgFreqInhFactor 1 -c Basket --from_uniform --iBranch 1 --filename ../../data/detailed_model/IT-Basket-w${w}ms-f10mHz.npy &
#done
# Martinotti cell
#for w in 6.25 12.5 25 50 100
#do
    #python intensity_timing_sim.py --test --with_presynaptic_spikes --freq 1e-2 --width $w --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --with_NMDA --filename ../../data/detailed_model/IT-Martinotti-w${w}ms-f10mHz.npy --iBranch 1 & 
#done
#for factor in 1 2 4 8 16
#do
    #w=$(echo "6.25 * $factor" | bc -l)
    #f=$(echo "20 / $factor" | bc -l)
    #python intensity_timing_sim.py --test --with_presynaptic_spikes --freq ${f}e-3 --width $w --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --with_NMDA --filename ../../data/detailed_model/IT-Martinotti-w${w}ms-f${f}mHz.npy --iBranch 1 & 
    #python intensity_timing_sim.py --test --with_presynaptic_spikes --freq ${f}e-3 --width $w --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --filename ../../data/detailed_model/IT-Martinotti-NO-NMDA-w${w}ms-f${f}mHz.npy --iBranch 1 & 
#done
# ---------- 
#python intensity_timing_sim.py --test --with_presynaptic_spikes --freq 1e-2 --width 10 --bgStimFreq 2e-3 --bgFreqInhFactor 1 -c Basket --from_uniform --iBranch 1 --filename ../../data/detailed_model/IT-Basket-w10ms-f10mHz.npy &
#python intensity_timing_sim.py --test --with_presynaptic_spikes --freq 2e-2 --width 100 --bgStimFreq 2e-3 --bgFreqInhFactor 1 -c Basket --from_uniform --iBranch 1 --filename ../../data/detailed_model/IT-Basket-100ms.npy &
# Martinotti cell
#python timing_sim.py --test --with_presynaptic_spikes --freq 2e-2 --width 10 --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --with_NMDA --filename ../../data/detailed_model/IT-Martinotti-10ms.npy --iBranch 1 & 
#python timing_sim.py --test --with_presynaptic_spikes --freq 2e-2 --width 100 --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --with_NMDA --filename ../../data/detailed_model/IT-Martinotti-100ms.npy --iBranch 1
# # --- full data --- # #
# Basket cell
python intensity_timing_sim.py --bgStimFreq 2e-3 --bgFreqInhFactor 1 -c Basket --nStimRepeat 10 --suffix Dual #--fix_missing_only
python intensity_timing_sim.py --bgStimFreq 2e-3 --bgFreqInhFactor 1 -c Basket --nStimRepeat 10 --suffix Dual --fix_missing_only
python intensity_timing_sim.py --bgStimFreq 2e-3 --bgFreqInhFactor 1 -c Basket --nStimRepeat 10 --suffix Dual --fix_missing_only
# Martinotti cell
python intensity_timing_sim.py --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --test_NMDA --nStimRepeat 10 --suffix Dual # --fix_missing_only
python intensity_timing_sim.py --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --test_NMDA --nStimRepeat 10 --suffix Dual --fix_missing_only
python intensity_timing_sim.py --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --test_NMDA --nStimRepeat 10 --suffix Dual --fix_missing_only
python intensity_timing_sim.py --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --test_NMDA --nStimRepeat 10 --suffix Dual --fix_missing_only
python intensity_timing_sim.py --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --test_NMDA --nStimRepeat 10 --suffix Dual --fix_missing_only
python intensity_timing_sim.py --bgStimFreq 5e-4 --bgFreqInhFactor 4 -c Martinotti --test_NMDA --nStimRepeat 10 --suffix Dual --fix_missing_only

##########################################################
########### clustered input simulations ##################
##########################################################
# Martinotti Cell
#python clustered_input_stim.py -c Matrinotti --test_uniform --sparsening 4 5 6 7 8 9 10 11 --passive
# Basket Cell
#python clustered_input_stim.py -c Basket --test_uniform --sparsening 3 4 5 6 7 8 9 10 --passive

##########################################################
########### other test simulations  ######################
##########################################################
# Martinotti Cell
#python stim_on_background.py -c Martinotti --nCluster 40 --bgStimFreq 5e-4 --bgFreqInhFactor 1 --nStimRepeat 100 --test_NMDA --suffix TimeCourse40 --ISI 300
#python stim_on_background.py -c Martinotti --nCluster 45 --bgStimFreq 5e-4 --bgFreqInhFactor 1 --nStimRepeat 100 --test_NMDA --suffix TimeCourse45 --ISI 300
#python stim_on_background.py -c Martinotti --nCluster 50 --bgStimFreq 5e-4 --bgFreqInhFactor 1 --nStimRepeat 100 --test_NMDA --suffix TimeCourse50 --ISI 300
# Basket Cell
#python stim_on_background.py -c Basket --nCluster 40 --bgStimFreq 2e-3 --bgFreqInhFactor 1 --nStimRepeat 100 --test_uniform --suffix TimeCourse40 --ISI 400 
#python stim_on_background.py -c Basket --nCluster 45 --bgStimFreq 2e-3 --bgFreqInhFactor 1 --nStimRepeat 100 --test_uniform --suffix TimeCourse45 --ISI 400 
#python stim_on_background.py -c Basket --nCluster 50 --bgStimFreq 2e-3 --bgFreqInhFactor 1 --nStimRepeat 100 --test_uniform --suffix TimeCourse50 --ISI 400 
#python clustered_input_stim.py -c Basket --test_uniform --sparsening 3 4 5 6 7 8 9 10





