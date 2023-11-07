##########################################################
########### dendro-somatic propagation ###################
##########################################################
# Martinotti Cell
python propag.py -c Martinotti --nCluster 1 --test_active --suffix Single
python propag.py -c Martinotti --nCluster 4 --test_active --suffix Multi
# Basket Cell
python propag.py -c Basket --nCluster 1 --test_active --suffix Single
python propag.py -c Basket --nCluster 4 --test_active --suffix Multi
##########################################################
########### stimulation on top of background #############
##########################################################
# Martinotti Cell
python stim_on_background.py -c Martinotti --nCluster 0 5 10 15 20 25 30 35 40 45 --bgStimFreq 5e-4 --bgFreqInhFactor 1 --nStimRepeat 100 --test_NMDA --suffix Full --ISI 400
python stim_on_background.py -c Martinotti --nCluster 0 5 10 15 20 25 30 35 --bgStimFreq 5e-4 --bgFreqInhFactor 1 --nStimRepeat 100 --test_uniform --suffix FullUniformComp --ISI 400 --with_NMDA
# Basket Cell
python stim_on_background.py -c Basket --nCluster 0 5 10 15 20 25 30 35 40 45 50 55 --bgStimFreq 2e-3 --bgFreqInhFactor 1 --nStimRepeat 100 --test_uniform --suffix Full --ISI 400 
##########################################################
########### clustered input simulations ##################
##########################################################
# Martinotti Cell
python clustered_input_stim.py -c Matrinotti --test_uniform --sparsening 4 5 6 7 8 9 10 11 --passive
# Basket Cell
python clustered_input_stim.py -c Basket --test_uniform --sparsening 3 4 5 6 7 8 9 10 --passive

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





