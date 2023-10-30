# Martinotti Cell
#python stim_on_background.py -c Martinotti --nCluster 40 --bgStimFreq 5e-4 --bgFreqInhFactor 1 --nStimRepeat 100 --test_NMDA --suffix TimeCourse40 --ISI 300
#python stim_on_background.py -c Martinotti --nCluster 45 --bgStimFreq 5e-4 --bgFreqInhFactor 1 --nStimRepeat 100 --test_NMDA --suffix TimeCourse45 --ISI 300
#python stim_on_background.py -c Martinotti --nCluster 50 --bgStimFreq 5e-4 --bgFreqInhFactor 1 --nStimRepeat 100 --test_NMDA --suffix TimeCourse50 --ISI 300
# Basket Cell
#python stim_on_background.py -c Basket --nCluster 40 --bgStimFreq 2e-3 --bgFreqInhFactor 1 --nStimRepeat 100 --test_uniform --suffix TimeCourse40 --ISI 400 
#python stim_on_background.py -c Basket --nCluster 45 --bgStimFreq 2e-3 --bgFreqInhFactor 1 --nStimRepeat 100 --test_uniform --suffix TimeCourse45 --ISI 400 
#python stim_on_background.py -c Basket --nCluster 50 --bgStimFreq 2e-3 --bgFreqInhFactor 1 --nStimRepeat 100 --test_uniform --suffix TimeCourse50 --ISI 400 
# Martinotti Cell
python stim_on_background.py -c Martinotti --nCluster 0 5 10 15 20 25 30 35 --bgStimFreq 5e-4 --bgFreqInhFactor 1 --nStimRepeat 100 --test_uniform --suffix Full2 --ISI 300 --with_NMDA
# Basket Cell
python stim_on_background.py -c Basket --nCluster 0 5 10 15 20 25 30 35 40 45 50 --bgStimFreq 2e-3 --bgFreqInhFactor 1 --nStimRepeat 100 --test_uniform --suffix Full --ISI 300 







