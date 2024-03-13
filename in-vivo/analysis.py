import sys, os, pathlib
import numpy as np

physion_folder = os.path.join(pathlib.Path(__file__).resolve().parent,
                              '..', 'physion', 'src')
sys.path.append(os.path.join(physion_folder))
from physion.analysis.process_NWB import EpisodeData

stat_test_props = dict(interval_pre=[-1.5,0],
                       interval_post=[1,2.5],
                       test='ttest',
                       positive=True)

response_significance_threshold = 0.01

def selectivity_index(angles, resp):
    """
    computes the selectivity index: (Pref-Orth)/(Pref+Orth)
    clipped in [0,1]
    """
    imax = np.argmax(resp)
    iop = np.argmin(((angles[imax]+90)%(180)-angles)**2)
    if (resp[imax]>0):
        return min([1,max([0,(resp[imax]-resp[iop])/(resp[imax]+resp[iop])])])
    else:
        return 0

def shift_orientation_according_to_pref(angle,
                                        pref_angle=0,
                                        start_angle=-45,
                                        angle_range=360):
    new_angle = (angle-pref_angle)%angle_range
    if new_angle>=angle_range+start_angle:
        return new_angle-angle_range
    else:
        return new_angle


def compute_tuning_response_per_cells(data,
                                      imaging_quantity='dFoF',
                                      prestim_duration=None,
                                      stat_test_props=stat_test_props,
                                      response_significance_threshold = response_significance_threshold,
                                      contrast=1,
                                      protocol_name='ff-gratings-8orientation-2contrasts-10repeats',
                                      return_significant_waveforms=False,
                                      verbose=True):

    RESPONSES = []

    protocol_id = data.get_protocol_id(protocol_name=protocol_name)

    EPISODES = EpisodeData(data,
                           quantities=[imaging_quantity],
                           protocol_id=protocol_id,
                           prestim_duration=prestim_duration,
                           verbose=verbose)

    shifted_angle = EPISODES.varied_parameters['angle']-\
                            EPISODES.varied_parameters['angle'][1]

    significant_waveforms= []
    significant = np.zeros(data.nROIs, dtype=bool)

    for roi in np.arange(data.nROIs):

        cell_resp = EPISODES.compute_summary_data(stat_test_props,
                        response_significance_threshold=response_significance_threshold,
                        response_args=dict(quantity=imaging_quantity, roiIndex=roi))

        condition = (cell_resp['contrast']==contrast)

        # if significant in at least one orientation
        if np.sum(cell_resp['significant'][condition]):

            significant[roi] = True

            ipref = np.argmax(cell_resp['value'][condition])
            prefered_angle = cell_resp['angle'][condition][ipref]

            RESPONSES.append(np.zeros(len(shifted_angle)))

            for angle, value in zip(cell_resp['angle'][condition],
                                    cell_resp['value'][condition]):

                new_angle = shift_orientation_according_to_pref(angle,
                                                                pref_angle=prefered_angle,
                                                                start_angle=-22.5,
                                                                angle_range=180)
                iangle = np.flatnonzero(shifted_angle==new_angle)[0]

                RESPONSES[-1][iangle] = value

            if return_significant_waveforms:
                full_cond = EPISODES.find_episode_cond(\
                        key=['contrast', 'angle'],
                        value=[contrast, prefered_angle])
                significant_waveforms.append(getattr(EPISODES, imaging_quantity)[full_cond,roi,:].mean(axis=0))

    if return_significant_waveforms:
        return EPISODES.t, significant_waveforms
    else:
        return RESPONSES, significant, shifted_angle
