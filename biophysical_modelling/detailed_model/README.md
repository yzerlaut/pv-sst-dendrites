# Biophysical Modelling

## Passive Properties

### Basket Cell

<!--| **leak conductance**       |                   |                 |                      |                        | N.B. x2.7 to match ~80-100MOhm input resistance         |-->
| **Parameter description**  | **location**      |  **Name**       |     **Value**        | **Unit**               |                                                         | 
|----------------------------|-------------------|-----------------|----------------------|------------------------|---------------------------------------------------------|
| **leak conductance**       |                   |                 |                      |                        |                                                         |
|                            | soma              | `BC_soma_gPas`  | 4.37 10<sup>-4</sup> | S/cm<sup>2</sup>       |                                                         |
|                            | proximal dendrite | `BC_prox_gPas`  | 4.37 10<sup>-4</sup> | S/cm<sup>2</sup>       |                                                         |
|                            | distal dendrite  | `BC_dist_gPas`  | 4.46 10<sup>-5</sup> | S/cm<sup>2</sup>       |                                                         |
| **leak reversal potential**|                   |                 |                      |                        |                                                         |
|                            | *all*             | `BC_ePas`       | -70.0                | mV                     |                                                         |
| **axial resistance**       |                   |                 |                      |                        |                                                         |
|                            | soma              | `BC_soma_Ra`    | 172                  | ohm.cm                 |                                                         |
|                            | proximal dendrite | `BC_prox_Ra`    | 142                  | ohm.cm                 |                                                         |
|                            | proximal dendrite | `BC_dist_Ra`    | 142                  | ohm.cm                 |                                                         |
| **membrane capacitance**   |                   |                 |                      |                        |                                                         |
|                            | soma              | `BC_soma_cm`    | 1.2                  | uF/cm<sup>2</sup>      |                                                         |
|                            | proximal dendrite | `BC_prox_cm`    | 1.2                  | uF/cm<sup>2</sup>      |                                                         |
|                            | proximal dendrite | `BC_dist_cm`    | 1.2                  | uF/cm<sup>2</sup>      |                                                         |

### Martinotti Cell

| **Parameter description**  | **location**      |  **Name**       |     **Value**        | **Unit**               |                                                         | 
|----------------------------|-------------------|-----------------|----------------------|------------------------|---------------------------------------------------------|
| **leak conductance**       |                   |                 |                      |                        |                                                             |
|                            | soma              | `MC_soma_gPas`  | 3.50 10<sup>-5</sup> | S/cm<sup>2</sup>       |                                                          |
|                            | proximal dendrite | `MC_prox_gPas`  | 3.50 10<sup>-5</sup> | S/cm<sup>2</sup>       |                                                          |
|                            | distal dendrite   | `MC_dist_gPas`  | 3.50 10<sup>-5</sup> | S/cm<sup>2</sup>       |                                                          |
| **leak reversal potential**|                   |                 |                      |                        |                                                         |
|                            | *all*             | `MC_ePas`       | -60.0                | mV                     |                                                         |
| **axial resistance**       |                   |                 |                      |                        |                                                         |
|                            | soma              | `MC_soma_Ra`    | 172                  | ohm.cm                 |                                                         |
|                            | proximal dendrite | `MC_prox_Ra`    | 142                  | ohm.cm                 |                                                         |
|                            | proximal dendrite | `MC_dist_Ra`    | 142                  | ohm.cm                 |                                                         |
| **membrane capacitance**   |                   |                 |                      |                        |                                                         |
|                            | soma              | `MC_soma_cm`    | 1.2                  | uF/cm<sup>2</sup>      |                                                         |
|                            | proximal dendrite | `MC_prox_cm`    | 1.2                  | uF/cm<sup>2</sup>      |                                                         |
|                            | proximal dendrite | `MC_dist_cm`    | 1.2                  | uF/cm<sup>2</sup>      |                                                         |


## Active Mechanisms

### Library of Channels

*channels used in the study:*

| **Channel Type**  |  **Description/Comments**  | **Link to `mod` file** |
| --- | --- | --- |
| Passive current                         | classical leak current                                 |    |
| Sodium current                          | classical Na+ channel, with a fast inactivation        |  [`nafx.mod`](./mechanisms/nafx.mod)  |
| Delayed rectifier Potassium current     |                                                        |  [`kdrin.mod`](./mechanisms/kdrin.mod)  |
| A-type Potassium current (proximal)     | Klee et al. *J. Physiol.* (1995)[^K95]                 |  [`kaproxin.mod`](./mechanisms/kaproxin.mod)  |
| A-type Potassium current (distal)       | Klee et al. *J. Physiol.* (1995)[^K95]                 |  [`kadistin.mod`](./mechanisms/kadistin.mod)  |
| M-type Potassium current                |                                                        |  [`kadistin.mod`](./mechanisms/IM.mod)  |
| + Calcium dynamics                      | simple first order model                               |  [`cadynin.mod`](./mechanisms/cadynin.mod)  |
| --- | --- | --- |


### Channel Types and Density in All Compartments

#### Basket Cell (based on PFC model[^T19])

|     | **Channel Type**  |  **Name**  |  **Density** (S/cm<sup>2</sup>) |           _comment_           |
| --- | --- | --- | --- | --- | 
| **soma** |     |     |     |          |
|          | Fast Sodium current                           | `BC_soma_gNafx`    | 3.35 10<sup>-1</sup> | original 1.35e-1     |
|          | Delayed rectifier Potassium current           | `BC_soma_gKdrin`   | 9.60 10<sup>-2</sup> | original 3.60e-2     |
|          | A-type Potassium current (proximal)           | `BC_soma_gKapin`   | 6.40 10<sup>-3</sup> | original 3.20e-3     |
| **proximal dendrites** | ($\leq$ 100 um from soma)       |                    |                      |                      |
|          | Fast Sodium current                           | `BC_prox_gNafx`    | 0.00 10<sup>01</sup> |                      |
|          | Delayed rectifier Potassium current           | `BC_prox_gKdrin`   | 0.00 10<sup>01</sup> |                      |
|          | A-type Potassium current (proximal)           | `BC_prox_gKapin`   | 2.00 10<sup>-3</sup> | original 1.00e-3     |
| **distal dendrites**   | ($\geq$ 100 um from soma)       |                    |                      |                      |
|          | Fast Sodium current                           | `BC_dist_gNafx`    | 0.00 10<sup>01</sup> |                      |
|          | Delayed rectifier Potassium current           | `BC_dist_gKdrin`   | 0.00 10<sup>01</sup> |                      |
|          | A-type Potassium current (proximal)           | `BC_dist_gKapin`   | 2.00 10<sup>-3</sup> | original 9.00e-4     |
|          | A-type Potassium current (distal)             | `BC_dist_gKadin`   | 4.00 10<sup>-3</sup> | original 2.16e-3     |
| --- | --- | --- | --- |


#### Martinotti Cell

|     | **Channel Type**  |  **Name**  |  **Density** (S/cm<sup>2</sup>) |           _comment_           |
| --- | --- | --- | --- | --- | 
| **soma** |     |     |     |          |
|          | Fast Sodium current                           | `MC_soma_gNafx`    | 7.00 10<sup>-1</sup> | original 1.35e-1 |
|          | Delayed rectifier Potassium current           | `MC_soma_gKdrin`   | 4.00 10<sup>-1</sup> | original 3.60e-2 |
|          | M-type current                                | `MC_soma_gM`       | 2.00 10<sup>-2</sup> |                  |
| **proximal dendrites** | ($\leq$ 100 um from soma) |     |     |     |
|          | Fast Sodium current                           | `MC_prox_gNafx`    | 0.00 10<sup>-2</sup> |
|          | Delayed rectifier Potassium current           | `MC_prox_gKdrin`   | 0.00 10<sup>-4</sup> |
|          | M-type current                                | `MC_prox_gM`       | 0.00 10<sup>01</sup> |                  |
| **distal dendrites** | ($\leq$ 100 um from soma) |    |     |     |
|          | Fast Sodium current                           | `MC_dist_gNafx`    | 0.00 10<sup>-2</sup> |
|          | Delayed rectifier Potassium current           | `MC_dist_gKdrin`   | 0.00 10<sup>-3</sup> |
|          | M-type current                                | `MC_dist_gM`       | 0.00 10<sup>01</sup> |                  |
| --- | --- | --- | --- |

## Synaptic properties

|                   | **Parameter description**  |  **Name**  |     **Value**      | **Unit**  |                                                                      |
|-------------------|----------------------------|------------|--------------------|-----------|----------------------------------------------------------------------|
| **AMPA receptor** |                            |            |                    |           |                                                                      |
|                   | conductance quantal        | `BC_qAMPA` | 1.0 10<sup>-3</sup>| uS        |    in Basket cell                                                    |
|                   | conductance quantal        | `MC_qAMPA` | 1.0 10<sup>-3</sup>| uS        |    in Martinotti cell                                                |
|                   | decay time constant        |            | 2.0                | ms        | changed directly in the [`ampain.mod`](./mechanisms/ampain.mod) file |
|-------------------|----------------------------|------------|--------------------|-----------|----------------------------------------------------------------------|
| **NMDA receptor** |                            |            |                    |           |                                                                      |
|                   | NMDA/AMPA quantal ratio    | `BC_NAR`   | 0                  |           |   in Basket cell                                                     |
|                   | NMDA/AMPA quantal ratio    | `MC_NAR`   | 1.5                |           |   in Martinotti cell                                                 |
|                   | decay time constant        |            | 80.0               | ms        | changed directly in the [`nmdain.mod`](./mechanisms/nmdain.mod) file |
|-------------------|----------------------------|------------|--------------------|-----------|----------------------------------------------------------------------|
| **GABAa receptor**|                            |            |                    |           |                                                                      |
|                   | conductance quantal        | `BC_qGABA` | 4.0 10<sup>-3</sup>| uS        |    in Basket cell                                                    |
|                   | conductance quantal        | `MC_qGABA` | 4.0 10<sup>-3</sup>| uS        |    in Martinotti cell                                                |
|-------------------|----------------------------|------------|--------------------|-----------|----------------------------------------------------------------------|

## References

[^T19]: https://www.nature.com/articles/s41467-019-11537-7
[^K95]: https://journals.physiology.org/doi/abs/10.1152/jn.1995.74.5.1982
[^S99]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2269638/

