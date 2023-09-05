# Biophysical Modelling

## Library of Channels

| **Channel Type**  |  **Description**  | **Link to `mod` file** |
| --- | --- | --- |
| Passive current                |                                     |    |
| A-type Potassium current       | see Klee, Ficker & Heinemann (1995) | [`kaproxin.mod`](https://github.com/ModelDBRepository/237595/blob/master/Multicompartmental_Biophysical_models/mechanism/kaproxin.mod)  |
| --- | --- | --- |

## Channel Types and Density in All Compartments

### PV+ cells

|     | **Channel Type**  |  **Name**  |  **Density** (mS/cm<sup>2</sup>) |
| --- | --- | --- | --- |
| **soma** |     |     |     |
|          | Passive current                               | `pas` | 1.32 10<sup>-4</sup> |
|          | Fast Sodium current ("with fast attenuation") | `Nafx`  | 1.35 10<sup>-1</sup> |
|          | Delayed rectifier pottassium current          | `Kdrin`  | 3.60 10<sup>-2</sup>  |
|          | Slowly inactivating Potassium current         | `Kslowin`  | 7.25 10<sup>-4</sup> |
|          | H-type Potassium current (uses Na+ ions ??)   | `Hin`  | 1.00 10<sup>-5</sup> |
|          | A-type Potassium current                      | `Kapin` | 3.20 10<sup>-3</sup> |
|          |                                               | `Can` | 0.0003 |
|          |                                               | `Cat` | 0.001*2 |
|          |                                               | `Kctin` | 0.0001 |
|          |                                               | `Kcain` | 20*0.001 |
|   **basal dendrites** |     |     |     |
| --- | --- | --- | --- |

### SST+ cells

