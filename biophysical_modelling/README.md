# Biophysical Modelling

## Library of Channels

| **Channel Type**  |  **Description**  | **Link to `mod` file** |
| --- | --- | --- |
| Passive current                         | classical leak current                          |    |
| Fast Sodium current                     | classical Na+ channel, with a fast inactivation |  [`nafx.mod`](./mechanisms/nafx.mod)  |
| Delayed rectifier Potassium current     |                                                 |  [`kdrin.mod`](./mechanisms/kdrin.mod)  |
| Slowly inactivating Potassium current   |                                                 |  [`iksin.mod`](./mechanisms/iksin.mod)  |
| H-type cation current                   |       (here implementation uses Na+ ions)       |  [`hin.mod`](./mechanisms/hin.mod)  |
| A-type Potassium current                | see Klee, Ficker & Heinemann (1995)             |  [`kaproxin.mod`](./mechanisms/kaproxin.mod)  |
| Calcium dependent Potassium current     | Shao et al., *J.Physiol* (1999)[^S99], spike broadening  | [`kctin.mod`](./mechanisms/kctin.mod)  |
| Calcium dependent Potassium current     | responsible for slow AHP dynamics               |  [`kcain.mod`](./mechanisms/kcain.mod)  |
| + Calcium dynamics                      | simple first order model                        |  [`cadynin.mod`](./mechanisms/cadynin.mod)  |
| --- | --- | --- |

## Channel Types and Density in All Compartments

### PV+ cells

|     | **Channel Type**  |  **Name**  |  **Density** (S/cm<sup>2</sup>) |
| --- | --- | --- | --- |
| **soma** |     |     |     |
|          | Passive current                               | `pas` | 1.32 10<sup>-4</sup> |
|          | Fast Sodium current ("with fast attenuation") | `Nafx` | 1.35 10<sup>-1</sup> |
|          | Delayed rectifier pottassium current          | `Kdrin` | 3.60 10<sup>-2</sup>  |
|          | Slowly inactivating Potassium current         | `Kslowin` | 7.25 10<sup>-4</sup> |
|          | H-type Potassium current (uses Na+ ions ??)   | `Hin` | 1.00 10<sup>-5</sup> |
|          | A-type Potassium current                      | `Kapin` | 3.20 10<sup>-3</sup> |
|          | A-type Potassium current                      | `Kapin` | 3.20 10<sup>-3</sup> |
| **axon** |     |     |     |
|          | Passive current                               | `pas` | 1.32 10<sup>-4</sup> |
| **proximal dendrites** | ($\leq$ 100 um) |     |     |     |
|          | Passive current                               | `pas` | 1.32 10<sup>-4</sup> |
|          |                                               | `Can` | 0.0003 |
|          |                                               | `Cat` | 0.001*2 |
|          |                                               | `Kctin` | 0.0001 |
|          |                                               | `Kcain` | 20*0.001 |
| **distal dendrites** |     |     |     |
| **basal dendrites** |     |     |     |
| --- | --- | --- | --- |

### SST+ cells


### References

[^S99]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2269638/
