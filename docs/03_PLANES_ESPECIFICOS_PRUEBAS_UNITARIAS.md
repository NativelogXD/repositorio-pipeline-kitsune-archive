# Documentación Específica de Cada Plan de Pruebas Unitarias (`PU-01` a `PU-08`)

> **Proyecto:** `Kitsune-py`  
> **Directorio de Implementación:** [`tests/unit/`](../tests/unit)  
> **Framework de Ejecución:** `pytest` + `pytest-cov`  
> **Semilla de Reproducibilidad:** `np.random.default_rng(seed=42)` (en [`tests/conftest.py`](../tests/conftest.py)) y `numpy.random.RandomState(1234)` (interno en `KitNET/dA.py`).

---

## Índice de Planes de Pruebas Unitarias por Módulo

1. [**Plan Unitario 1 (`PU-01` — `PU-UTILS`)**: Funciones Matemáticas, Distribuciones y Ventana Móvil (`KitNET/utils.py`)](#1-plan-de-pruebas-unitarias-pu-01--módulo-kitnetutilspy)
2. [**Plan Unitario 2 (`PU-02` — `PU-DA`)**: Denoising Autoencoder y Retropropagación SGD (`KitNET/dA.py`)](#2-plan-de-pruebas-unitarias-pu-02--módulo-kitnetdapy)
3. [**Plan Unitario 3 (`PU-03` — `PU-CORCLUST`)**: Clustering Incremental de Correlación (`KitNET/corClust.py`)](#3-plan-de-pruebas-unitarias-pu-03--módulo-kitnetcorclustpy)
4. [**Plan Unitario 4 (`PU-04` — `PU-KITNET`)**: Orquestador del Ensamble de Autoencoders (`KitNET/KitNET.py`)](#4-plan-de-pruebas-unitarias-pu-04--módulo-kitnetkitnetpy)
5. [**Plan Unitario 5 (`PU-05` — `PU-AFTERIMAGE`)**: Estadísticas Incrementales Amortiguadas 1D/2D y Base Hash (`AfterImage.py`)](#5-plan-de-pruebas-unitarias-pu-05--módulo-afterimagepy)
6. [**Plan Unitario 6 (`PU-06` — `PU-NETSTAT`)**: Estadísticas Multiperspectiva de Tráfico de Red (`netStat.py`)](#6-plan-de-pruebas-unitarias-pu-06--módulo-netstatpy)
7. [**Plan Unitario 7 (`PU-07` — `PU-FE`)**: Captura y Extracción de Paquetes TSV/PCAP (`FeatureExtractor.py`)](#7-plan-de-pruebas-unitarias-pu-07--módulo-featureextractorpy)
8. [**Plan Unitario 8 (`PU-08` — `PU-KITSUNE`)**: Fachada Principal del Detector NIDS (`Kitsune.py`)](#8-plan-de-pruebas-unitarias-pu-08--módulo-kitsunepy)

*(Nota: Cada plan también dispone de su ficha individual en [`docs/planes_unitarios/`](planes_unitarios/)).*

---

## 1. Plan de Pruebas Unitarias `PU-01` — Módulo [`KitNET/utils.py`](../KitNET/utils.py)

* **Identificador del Plan:** `PU-01-UTILS`
* **Archivo de Prueba:** [`tests/unit/test_utils.py`](../tests/unit/test_utils.py)
* **Documento Individual:** [`docs/planes_unitarios/PU_01_KitNET_utils.md`](planes_unitarios/PU_01_KitNET_utils.md)
* **Objetivo Técnico:** Verificar la exactitud numérica, estabilidad ante desbordamiento (*overflow*), simetría y vectorización de las funciones estadísticas (`pdf`, `invLogCDF`), funciones de activación neuronal (`sigmoid`, `dsigmoid`, `tanh`, `dtanh`, `softmax`, `ReLU`, `dReLU`) y el buffer circular `rollmean`.
* **Cobertura Alcanzada:** **100%** (35/35 sentencias).
* **Comando de Ejecución Aislada:** `pytest tests/unit/test_utils.py -v`

### Matriz de Casos de Prueba (`PU-01`)

| ID Caso | Método / Función Bajo Prueba | Entrada de Prueba | Resultado Esperado / Invariante Matemático | Estado |
| :--- | :--- | :--- | :--- | :---: |
| **`UT-UTILS-001`** | `pdf(x, mu, sigma)` | `x=0.0, mu=0.0, sigma=1.0` | Retorna $\frac{1}{\sqrt{2\pi}} \approx 0.39894228$ con tolerancia relativa $10^{-7}$. | ✅ PASS |
| **`UT-UTILS-002`** | `pdf(x, mu, sigma)` | `mu=5.0, sigma=2.0`, desplazamientos $\pm\Delta x$ | Simetría gaussiana exacta: $\text{pdf}(\mu - \Delta x) == \text{pdf}(\mu + \Delta x)$. | ✅ PASS |
| **`UT-UTILS-003`** | `invLogCDF(x, mu, sigma)` | `x=[0, 1, 2, 3], mu=1.0, sigma=0.5` | Coincide exactamente con `scipy.stats.norm.logcdf(-(x - mu)/sigma)`. | ✅ PASS |
| **`UT-UTILS-004`** | `sigmoid(x)` | `x=[-10, -1, 0, 1, 10]` | $\sigma(0) = 0.5$, acotamiento estricto en $(0, 1)$ y monotonía creciente ($\Delta \sigma > 0$). | ✅ PASS |
| **`UT-UTILS-005`** | `dsigmoid(y)` | `y=[0.1, 0.25, 0.5, 0.75, 0.9]` | Satisface la identidad $\text{dsigmoid}(y) = y(1-y)$ y máximo local $0.25$ en $y=0.5$. | ✅ PASS |
| **`UT-UTILS-006`** | `tanh(x)` y `dtanh(y)` | `x=[-2.0, -0.5, 0.0, 0.5, 2.0]` | Imparidad $\tanh(-x) = -\tanh(x)$ y derivada $\text{dtanh}(y) = 1 - y^2$. | ✅ PASS |
| **`UT-UTILS-007`** | `softmax(x)` | Vector 1D `[1, 2, 3]`, desplazado `+1000`, y matriz 2D `(2, 3)` | Suma de probabilidades igual a $1.0$ por fila e invarianza ante traslación constante (sin `NaN`/`Inf`). | ✅ PASS |
| **`UT-UTILS-008`** | `ReLU(x)` y `dReLU(x)` | `x=[-5.0, -0.01, 0.0, 0.01, 5.0]` | $\text{ReLU}(x) = \max(0, x)$ y escalón unitario $\text{dReLU}(x) = \mathbb{I}(x > 0)$. | ✅ PASS |
| **`UT-UTILS-009`** | `rollmean(k=3).apply(val)` | Secuencia `3.0 -> 6.0 -> 9.0 -> 12.0` | Medias sucesivas `1.0 -> 3.0 -> 6.0 -> 9.0` y aritmética modular del puntero `(pointer + 1) % k`. | ✅ PASS |

---

## 2. Plan de Pruebas Unitarias `PU-02` — Módulo [`KitNET/dA.py`](../KitNET/dA.py)

* **Identificador del Plan:** `PU-02-DA`
* **Archivo de Prueba:** [`tests/unit/test_da.py`](../tests/unit/test_da.py)
* **Documento Individual:** [`docs/planes_unitarios/PU_02_KitNET_dA.md`](planes_unitarios/PU_02_KitNET_dA.md)
* **Objetivo Técnico:** Verificar la construcción del Denoising Autoencoder (`dA` y `dA_params`), el cálculo del número de neuronas ocultas por `hiddenRatio`, la reproducibilidad por semilla fija (`1234`), el seguimiento de extremos min-max online, la corrupción binomial de entradas, la convergencia del entrenamiento por descenso de gradiente estocástico (SGD) y la compuerta de periodo de gracia (`inGrace` / `execute`).
* **Cobertura Alcanzada:** **100%** (65/65 sentencias).
* **Comando de Ejecución Aislada:** `pytest tests/unit/test_da.py -v`

### Matriz de Casos de Prueba (`PU-02`)

| ID Caso | Método / Función Bajo Prueba | Entrada de Prueba | Resultado Esperado / Invariante Matemático | Estado |
| :--- | :--- | :--- | :--- | :---: |
| **`UT-DA-001`** | `dA.__init__(params)` | `n_visible=10, hiddenRatio=0.75` | `n_hidden == ceil(10 * 0.75) == 8`, `W.shape == (10, 8)`, pesos atados `W_prime == W.T` y $W_{ij} \in [-0.1, 0.1]$. | ✅ PASS |
| **`UT-DA-002`** | `dA.__init__(params)` | Dos instancias independientes con `n_visible=6, hiddenRatio=0.5` | Pesos iniciales idénticos gracias al generador determinista `RandomState(1234)`. | ✅ PASS |
| **`UT-DA-003`** | `dA.train(x)` (Normalización) | `x1=[10, -5, 0]`, `x2=[2, 8, -3]` | `norm_min` actualiza a `[2, -5, -3]`, `norm_max` a `[10, 8, 0]`, contador `n == 2` y RMSE $\ge 0$. | ✅ PASS |
| **`UT-DA-004`** | `dA.get_corrupted_input()` y `train()` con corrupción | Vector `x` de 20 elementos, `corruption_level=0.3` y `1.0` | Con `0.3` enmascara entradas a `{0.0, 5.0}` y ejecuta `train(x)`; con `1.0` lanza `AssertionError`. | ✅ PASS |
| **`UT-DA-005`** | `dA.execute(x)` e `inGrace()` | `gracePeriod=3`, 3 llamadas a `train()` | Antes de 3 entrenamientos: `inGrace() == True` y `execute(x) == 0.0`. Tras 3 entrenamientos: `inGrace() == False` y `execute(x) > 0.0`. | ✅ PASS |
| **`UT-DA-006`** | `dA.train(x)` y `execute(x)` (Convergencia SGD) | 150 iteraciones sobre patrón estacionario en $\mathbb{R}^5$ vs patrón anómalo ortogonal | El RMSE final de entrenamiento es estrictamente menor que el inicial ($\text{RMSE}_{\text{final}} < \text{RMSE}_{\text{inicial}}$) y $\text{RMSE}_{\text{anomalía}} > \text{RMSE}_{\text{normal}}$. | ✅ PASS |

---

## 3. Plan de Pruebas Unitarias `PU-03` — Módulo [`KitNET/corClust.py`](../KitNET/corClust.py)

* **Identificador del Plan:** `PU-03-CORCLUST`
* **Archivo de Prueba:** [`tests/unit/test_corclust.py`](../tests/unit/test_corclust.py)
* **Documento Individual:** [`docs/planes_unitarios/PU_03_KitNET_corClust.md`](planes_unitarios/PU_03_KitNET_corClust.md)
* **Objetivo Técnico:** Verificar la actualización incremental vectorizada de residuos y matriz de covariación parcial $C$, la estabilidad numérica de `corrDist()` ante características constantes (varianza nula), y la partición jerárquica recursiva del dendrograma `cluster(maxClust)` garantizando que ningún clúster supere `maxClust` y que se cubran las $n$ dimensiones sin pérdida ni duplicidad.
* **Cobertura Alcanzada:** **100%** (37/37 sentencias).
* **Comando de Ejecución Aislada:** `pytest tests/unit/test_corclust.py -v`

### Matriz de Casos de Prueba (`PU-03`)

| ID Caso | Método / Función Bajo Prueba | Entrada de Prueba | Resultado Esperado / Invariante Matemático | Estado |
| :--- | :--- | :--- | :--- | :---: |
| **`UT-CC-001`** | `corClust.__init__(n=8)` | `n=8` | Vectores `c, c_r, c_rs` de forma `(8,)`, matriz `C` de forma `(8, 8)` en ceros y `N == 0`. | ✅ PASS |
| **`UT-CC-002`** | `corClust.update(x)` | `x1=[1, 2, 3]`, `x2=[2, 4, 6]` | En $N=1$, residuo $c_{rt} = 0$; en $N=2$, diagonal de $C > 0$ y simetría exacta $C == C^\top$. | ✅ PASS |
| **`UT-CC-003`** | `corClust.corrDist()` | 10 vectores en $\mathbb{R}^4$ con 2 features constantes (`0.0` y `5.0`) | Protección `C_rs_sqrt == 0 -> 1e-100` y `D[D<0] = 0`; matriz $D \ge 0$ libre de `NaN` e `Inf`. | ✅ PASS |
| **`UT-CC-004`** | `corClust.cluster(maxClust)` | 40 muestras en $\mathbb{R}^{15}$, `maxClust=4` | Todos los subgrupos satisfacen $1 \le |v_i| \le 4$ y la unión disjunta de índices es exactamente `{0, 1, ..., 14}`. | ✅ PASS |
| **`UT-CC-005`** | `corClust.cluster()` (Fronteras) | `n=6`, con `maxClust=0` y `maxClust=100` | Para `maxClust=0` clampa a $1$ generando 6 singletons; para `maxClust=100` clampa a $n=6$ generando 1 único clúster. | ✅ PASS |

---

## 4. Plan de Pruebas Unitarias `PU-04` — Módulo [`KitNET/KitNET.py`](../KitNET/KitNET.py)

* **Identificador del Plan:** `PU-04-KITNET`
* **Archivo de Prueba:** [`tests/unit/test_kitnet.py`](../tests/unit/test_kitnet.py)
* **Documento Individual:** [`docs/planes_unitarios/PU_04_KitNET_Orquestador.md`](planes_unitarios/PU_04_KitNET_Orquestador.md)
* **Objetivo Técnico:** Validar la máquina de estados del detector neuronal (`FM train-mode` $\to$ `AD train-mode` $\to$ `AD execute-mode`), el manejo de parámetros por defecto, la protección contra ejecución sin mapeo de características (`RuntimeError`), la construcción de `ensembleLayer` + `outputLayer` (`__createAD__`) y la propagación de errores de reconstrucción.
* **Cobertura Alcanzada:** **100%** (61/61 sentencias).
* **Comando de Ejecución Aislada:** `pytest tests/unit/test_kitnet.py -v`

### Matriz de Casos de Prueba (`PU-04`)

| ID Caso | Método / Función Bajo Prueba | Entrada de Prueba | Resultado Esperado / Invariante Matemático | Estado |
| :--- | :--- | :--- | :--- | :---: |
| **`UT-KN-001`** | `KitNET.__init__()` | `n=10, max_autoencoder_size=0, FM_grace_period=None, AD_grace_period=500` | `FM_grace_period` hereda `500`, `m` se ajusta a `1` y `v` permanece `None`. | ✅ PASS |
| **`UT-KN-002`** | `KitNET.execute(x)` | Llamada directa a `execute(x)` con `v=None` | Lanza `RuntimeError` indicando que el mapeo de características aún no ha sido aprendido ni suministrado. | ✅ PASS |
| **`UT-KN-003`** | `KitNET.__init__(feature_map=...)` y `process(x)` | `feature_map=[[0,1,2],[3,4],[5]]`, `FM_grace=0, AD_grace=5` | Omite fase FM, crea 3 autoencoders en `ensembleLayer` y 1 en `outputLayer` (`n_visible=3`), entrena 6 pasos retornando `0.0` y ejecuta en el paso 7 retornando $\text{RMSE} > 0$. | ✅ PASS |
| **`UT-KN-004`** | `KitNET.process(x)` (Ciclo completo) | `n=12, m=4, FM_grace=10, AD_grace=15` | Recorre exactamente `11` pasos en fase FM (instanciando `v` en el paso 11), `15` pasos en fase AD (retornando `0.0`) y emite $\text{RMSE} > 0$ a partir del paso `27`. | ✅ PASS |

---

## 5. Plan de Pruebas Unitarias `PU-05` — Módulo [`AfterImage.py`](../AfterImage.py)

* **Identificador del Plan:** `PU-05-AFTERIMAGE`
* **Archivo de Prueba:** [`tests/unit/test_afterimage.py`](../tests/unit/test_afterimage.py)
* **Documento Individual:** [`docs/planes_unitarios/PU_05_AfterImage.md`](planes_unitarios/PU_05_AfterImage.md)
* **Objetivo Técnico:** Verificar la precisión matemática de las estadísticas incrementales amortiguadas 1D (`incStat`), el modo diferencial de jitter (`isTypeDiff`), las estadísticas covariantes 2D (`incStat_cov`), y las operaciones de registro, consulta, límite de memoria (`LookupError`) y limpieza de registros antiguos (`cleanOutOldRecords`) en `incStatDB`.
* **Cobertura Alcanzada:** **98%** (284/291 sentencias).
* **Comando de Ejecución Aislada:** `pytest tests/unit/test_afterimage.py -v`

### Matriz de Casos de Prueba (`PU-05`)

| ID Caso | Método / Función Bajo Prueba | Entrada de Prueba | Resultado Esperado / Invariante Matemático | Estado |
| :--- | :--- | :--- | :--- | :---: |
| **`UT-AI-001`** | `incStat.insert()`, `processDecay()`, `weight()`, `mean()`, `var()`, `std()` | `Lambda=1.0`, inserciones `(v=10, t=0)` y `(v=20, t=1)` | En $t=1$, $\gamma = 2^{-1} = 0.5$, peso $w = 1.5$, $CF_1 = 25.0$, $\mu = 16.6667$, $\sigma = \sqrt{\text{var}}$. | ✅ PASS |
| **`UT-AI-002`** | `incStat.insert()` (`isTypeDiff=True`) | `init_time=10.0`, llegadas en `t=10.0` y `t=12.0` | Ignora el valor `v` nominal y registra $\Delta t = 0$ en $t=10$ y $\Delta t = 2.0$ en $t=12$. | ✅ PASS |
| **`UT-AI-003`** | `incStat.magnitude()` y `radius()` | Flujos `s1(v=3)` y `s2(v=4)` en $t=0$ | Magnitud euclidiana de medias $\sqrt{3^2 + 4^2} = 5.0$ y radio $0.0$ para varianzas nulas. | ✅ PASS |
| **`UT-AI-004`** | `incStat.getHeaders_1D()` y `getHeaders_2D()` | `ID="host1"`, `ID2="host2"`, `suffix=True/False` | Genera 3 encabezados 1D (`weight, mean, std`) y 7 encabezados combinados 1D+2D. | ✅ PASS |
| **`UT-AI-005`** | `incStat_cov.update_cov()`, `cov()`, `pcc()`, `get_stats1..4()`, `getHeaders(1..5)` | 7 actualizaciones acopladas entre flujos `A` y `B` | Calcula `CF3`, covarianza, PCC y vectores de longitud `2, 4, 8, 10` sin valores `NaN`, más delegación desde `incStat`. | ✅ PASS |
| **`UT-AI-006`** | `incStatDB.register()` | `limit=2`, inserción de 3 claves distintas | Las dos primeras claves e inserciones repetidas tienen éxito; la tercera clave distinta lanza `LookupError`. | ✅ PASS |
| **`UT-AI-007`** | `incStatDB.get_1D_Stats()`, `get_2D_Stats()`, `get_all_2D_Stats()` | Claves inexistentes en la tabla hash | Retorna `[np.nan]*3`, `[np.nan]*2` y `([], [])` sin lanzar `AttributeError`. | ✅ PASS |
| **`UT-AI-008`** | `incStatDB.update_get_1D2D_Stats()`, `get_nD_Stats()`, `cleanOutOldRecords()` | Dos hosts en $t=1.0$ y limpieza en $t=20.0$ con `cutoffWeight=0.01` | Vector 1D+2D de 7 elementos, consultas existentes 1D/2D/nD válidas y eliminación de los 2 registros caducados (`len(HT) == 0`). | ✅ PASS |

---

## 6. Plan de Pruebas Unitarias `PU-06` — Módulo [`netStat.py`](../netStat.py)

* **Identificador del Plan:** `PU-06-NETSTAT`
* **Archivo de Prueba:** [`tests/unit/test_netstat.py`](../tests/unit/test_netstat.py)
* **Documento Individual:** [`docs/planes_unitarios/PU_06_netStat.md`](planes_unitarios/PU_06_netStat.md)
* **Objetivo Técnico:** Verificar la configuración de las 4 tablas hash (`HT_MI`, `HT_H`, `HT_jit`, `HT_Hp`), la aceptación de listas personalizadas de decaimiento $\Lambda$, la identificación de subredes (`findDirection`) y la extracción del vector concatenado de 100 características por paquete (`updateGetStats`) tanto para tráfico IP/TCP/UDP como para tramas ARP.
* **Cobertura Alcanzada:** **100%** (59/59 sentencias).
* **Comando de Ejecución Aislada:** `pytest tests/unit/test_netstat.py -v`

### Matriz de Casos de Prueba (`PU-06`)

| ID Caso | Método / Función Bajo Prueba | Entrada de Prueba | Resultado Esperado / Invariante Matemático | Estado |
| :--- | :--- | :--- | :--- | :---: |
| **`UT-NS-001`** | `netStat.__init__()` y `getNetStatHeaders()` | Por defecto (`Lambdas=np.nan`) y personalizado (`Lambdas=[1.0, 0.1]`) | Por defecto crea 5 lambdas y 100 encabezados ($20 \times 5$); con 2 lambdas genera 40 encabezados ($20 \times 2$). | ✅ PASS |
| **`UT-NS-002`** | `netStat.findDirection()` | Direcciones IPv4 (`IPtype=0`), IPv6 (`IPtype=1`) y L2 (`IPtype=np.nan`) | Extrae prefijos `/24` en IPv4 (`192.168.1`), mitad de cadena en IPv6 y direcciones MAC completas en capa 2. | ✅ PASS |
| **`UT-NS-003`** | `netStat.updateGetStats()` (Tráfico TCP/UDP) | Par de paquetes bidireccionales entre `192.168.1.10:443` y `192.168.1.20:51234` | Retorna `np.ndarray` de forma `(100,)` completamente finito (sin `NaN` ni `Inf`). | ✅ PASS |
| **`UT-NS-004`** | `netStat.updateGetStats()` (Rama ARP) | Paquete con `srcProtocol="arp"`, `dstProtocol="arp"` | Retorna vector `(100,)` finito y registra las claves de sesión en `HT_Hp` usando las direcciones MAC. | ✅ PASS |

---

## 7. Plan de Pruebas Unitarias `PU-07` — Módulo [`FeatureExtractor.py`](../FeatureExtractor.py)

* **Identificador del Plan:** `PU-07-FE`
* **Archivo de Prueba:** [`tests/unit/test_feature_extractor.py`](../tests/unit/test_feature_extractor.py)
* **Documento Individual:** [`docs/planes_unitarios/PU_07_FeatureExtractor.md`](planes_unitarios/PU_07_FeatureExtractor.md)
* **Objetivo Técnico:** Verificar la validación de archivos de entrada, el parseo línea por línea de archivos `.tsv` cubriendo todas las ramas de protocolo de red (IPv4/TCP, IPv4/UDP, IPv6/TCP, ARP, ICMP y L2 raw MAC), el respeto estricto del parámetro `limit`, el cierre del descriptor de archivo al terminar y el mecanismo de fallback a `Scapy` sobre archivos `.pcap`.
* **Cobertura Alcanzada:** **83%** (139/167 sentencias; las líneas restantes corresponden a invocación de binario externo `tshark.exe` del sistema operativo).
* **Comando de Ejecución Aislada:** `pytest tests/unit/test_feature_extractor.py -v`

### Matriz de Casos de Prueba (`PU-07`)

| ID Caso | Método / Función Bajo Prueba | Entrada de Prueba | Resultado Esperado / Invariante Matemático | Estado |
| :--- | :--- | :--- | :--- | :---: |
| **`UT-FE-001`** | `FE.__init__()` / `__prep__()` | Ruta inexistente `non_existent.pcap` y archivo con extensión `.txt` | Lanza `Exception` en ambos casos impidiendo inicializar un lector inválido. | ✅ PASS |
| **`UT-FE-002`** | `FE.get_next_vector()` (Modo TSV Multiprotocolo) | Archivo `.tsv` sintético con 6 paquetes (IPv4/TCP, IPv4/UDP, IPv6/TCP, ARP, ICMP, L2 MAC) | Extrae 6 vectores de forma `(100,)` sin `NaN`; en la 7ª llamada cierra `tsvinf` y retorna `[]`. | ✅ PASS |
| **`UT-FE-003`** | `FE.get_next_vector()` (Truncamiento por `limit`) | Archivo `.tsv` con 5 paquetes y parámetro `limit=2` | Ajusta `extractor.limit == 2`, procesa exactamente 2 vectores y en la 3ª llamada retorna `[]`. | ✅ PASS |
| **`UT-FE-004`** | `FE.get_next_vector()` (Modo Fallback Scapy `.pcap`) | Archivo `.pcap` generado con Scapy (5 paquetes: TCP, UDP, ARP, ICMP, L2 Ether) con `_get_tshark_path` mockeado a `""` | Selecciona `parse_type == "scapy"`, extrae 5 vectores `(100,)` válidos y retorna `[]` al finalizar. | ✅ PASS |

---

## 8. Plan de Pruebas Unitarias `PU-08` — Módulo [`Kitsune.py`](../Kitsune.py)

* **Identificador del Plan:** `PU-08-KITSUNE`
* **Archivo de Prueba:** [`tests/unit/test_kitsune.py`](../tests/unit/test_kitsune.py)
* **Documento Individual:** [`docs/planes_unitarios/PU_08_Kitsune_Facade.md`](planes_unitarios/PU_08_Kitsune_Facade.md)
* **Objetivo Técnico:** Verificar la integración unitaria en la fachada `Kitsune` entre `FE` y `KitNET`, comprobando el cableado de hiperparámetros en el constructor y el ciclo de vida completo de `proc_next_packet()` desde la fase de mapeo y entrenamiento (`RMSE == 0.0`), pasando por la fase de detección (`RMSE > 0.0`), hasta el retorno del centinela `-1` al agotarse los paquetes.
* **Cobertura Alcanzada:** **100%** (11/11 sentencias).
* **Comando de Ejecución Aislada:** `pytest tests/unit/test_kitsune.py -v`

### Matriz de Casos de Prueba (`PU-08`)

| ID Caso | Método / Función Bajo Prueba | Entrada de Prueba | Resultado Esperado / Invariante Matemático | Estado |
| :--- | :--- | :--- | :--- | :---: |
| **`UT-KIT-001`** | `Kitsune.__init__()` | Archivo `.tsv`, `limit=10, max_autoencoder_size=8, FM_grace_period=5, AD_grace_period=10` | Inicializa `FE` con `100` features y configura `AnomDetector` (`n=100, m=8, FM_grace=5, AD_grace=10`). | ✅ PASS |
| **`UT-KIT-002`** | `Kitsune.proc_next_packet()` | Archivo `.tsv` de 15 paquetes con `FM_grace=4, AD_grace=5` | Primeros 10 paquetes retornan `0.0` (entrenamiento); paquetes 11 a 15 retornan $\text{RMSE} > 0.0$ (ejecución); paquete 16 retorna `-1` (EOF). | ✅ PASS |

---

## 9. Resumen Consolidado de Ejecución y Cobertura Unitaria

| Plan Unitario | Módulo Bajo Prueba | Archivo de Test | Casos Ejecutados | Aprobados | Cobertura de Líneas |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **`PU-01`** | `KitNET/utils.py` | `tests/unit/test_utils.py` | 9 | 9 (100%) | **100%** |
| **`PU-02`** | `KitNET/dA.py` | `tests/unit/test_da.py` | 6 | 6 (100%) | **100%** |
| **`PU-03`** | `KitNET/corClust.py` | `tests/unit/test_corclust.py` | 5 | 5 (100%) | **100%** |
| **`PU-04`** | `KitNET/KitNET.py` | `tests/unit/test_kitnet.py` | 4 | 4 (100%) | **100%** |
| **`PU-05`** | `AfterImage.py` | `tests/unit/test_afterimage.py` | 8 | 8 (100%) | **98%** |
| **`PU-06`** | `netStat.py` | `tests/unit/test_netstat.py` | 4 | 4 (100%) | **100%** |
| **`PU-07`** | `FeatureExtractor.py` | `tests/unit/test_feature_extractor.py` | 4 | 4 (100%) | **83%** |
| **`PU-08`** | `Kitsune.py` | `tests/unit/test_kitsune.py` | 2 | 2 (100%) | **100%** |
| **TOTAL** | **8 Módulos Core** | **8 Suites Unitarias** | **42** | **42 (100%)** | **95.19%** |
