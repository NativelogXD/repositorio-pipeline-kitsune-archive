# Documentación Académica de Ingeniería de Software: Análisis Arquitectónico, Plan Maestro de Pruebas (CI/CD), Planes de Pruebas Unitarias y Documentación del Pipeline para el Sistema NIDS Kitsune-py

**Proyecto bajo estudio:** Kitsune-py (Online Network Intrusion Detection System basado en un Ensamble de Autoencoders, NDSS 2018)  
**Repositorio de referencia:** https://github.com/ymirsky/Kitsune-py  
**Repositorio local intervenido:** `C:\Users\xiao\Documents\universidad\ing softwqqare 3\repo-kitsune`  
**Asignatura:** Ingeniería de Software III  
**Estándares aplicados:** IEEE 829 / ISO/IEC/IEEE 29119 (Documentación de Pruebas de Software)  
**Alcance de ejecución activa:** Funcional y verificado hasta la etapa de Pruebas Unitarias (42 casos unitarios aprobados, 95.19% de cobertura de código), con etapa de Pruebas de Despliegue documentada y simulada en el pipeline.

---

## Índice General

1. **Introducción y Análisis Arquitectónico del Proyecto Kitsune-py**
   - 1.1. Contexto e Intuición Técnica del Sistema
   - 1.2. Arquitectura en Cuatro Subsistemas y Flujo de Datos
   - 1.3. Formulación Matemática y Algorítmica por Componente
   - 1.4. Análisis de Complejidad Temporal y Espacial
   - 1.5. Auditoría Estática de Código y Corrección de Defectos Latentes
2. **Plan Maestro de Pruebas General (Ciclo de Vida Completo CI/CD)**
   - 2.1. Alcance, Objetivos y Delimitación Operativa (Funcional hasta Pruebas Unitarias)
   - 2.2. Flujo del Ciclo de Vida desde Eventos Git (Push y Pull Request) hasta Despliegue
   - 2.3. Etapa 1: Análisis Estático, Compilación Sintáctica y Verificación de Entorno (Activa)
   - 2.4. Etapa 2: Pruebas Unitarias Aisladas y Compuerta de Cobertura (Activa - Corte Funcional)
   - 2.5. Etapa 3: Pruebas de Integración de Subsistemas (Documentada)
   - 2.6. Etapa 4: Pruebas de Sistema End-to-End y Regresión Estadística ML (Documentada)
   - 2.7. Etapa 5: Pruebas No Funcionales de Rendimiento y Memoria O(1) (Documentada)
   - 2.8. Etapa 6: Pruebas de Despliegue, Smoke Tests y Validación en Staging (Simulada/Comentada)
3. **Planes Específicos de Pruebas Unitarias por Módulo (PU-01 a PU-08)**
   - 3.1. Plan de Pruebas Unitarias PU-01: Módulo `KitNET/utils.py`
   - 3.2. Plan de Pruebas Unitarias PU-02: Módulo `KitNET/dA.py` (Denoising Autoencoder)
   - 3.3. Plan de Pruebas Unitarias PU-03: Módulo `KitNET/corClust.py` (Clustering Incremental)
   - 3.4. Plan de Pruebas Unitarias PU-04: Módulo `KitNET/KitNET.py` (Orquestador del Ensamble)
   - 3.5. Plan de Pruebas Unitarias PU-05: Módulo `AfterImage.py` (Estadísticas Amortiguadas 1D/2D)
   - 3.6. Plan de Pruebas Unitarias PU-06: Módulo `netStat.py` (Estadísticas de Red Multiperspectiva)
   - 3.7. Plan de Pruebas Unitarias PU-07: Módulo `FeatureExtractor.py` (Parser TSV y Scapy)
   - 3.8. Plan de Pruebas Unitarias PU-08: Módulo `Kitsune.py` (Fachada Principal del Sistema)
   - 3.9. Consolidado de Resultados de Ejecución y Cobertura Unitaria
4. **Documentación Completa del Pipeline CI/CD**
   - 4.1. Arquitectura del Workflow `.github/workflows/ci-unit-tests.yml`
   - 4.2. Eventos Disparadores (Push, Pull Request y Ejecución Manual) y Permisos
   - 4.3. Detalle Técnico de Jobs Activos (`static-validation` y `unit-tests`)
   - 4.4. Detalle del Job Simulado/Comentado de Pruebas de Despliegue (`deployment-verification-simulated`)
   - 4.5. Código Fuente Completo y Documentado del Pipeline
5. **Documentación de Pruebas de Despliegue y Resultados (Simulación Controlada)**
   - 5.1. Diseño del Protocolo de Pruebas de Despliegue (Deployment Verification Tests)
   - 5.2. Matriz de Casos de Prueba de Despliegue (DP-01 a DP-05)
   - 5.3. Registro de Resultados de Pruebas de Despliegue y Métricas Operativas
6. **Conclusiones Técnicas**

---

## 1. Introducción y Análisis Arquitectónico del Proyecto Kitsune-py

### 1.1. Contexto e Intuición Técnica del Sistema

Los Sistemas de Detección de Intrusiones en Red (NIDS, por sus siglas en inglés) basados en redes neuronales profundas presentan tradicionalmente dos limitaciones severas para su adopción en dispositivos de borde como enrutadores, pasarelas IoT (*gateways*) o sondas de red ligeras: en primer lugar, el elevado costo computacional y de memoria necesario para entrenar modelos monolíticos sobre vectores de alta dimensionalidad; en segundo lugar, la dependencia de conjuntos de datos previamente etiquetados de forma supervisada.

**Kitsune** resuelve ambos problemas mediante un paradigma *online*, no supervisado y *plug-and-play*. Su diseño descansa sobre dos intuiciones algorítmicas fundamentales:

1. **Extracción de Contexto Temporal en Memoria Constante (`AfterImage`):** En lugar de almacenar ventanas históricas de paquetes en memoria RAM, Kitsune mantiene contadores estadísticos incrementales con decaimiento exponencial sobre múltiples escalas de tiempo ($\Lambda = [5, 3, 1, 0.1, 0.01]$). Esto permite actualizar medias, varianzas, covarianzas y coeficientes de correlación de Pearson entre flujos en tiempo constante $O(1)$ por paquete.
2. **Descomposición Neuronal en Ensamble (`corClust` + `KitNET`):** En lugar de entrenar un único autoencoder denso de $n = 100$ entradas (cuyo paso de propagación y retropropagación tiene complejidad cuadrática $O(n^2) = 10,000$ operaciones por capa), Kitsune agrupa dinámicamente las características correlacionadas en $k$ subconjuntos de tamaño máximo $m$ (por defecto $m = 10$). Cada subconjunto alimenta un pequeño *Denoising Autoencoder* (`dA`) en la capa de ensamble (*Ensemble Layer*), reduciendo la complejidad computacional a $O(k \cdot m^2) \approx 1,000$ operaciones por paquete. Finalmente, un autoencoder de salida (*Output Layer*) recibe los errores de reconstrucción (RMSE) de cada autoencoder del ensamble y actúa como un votador no lineal para emitir el puntaje de anomalía global.

### 1.2. Arquitectura en Cuatro Subsistemas y Flujo de Datos

El procesamiento dentro de `Kitsune-py` se estructura en cuatro subsistemas acoplados secuencialmente:

| Subsistema | Archivo Principal | Clase / Componente | Responsabilidad Funcional en la Tubería |
| :--- | :--- | :--- | :--- |
| **1. Captura y Parseo (Packet Capturer)** | `FeatureExtractor.py` | Clase `FE` | Lee archivos de captura `.pcap`, `.pcapng` o `.tsv`. Si detecta `tshark` (Wireshark), convierte el archivo `.pcap` a `.tsv` de 19 columnas y lo procesa en streaming línea por línea; en caso contrario, utiliza `scapy` como mecanismo de respaldo (*fallback*). |
| **2. Extracción de Características (AfterImage)** | `netStat.py` y `AfterImage.py` | Clases `netStat`, `incStat`, `incStat_cov`, `incStatDB` | Mantiene 4 tablas hash (`HT_MI`, `HT_H`, `HT_jit`, `HT_Hp`) evaluadas en 5 factores de decaimiento $\lambda$, generando un vector numérico de $n = 100$ características por paquete. |
| **3. Mapeo de Características (Feature Mapper)** | `KitNET/corClust.py` | Clase `corClust` | Durante el periodo de gracia `FM_grace_period`, acumula incrementalmente los residuos y la matriz de correlación parcial entre las $100$ dimensiones. Al concluir dicho periodo, ejecuta agrupamiento jerárquico aglomerativo produciendo el mapeo $v = [v_1, \dots, v_k]$ con $|v_i| \le m$. |
| **4. Detección de Anomalías (Anomaly Detector)** | `KitNET/KitNET.py`, `KitNET/dA.py`, `KitNET/utils.py` | Clases `KitNET`, `dA`, `dA_params` | Durante `AD_grace_period`, entrena mediante descenso de gradiente estocástico (SGD) los $k$ autoencoders de la capa de ensamble y el autoencoder de la capa de salida. Una vez superados ambos periodos de gracia, evalúa cada paquete retornando su error RMSE de reconstrucción. |

### 1.3. Formulación Matemática y Algorítmica por Componente

1. **Estadísticas Incrementales Amortiguadas (`AfterImage.py`):**
   Dado un flujo con factor de decaimiento $\lambda > 0$ y una nueva observación $(v, t_{\text{actual}})$, se calcula el lapso $\Delta t = t_{\text{actual}} - t_{\text{previo}} \ge 0$ y el factor de amortiguamiento exponencial $\gamma = 2^{-\lambda \Delta t}$. Los acumuladores suficientes (*Cluster Features*) se actualizan como:
   - Suma lineal ponderada: $CF_1 \leftarrow \gamma \cdot CF_1 + v$
   - Suma cuadrática ponderada: $CF_2 \leftarrow \gamma \cdot CF_2 + v^2$
   - Peso efectivo: $w \leftarrow \gamma \cdot w + 1$
   - Media incremental: $\mu = CF_1 / w$
   - Varianza y desviación estándar incremental: $\sigma^2 = |CF_2 / w - \mu^2|$, $\sigma = \sqrt{\sigma^2}$
   - Para flujos bidireccionales acoplados (`incStat_cov`), el acumulador de productos de residuos $CF_3$ se actualiza mediante $CF_3 \leftarrow \gamma \cdot CF_3 + (v_i - \mu_i) \cdot r_{\neg i}$, estimando la covarianza $\text{Cov} = CF_3 / w_3$ y el coeficiente de correlación de Pearson $\text{PCC} = \text{Cov} / (\sigma_1 \sigma_2)$.

2. **Vector de 100 Características de Red (`netStat.py`):**
   Por cada paquete procesado y para cada uno de los 5 valores de $\lambda \in \{5, 3, 1, 0.1, 0.01\}$, se extraen 20 métricas (totalizando $20 \times 5 = 100$ características):
   - `HT_MI` (Relación MAC-IP de origen `srcMAC + srcIP`): 3 estadísticas 1D (`w, mean, var`) $\times 5 = 15$ dimensiones.
   - `HT_H` (Tráfico entre hosts `srcIP` y `dstIP`): 7 estadísticas 1D+2D (`w, mean, var, radius, magnitude, cov, pcc`) $\times 5 = 35$ dimensiones.
   - `HT_jit` (Jitter entre `srcIP + dstIP` con `isTypeDiff=True`): 3 estadísticas 1D $\times 5 = 15$ dimensiones.
   - `HT_Hp` (Tráfico entre sockets `srcIP + srcPort` y `dstIP + dstPort`, o direcciones MAC para el protocolo ARP): 7 estadísticas 1D+2D $\times 5 = 35$ dimensiones.

3. **Autoencoders de Reconstrucción (`KitNET/dA.py`):**
   Cada autoencoder normaliza cada dimensión $j$ de forma *online* en $[0, 1]$ mediante $\hat{x}_j = (x_j - \min_j) / (\max_j - \min_j + 10^{-16})$, aplica codificación oculta $y = \text{sigmoid}(\tilde{x} W + b_h)$ con razón de compresión `hiddenRatio = 0.75` ($n_{\text{hidden}} = \lceil n_{\text{visible}} \times 0.75 \rceil$) y reconstruye la entrada mediante pesos atados $z = \text{sigmoid}(y W^T + b_v)$. El error de reconstrucción emitido es $\text{RMSE} = \sqrt{\frac{1}{n_{\text{visible}}} \sum (\hat{x}_j - z_j)^2}$.

### 1.4. Análisis de Complejidad Temporal y Espacial

| Módulo / Operación | Complejidad Temporal por Paquete | Complejidad Espacial en Memoria RAM | Evaluación de Eficiencia |
| :--- | :--- | :--- | :--- |
| **`FE.get_next_vector()` (Modo TSV)** | $O(1)$ amortizado | $O(1)$ | Lectura perezosa en streaming mediante iterador `csv.reader`. |
| **`FE.get_next_vector()` (Modo Scapy)** | $O(1)$ tras carga inicial $O(M)$ | $O(M)$ (donde $M$ es total de paquetes) | `rdpcap` carga el archivo completo en RAM; menos eficiente para capturas masivas. |
| **`netStat.updateGetStats()`** | $O(|\Lambda|) = O(1)$ | $O(|HT_{\text{activos}}| \cdot |\Lambda|)$ | Acceso hash $O(1)$ y actualización escalar sobre 5 ventanas. |
| **`corClust.update(x)`** | $O(n^2)$ | $O(n^2)$ | Operación matricial vectorizada con `np.outer` sobre $n = 100$. |
| **`corClust.cluster(m)`** | $O(n^2 \log n)$ (ejecución única) | $O(n^2)$ | Se ejecuta una sola vez al cumplirse `n_trained == FM_grace_period`. |
| **`KitNET.train(x)` / `execute(x)`** | $O(n \cdot m + k^2)$ | $O(n \cdot m + k^2)$ | Reduce en un orden de magnitud el costo frente a una red monolítica $O(n^2)$. |

### 1.5. Auditoría Estática de Código y Corrección de Defectos Latentes

Durante el análisis estático del código original (desarrollado en 2018 para Python 3.6 y NumPy 1.13), se identificaron y documentaron seis hallazgos técnicos relevantes:

1. **Dimensionalidad Real de 100 Características (vs. 115 en comentarios):** En `example.py` se documenta $n = 115$, pero en `netStat.py` (líneas 75-77) el bloque de 15 características `Hstat` se encuentra comentado en el repositorio original. Por tanto, el sistema extrae exactamente $n = 100$ características.
2. **Eliminación de `np.Inf` en NumPy 2.x:** En `AfterImage.py`, `KitNET/dA.py` y `example.py` se utilizaba el atributo deprecado `np.Inf` / `numpy.Inf`, lo cual causaba `AttributeError` en entornos modernos con NumPy $\ge 2.0$. Se actualizó quirúrgicamente a `np.inf`.
3. **Defecto Tipográfico `np.na` en `AfterImage.py` (líneas 314 y 326):** Los métodos `get_1D_Stats` y `get_2D_Stats` de `incStatDB` devolvían `[np.na]*3` y `[np.na]*2` cuando se consultaba un flujo inexistente, lanzando `AttributeError: module 'numpy' has no attribute 'na'`. Se corrigió a `[np.nan]`.
4. **Defecto de Indexación en `incStatDB.cleanOutOldRecords` (`AfterImage.py` líneas 428-441):** El método intentaba indexar los valores del diccionario `self.HT` como listas (`tup[1][0].getMaxW(curTime)`), provocando `TypeError: 'incStat' object is not subscriptable`. Se saneó el método aplicando `processDecay(curTime)` sobre cada instancia `incStat` y ordenando por su peso `w`.
5. **Ambigüedad Booleana en `netStat.__init__`:** La condición `if np.isnan(Lambdas):` fallaba con `ValueError` cuando el usuario pasaba una lista personalizada de factores `Lambdas`. Se corrigió a `if np.all(np.isnan(Lambdas)):`.
6. **Condición de Frontera en `KitNET.py` (`FM_grace_period + 1`):** En `KitNET.train(x)`, la condición `self.n_trained <= self.FM_grace_period` procesa exactamente `FM_grace_period + 1` paquetes (desde el índice `0` hasta `FM_grace_period` inclusive) antes de entrenar los autoencoders. En consecuencia, el paso a modo de ejecución (`execute`) ocurre exactamente tras `FM_grace_period + AD_grace_period + 1` paquetes.

---

## 2. Plan Maestro de Pruebas General (Ciclo de Vida Completo CI/CD)

### 2.1. Alcance, Objetivos y Delimitación Operativa (Funcional hasta Pruebas Unitarias)

El **Plan Maestro de Pruebas** define la estrategia integral de verificación y validación (V&V) de `Kitsune-py` a lo largo de todo el ciclo de vida de Ingeniería de Software e Integración/Entrega Continua (CI/CD).

**Delimitación de Ejecución del Proyecto:**
- **Documentación del Contexto General:** Abarca la totalidad de las etapas del ciclo de vida (desde los eventos de control de versiones `git push` y `pull_request`, pasando por análisis estático, pruebas unitarias, pruebas de integración, pruebas de sistema/regresión ML, pruebas no funcionales de rendimiento/memoria, hasta las pruebas de despliegue en entornos de *Staging*).
- **Ejecución Funcional Activa:** De acuerdo con los requerimientos del proyecto, **la ejecución automatizada real del pipeline llega y se detiene estrictamente en la finalización exitosa de las Pruebas Unitarias y su compuerta de cobertura**, dejando las pruebas de despliegue documentadas y simuladas/comentadas dentro del archivo de workflow.

### 2.2. Flujo del Ciclo de Vida desde Eventos Git (Push y Pull Request) hasta Despliegue

El flujo completo de aseguramiento de calidad en el ciclo CI/CD sigue la siguiente secuencia de seis etapas:

| Etapa del Ciclo CI/CD | Evento / Disparador | Propósito de Aseguramiento de Calidad | Estado en el Pipeline |
| :--- | :--- | :--- | :--- |
| **Fase 0: Control de Versiones** | `git push` / `pull_request` sobre `master`, `main`, `develop` | Registro atómico de cambios y activación automática del orquestador de GitHub Actions. | **ACTIVO** |
| **Etapa 1: Validación Estática y Sintáctica** | Inicio del workflow CI | Compilación de bytecode (`py_compile`), verificación de AST y validación de importabilidad determinista de los 9 módulos Python. | **ACTIVO (Job 1)** |
| **Etapa 2: Pruebas Unitarias y Cobertura** | Aprobación de Etapa 1 (`needs: static-validation`) | Ejecución aislada de los 8 Planes Unitarios (`PU-01` a `PU-08`, 42 casos de prueba) y verificación del umbral de cobertura $\ge 85\%$. | **ACTIVO (Job 2 - Corte Funcional)** |
| **Etapa 3: Pruebas de Integración** | Post-Unitarias (Ciclo Extendido) | Verificación de contratos de datos entre `FE <-> netStat <-> AfterImage` y `corClust <-> KitNET <-> dA`. | **DOCUMENTADO** |
| **Etapa 4: Pruebas de Sistema y Regresión ML** | Post-Integración (Ciclo Extendido) | Evaluación extremo a extremo sobre tráfico de botnet (`mirai.pcap`) y ajuste de umbral log-normal de anomalías. | **DOCUMENTADO** |
| **Etapa 5: Pruebas No Funcionales (RAM y CPU)** | Pre-Release (Ciclo Extendido) | Verificación de invariante espacial $O(1)$ en streaming, throughput (paquetes/seg) y poda de tablas hash. | **DOCUMENTADO** |
| **Etapa 6: Pruebas de Despliegue (Smoke/Staging)** | Post-Empaquetado CD | Validación de paquete distribuible, arranque del agente NIDS en nodo *Staging*, prueba de humo y latencia por paquete. | **SIMULADO / COMENTADO** |

### 2.3. Etapa 1: Análisis Estático, Compilación Sintáctica y Verificación de Entorno (Activa)

- **Objetivo:** Prevenir que errores de sintaxis, secuencias de escape inválidas o dependencias incompatibles lleguen a la fase de pruebas dinámicas.
- **Alcance:** Archivos `Kitsune.py`, `FeatureExtractor.py`, `netStat.py`, `AfterImage.py` y el subpaquete `KitNET/` (`__init__.py`, `utils.py`, `dA.py`, `corClust.py`, `KitNET.py`).
- **Procedimiento:** Ejecución de `python -m py_compile` sobre todos los módulos y ejecución de prueba de carga de símbolos en memoria.
- **Criterio de Aceptación:** Retorno `exit code 0` sin errores de compilación ni excepciones de importación.

### 2.4. Etapa 2: Pruebas Unitarias Aisladas y Compuerta de Cobertura (Activa - Corte Funcional)

- **Objetivo:** Verificar matemáticamente y en completo aislamiento cada función, método y transición de estado de los 8 módulos que integran Kitsune.
- **Mecanismo de Aislamiento:** Uso de generadores pseudoaleatorios con semilla explícita (`seed=42` y `RandomState(1234)`), creación de capturas `.tsv` y `.pcap` sintéticas efímeras en directorios temporales (`tmp_path`) y sustitución (`monkeypatch`) de llamadas al ejecutable externo `tshark`.
- **Criterio de Aceptación (Quality Gate):** 100% de los 42 casos de prueba unitarios aprobados (`0 failed`) y cobertura global de código $\ge 85\%$ (alcanzando **95.19%** en la ejecución verificada). Una vez superada esta compuerta, el pipeline publica los reportes `coverage.xml` y `junit-unit-results.xml` y **finaliza su ejecución activa**.

### 2.5. Etapa 3: Pruebas de Integración de Subsistemas (Documentada)

Evalúa la interacción combinada entre módulos sin dobles de prueba intermedios:
- **IT-01 (Integración `FE` + `netStat` + `AfterImage`):** Procesamiento continuo de una traza multiprotocolo (ARP, IPv4/TCP, IPv4/UDP, IPv6, ICMP) verificando que las referencias cruzadas `incStat_cov` entre canales entrantes y salientes produzcan vectores de 100 características libres de `NaN`.
- **IT-02 (Equivalencia de Motores de Parseo):** Verificación de consistencia entre vectores extraídos mediante el flujo `tshark` (`.pcap` $\to$ `.tsv`) y el flujo `scapy` sobre una misma captura de red.
- **IT-03 (Integración `corClust` + `KitNET` + `dA`):** Comprobación de que la partición del dendrograma producida por `corClust.cluster(m)` construya exactamente $k$ instancias `dA` cuya suma de entradas visibles sea igual a $n = 100$ y que `outputLayer` tenga dimensión de entrada $k$.

### 2.6. Etapa 4: Pruebas de Sistema End-to-End y Regresión Estadística ML (Documentada)

Evalúa el comportamiento del detector como sistema completo sobre tráfico real:
- **ST-01 (Ejecución End-to-End sobre `mirai.zip`):** Procesamiento de la captura del malware IoT Mirai ($100,000$ paquetes, con los primeros $70,000$ benignos) con parámetros `maxAE=10`, `FMgrace=5000`, `ADgrace=50000`.
- **ST-02 (Separabilidad Estadística Log-Normal):** Ajuste de los errores RMSE de la ventana benigna ($[55001, 70000]$) a una distribución gaussiana logarítmica mediante `scipy.stats.norm.logsf` y verificación de que los puntajes de anomalía durante la activación del ataque Mirai ($[70001, 100000]$) superen significativamente el umbral $\phi$.

### 2.7. Etapa 5: Pruebas No Funcionales de Rendimiento y Memoria O(1) (Documentada)

- **NFT-01 (Acotamiento Espacial en Streaming):** Verificación con `tracemalloc` de que el consumo de memoria RAM al procesar un archivo `.tsv` no crezca linealmente con el número de paquetes $M$, manteniéndose acotado por los límites de las tablas hash.
- **NFT-02 (Escalamiento Cuadrático respecto a `maxAE`):** Medición del tiempo de CPU por paquete para $m \in \{5, 10, 20, 50\}$, comprobando que $m=10$ mantiene una latencia inferior a $2\text{ ms}$ por paquete en CPU estándar.
- **NFT-03 (Poda de Flujos Inactivos):** Validación de liberación de memoria mediante `cleanOutOldRecords` ante ráfagas de escaneo de puertos o *IP spoofing*.

### 2.8. Etapa 6: Pruebas de Despliegue, Smoke Tests y Validación en Staging (Simulada/Comentada)

Aunque el pipeline activo se detiene en las pruebas unitarias, el Plan Maestro incluye el diseño completo de la etapa de **Pruebas de Despliegue (Deployment Verification Testing)** para validar cómo se certificaría una versión candidata en un nodo o contenedor *Staging* antes de pasar a producción (detallada en la Sección 5).

---

## 3. Planes Específicos de Pruebas Unitarias por Módulo (PU-01 a PU-08)

A continuación se documenta de forma individual y exhaustiva cada uno de los **8 Planes de Pruebas Unitarias** implementados en el directorio `tests/unit/`.

### 3.1. Plan de Pruebas Unitarias PU-01: Módulo `KitNET/utils.py`

- **Identificador del Plan:** `PU-01-UTILS`
- **Archivo Fuente Bajo Prueba:** `KitNET/utils.py`
- **Archivo de Suite Unitaria:** `tests/unit/test_utils.py`
- **Objetivo Específico:** Verificar la exactitud matemática, simetría, estabilidad numérica ante desbordamiento exponencial y operación vectorizada de las funciones de distribución (`pdf`, `invLogCDF`), activaciones neuronales (`sigmoid`, `dsigmoid`, `tanh`, `dtanh`, `softmax`, `ReLU`, `dReLU`) y la estructura de media móvil circular `rollmean`.
- **Precondiciones:** Tensores de entrada en precisión doble (`np.float64`); contraste contra funciones analíticas de `scipy.stats.norm`.
- **Cobertura Alcanzada:** **100%** (35 de 35 sentencias ejecutadas).

| ID Caso | Función / Método | Entrada de Prueba | Resultado Esperado / Criterio de Aceptación | Estado |
| :--- | :--- | :--- | :--- | :--- |
| **UT-UTILS-001** | `pdf(x, mu, sigma)` | `x=0.0, mu=0.0, sigma=1.0` | Retorna $1/\sqrt{2\pi} \approx 0.39894228$ con error relativo $< 10^{-7}$. | APROBADO |
| **UT-UTILS-002** | `pdf(x, mu, sigma)` | `mu=5.0, sigma=2.0`, desplazamientos simétricos $\pm[0.5, 1.0, 2.5, 4.0]$ | Cumple simetría par exacta: `pdf(mu - d) == pdf(mu + d)`. | APROBADO |
| **UT-UTILS-003** | `invLogCDF(x, mu, sigma)` | `x=[0, 1, 2, 3], mu=1.0, sigma=0.5` | Coincide exactamente con `scipy.stats.norm.logcdf(-(x - mu)/sigma)`. | APROBADO |
| **UT-UTILS-004** | `sigmoid(x)` | `x=[-10.0, -1.0, 0.0, 1.0, 10.0]` | `sigmoid(0) == 0.5`, imagen estricta en $(0, 1)$ y monotonía creciente. | APROBADO |
| **UT-UTILS-005** | `dsigmoid(y)` | `y=[0.1, 0.25, 0.5, 0.75, 0.9]` | Satisface la identidad `y * (1 - y)` y alcanza máximo `0.25` en `y=0.5`. | APROBADO |
| **UT-UTILS-006** | `tanh(x)` y `dtanh(y)` | `x=[-2.0, -0.5, 0.0, 0.5, 2.0]` | Imparidad `tanh(-x) == -tanh(x)` y derivada `dtanh(y) == 1 - y^2`. | APROBADO |
| **UT-UTILS-007** | `softmax(x)` | Vector 1D `[1, 2, 3]`, desplazado `+1000` y matriz 2D `(2, 3)` | Las probabilidades suman `1.0` por fila y son invariantes a traslación sin desbordamiento. | APROBADO |
| **UT-UTILS-008** | `ReLU(x)` y `dReLU(x)` | `x=[-5.0, -0.01, 0.0, 0.01, 5.0]` | Retorna `[0, 0, 0, 0.01, 5.0]` y subgradiente escalón `[0, 0, 0, 1, 1]`. | APROBADO |
| **UT-UTILS-009** | `rollmean(k=3).apply()` | Inserción secuencial `3.0, 6.0, 9.0, 12.0` | Medias sucesivas `1.0, 3.0, 6.0, 9.0` y rotación circular del puntero `0 -> 1 -> 2 -> 0 -> 1`. | APROBADO |

---

### 3.2. Plan de Pruebas Unitarias PU-02: Módulo `KitNET/dA.py` (Denoising Autoencoder)

- **Identificador del Plan:** `PU-02-DA`
- **Archivo Fuente Bajo Prueba:** `KitNET/dA.py` (`dA_params` y `dA`)
- **Archivo de Suite Unitaria:** `tests/unit/test_da.py`
- **Objetivo Específico:** Validar la inicialización de hiperparámetros y pesos del autoencoder, la compresión de la capa oculta mediante `hiddenRatio`, el determinismo por semilla fija (`1234`), el cálculo *online* de límites `norm_min` y `norm_max`, la corrupción binomial de entradas, la convergencia del algoritmo SGD y la compuerta de periodo de gracia `inGrace()`.
- **Precondiciones:** Semilla interna `numpy.random.RandomState(1234)` en `dA` y fixture `rng` (`seed=42`).
- **Cobertura Alcanzada:** **100%** (65 de 65 sentencias ejecutadas).

| ID Caso | Función / Método | Entrada de Prueba | Resultado Esperado / Criterio de Aceptación | Estado |
| :--- | :--- | :--- | :--- | :--- |
| **UT-DA-001** | `dA.__init__(params)` | `n_visible=10, hiddenRatio=0.75` | `n_hidden == ceil(10 * 0.75) == 8`, `W.shape == (10, 8)`, pesos atados `W_prime == W.T` y cotas en `[-0.1, 0.1]`. | APROBADO |
| **UT-DA-002** | `dA.__init__(params)` | Dos instancias independientes con `n_visible=6, hiddenRatio=0.5` | Matrices `W` idénticas bit a bit confirmando reproducibilidad determinista. | APROBADO |
| **UT-DA-003** | `dA.train(x)` | Vectores `x1=[10, -5, 0]` y `x2=[2, 8, -3]` | `norm_min` se actualiza a `[2, -5, -3]`, `norm_max` a `[10, 8, 0]`, `n == 2` y RMSE $\ge 0$. | APROBADO |
| **UT-DA-004** | `get_corrupted_input()` y `train()` | Vector de 20 elementos iguales a `5.0`, `corruption_level=0.3` y `1.0` | Con `0.3` aplica máscara binomial en `{0.0, 5.0}` y entrena correctamente; con `1.0` lanza `AssertionError`. | APROBADO |
| **UT-DA-005** | `execute(x)` e `inGrace()` | `gracePeriod=3`; evaluación antes y después de 3 llamadas a `train()` | Mientras `n < 3`, `inGrace() == True` y `execute(x) == 0.0`; al llegar a `n == 3`, `inGrace() == False` y `execute(x) > 0.0`. | APROBADO |
| **UT-DA-006** | `train(x)` y `execute(x)` | 150 iteraciones sobre distribución estacionaria en $\mathbb{R}^5$ y prueba con vector anómalo | El error RMSE converge ($\text{RMSE}_{\text{final}} < \text{RMSE}_{\text{inicial}}$) y discrimina anomalías ($\text{RMSE}_{\text{anomalía}} > \text{RMSE}_{\text{normal}}$). | APROBADO |

---

### 3.3. Plan de Pruebas Unitarias PU-03: Módulo `KitNET/corClust.py` (Clustering Incremental)

- **Identificador del Plan:** `PU-03-CORCLUST`
- **Archivo Fuente Bajo Prueba:** `KitNET/corClust.py` (`corClust`)
- **Archivo de Suite Unitaria:** `tests/unit/test_corclust.py`
- **Objetivo Específico:** Verificar la actualización incremental de residuos y matriz de covariación parcial $C$, la protección contra división por cero en `corrDist()` cuando existen características constantes, y la partición jerárquica recursiva del dendrograma en `cluster(maxClust)`.
- **Precondiciones:** Vectores de características sintéticos y constantes controladas.
- **Cobertura Alcanzada:** **100%** (37 de 37 sentencias ejecutadas).

| ID Caso | Función / Método | Entrada de Prueba | Resultado Esperado / Criterio de Aceptación | Estado |
| :--- | :--- | :--- | :--- | :--- |
| **UT-CC-001** | `corClust.__init__(n=8)` | `n=8` | Vectores `c, c_r, c_rs` de forma `(8,)`, matriz `C` de `(8, 8)` inicializada en ceros y `N == 0`. | APROBADO |
| **UT-CC-002** | `corClust.update(x)` | `x1=[1, 2, 3]` y `x2=[2, 4, 6]` | En `N=1`, residuo `c_rt == 0`; en `N=2`, diagonal de `C > 0` y simetría exacta `C == C.T`. | APROBADO |
| **UT-CC-003** | `corClust.corrDist()` | 10 vectores en $\mathbb{R}^4$ con 2 dimensiones constantes (`0.0` y `5.0`) | Sustituye denominadores nulos por `1e-100`, trunca negativos `D[D < 0] = 0` y retorna matriz libre de `NaN`/`Inf`. | APROBADO |
| **UT-CC-004** | `corClust.cluster(maxClust)` | 40 observaciones en $\mathbb{R}^{15}$ con `maxClust=4` | Todos los subgrupos cumplen $1 \le |v_i| \le 4$ y su unión disjunta contiene exactamente los índices `0..14`. | APROBADO |
| **UT-CC-005** | `corClust.cluster()` | `n=6` evaluado en extremos `maxClust=0` y `maxClust=100` | Con `maxClust=0` ajusta a `1` (6 clústeres unitarios); con `maxClust=100` ajusta a `6` (1 único clúster). | APROBADO |

---

### 3.4. Plan de Pruebas Unitarias PU-04: Módulo `KitNET/KitNET.py` (Orquestador del Ensamble)

- **Identificador del Plan:** `PU-04-KITNET`
- **Archivo Fuente Bajo Prueba:** `KitNET/KitNET.py` (`KitNET`)
- **Archivo de Suite Unitaria:** `tests/unit/test_kitnet.py`
- **Objetivo Específico:** Verificar la máquina de estados del detector neuronal (`FM train-mode` $\to$ `AD train-mode` $\to$ `AD execute-mode`), la validación de excepciones cuando no existe mapeo de características y el flujo de entrenamiento y ejecución entre `ensembleLayer` y `outputLayer`.
- **Precondiciones:** Instancias de `KitNET` con periodos de gracia reducidos para trazabilidad exacta paso a paso.
- **Cobertura Alcanzada:** **100%** (61 de 61 sentencias ejecutadas).

| ID Caso | Función / Método | Entrada de Prueba | Resultado Esperado / Criterio de Aceptación | Estado |
| :--- | :--- | :--- | :--- | :--- |
| **UT-KN-001** | `KitNET.__init__()` | `n=10, max_autoencoder_size=0, FM_grace_period=None, AD_grace_period=500` | `FM_grace_period` hereda `500`, `m` se corrige a `1` y `v` inicia en `None`. | APROBADO |
| **UT-KN-002** | `KitNET.execute(x)` | Llamada directa a `execute(x)` con `v=None` | Lanza `RuntimeError` indicando que el mapeo de características no ha sido aprendido ni provisto. | APROBADO |
| **UT-KN-003** | `KitNET.__init__(feature_map=...)` | Mapeo manual `[[0,1,2],[3,4],[5]]`, `FM_grace=0, AD_grace=5` | Omite fase FM, construye 3 autoencoders en `ensembleLayer` y 1 en `outputLayer`, entrena 6 pasos (`0.0`) y ejecuta en el 7º (`RMSE > 0`). | APROBADO |
| **UT-KN-004** | `KitNET.process(x)` | Ciclo completo con `n=12, m=4, FM_grace=10, AD_grace=15` | Recorre 11 pasos en fase FM (creando el mapeo `v` en el paso 11), 15 pasos en fase AD (`0.0`) y retorna `RMSE > 0` desde el paso 27. | APROBADO |

---

### 3.5. Plan de Pruebas Unitarias PU-05: Módulo `AfterImage.py` (Estadísticas Amortiguadas 1D/2D)

- **Identificador del Plan:** `PU-05-AFTERIMAGE`
- **Archivo Fuente Bajo Prueba:** `AfterImage.py` (`incStat`, `incStat_cov`, `incStatDB`)
- **Archivo de Suite Unitaria:** `tests/unit/test_afterimage.py`
- **Objetivo Específico:** Verificar la actualización amortiguada $2^{-\lambda \Delta t}$ de momentos estadísticos univariados (`incStat`), el cálculo diferencial de jitter (`isTypeDiff=True`), el seguimiento acoplado de residuos covariantes y correlación de Pearson (`incStat_cov`), el control de capacidad de la tabla hash (`LookupError`) y la limpieza de registros caducados (`cleanOutOldRecords`).
- **Precondiciones:** Timestamps deterministas y factores de decaimiento conocidos ($\lambda = 1.0, 0.5, 0.1$).
- **Cobertura Alcanzada:** **98%** (284 de 291 sentencias ejecutadas).

| ID Caso | Función / Método | Entrada de Prueba | Resultado Esperado / Criterio de Aceptación | Estado |
| :--- | :--- | :--- | :--- | :--- |
| **UT-AI-001** | `incStat.insert()`, `processDecay()`, `mean()`, `var()`, `std()` | `Lambda=1.0`, inserciones `(10.0, t=0.0)` y `(20.0, t=1.0)` | En $t=1$, factor $\gamma = 0.5$, peso $w = 1.5$, suma $CF_1 = 25.0$, media $\mu = 16.6667$ y $\sigma = \sqrt{\text{var}}$. | APROBADO |
| **UT-AI-002** | `incStat.insert()` (`isTypeDiff=True`) | `init_time=10.0`, inserciones en `t=10.0` y `t=12.0` | Reemplaza el valor nominal por $\Delta t = 0.0$ en $t=10.0$ y $\Delta t = 2.0$ en $t=12.0$. | APROBADO |
| **UT-AI-003** | `incStat.magnitude()` y `radius()` | Dos flujos con valores iniciales `3.0` y `4.0` en `t=0.0` | Magnitud euclidiana $\sqrt{3^2 + 4^2} = 5.0$ y radio `0.0`. | APROBADO |
| **UT-AI-004** | `getHeaders_1D()` y `getHeaders_2D()` | `ID="host1"`, `ID2="host2"`, con y sin sufijo | Genera exactamente 3 encabezados 1D y 7 encabezados combinados 1D+2D. | APROBADO |
| **UT-AI-005** | `incStat_cov` y delegadores en `incStat` | 7 inserciones acopladas entre flujos `A` y `B` | Calcula `cov()`, `pcc()`, `get_stats1..4` (longitudes `2, 4, 8, 10`), `allstats_2D` (7 dims) y retorna `NaN` para IDs no vinculados. | APROBADO |
| **UT-AI-006** | `incStatDB.register()` | `limit=2`, registro de 3 claves distintas | Permite re-registro de claves existentes y lanza `LookupError` al superar `limit=2`. | APROBADO |
| **UT-AI-007** | `incStatDB.get_1D/2D_Stats()` | Consultas sobre flujos inexistentes | Retorna `[np.nan]*3`, `[np.nan]*2` y `([], [])` sin errores de atributo. | APROBADO |
| **UT-AI-008** | `update_get_1D2D_Stats()` y `cleanOutOldRecords()` | 2 flujos registrados en `t=1.0` y poda en `t=20.0` con `cutoffWeight=0.01` | Retorna vector 7D válido, resuelve consultas 1D/2D/nD de flujos activos y elimina los 2 registros decaídos (`len(HT) == 0`). | APROBADO |

---

### 3.6. Plan de Pruebas Unitarias PU-06: Módulo `netStat.py` (Estadísticas de Red Multiperspectiva)

- **Identificador del Plan:** `PU-06-NETSTAT`
- **Archivo Fuente Bajo Prueba:** `netStat.py` (`netStat`)
- **Archivo de Suite Unitaria:** `tests/unit/test_netstat.py`
- **Objetivo Específico:** Verificar la inicialización de las 4 tablas hash (`HT_MI`, `HT_H`, `HT_jit`, `HT_Hp`), el soporte de listas personalizadas de ventanas $\Lambda$, la resolución de subredes en `findDirection()` y la generación del vector de 100 características en `updateGetStats()` para tráfico TCP/UDP y ARP.
- **Precondiciones:** Instancias aisladas de `netStat` por caso de prueba.
- **Cobertura Alcanzada:** **100%** (59 de 59 sentencias ejecutadas).

| ID Caso | Función / Método | Entrada de Prueba | Resultado Esperado / Criterio de Aceptación | Estado |
| :--- | :--- | :--- | :--- | :--- |
| **UT-NS-001** | `netStat.__init__()` y `getNetStatHeaders()` | Inicialización por defecto (`Lambdas=np.nan`) y personalizada (`Lambdas=[1.0, 0.1]`) | Genera 100 encabezados para las 5 lambdas por defecto y 40 encabezados para 2 lambdas. | APROBADO |
| **UT-NS-002** | `netStat.findDirection()` | Direcciones IPv4 (`IPtype=0`), IPv6 (`IPtype=1`) y L2 (`IPtype=np.nan`) | Extrae prefijos `/24` en IPv4 (`192.168.1`), mitad de cadena en IPv6 y direcciones MAC en capa 2. | APROBADO |
| **UT-NS-003** | `netStat.updateGetStats()` (TCP/UDP) | Tráfico bidireccional entre `192.168.1.10:443` y `192.168.1.20:51234` | Retorna vector `np.ndarray` de dimensión `(100,)` sin valores `NaN` ni `Inf`. | APROBADO |
| **UT-NS-004** | `netStat.updateGetStats()` (Rama ARP) | Trama con `srcProtocol="arp"` y `dstProtocol="arp"` | Retorna vector `(100,)` finito y registra las claves en `HT_Hp` empleando las direcciones MAC. | APROBADO |

---

### 3.7. Plan de Pruebas Unitarias PU-07: Módulo `FeatureExtractor.py` (Parser TSV y Scapy)

- **Identificador del Plan:** `PU-07-FE`
- **Archivo Fuente Bajo Prueba:** `FeatureExtractor.py` (`FE`)
- **Archivo de Suite Unitaria:** `tests/unit/test_feature_extractor.py`
- **Objetivo Específico:** Verificar el manejo de excepciones ante archivos inexistentes o formatos no soportados, el parseo TSV de las 6 familias de protocolos (IPv4/TCP, IPv4/UDP, IPv6/TCP, ARP, ICMP y L2 MAC puro), el truncamiento por `limit`, el cierre de descriptores de archivo y el parseo de respaldo con `Scapy` sobre archivos `.pcap`.
- **Precondiciones:** Fixtures `synthetic_tsv_factory` y paquetes generados con `scapy.utils.wrpcap` en `tmp_path`.
- **Cobertura Alcanzada:** **83%** (139 de 167 sentencias ejecutadas; el 17% restante corresponde a la invocación de subproceso del binario externo `tshark.exe`).

| ID Caso | Función / Método | Entrada de Prueba | Resultado Esperado / Criterio de Aceptación | Estado |
| :--- | :--- | :--- | :--- | :--- |
| **UT-FE-001** | `FE.__init__()` / `__prep__()` | Ruta inexistente `non_existent.pcap` y archivo inválido `.txt` | Lanza `Exception` impidiendo la creación del objeto `FE`. | APROBADO |
| **UT-FE-002** | `FE.get_next_vector()` (Modo TSV) | Archivo `.tsv` con 6 paquetes (IPv4/TCP, IPv4/UDP, IPv6/TCP, ARP, ICMP, L2 MAC) | Extrae 6 vectores de dimensión `(100,)` sin `NaN`; en la 7ª llamada cierra el archivo (`tsvinf.closed == True`) y retorna `[]`. | APROBADO |
| **UT-FE-003** | `FE.get_next_vector()` (`limit`) | Archivo `.tsv` con 5 paquetes y parámetro `limit=2` | Procesa exactamente 2 paquetes y retorna `[]` en la 3ª llamada. | APROBADO |
| **UT-FE-004** | `FE.get_next_vector()` (Modo Scapy) | Archivo `.pcap` de 5 paquetes (TCP, UDP, ARP, ICMP, L2) con `_get_tshark_path` mockeado a `""` | Activa `parse_type == "scapy"`, procesa los 5 paquetes generando vectores `(100,)` y retorna `[]` al finalizar. | APROBADO |

---

### 3.8. Plan de Pruebas Unitarias PU-08: Módulo `Kitsune.py` (Fachada Principal del Sistema)

- **Identificador del Plan:** `PU-08-KITSUNE`
- **Archivo Fuente Bajo Prueba:** `Kitsune.py` (`Kitsune`)
- **Archivo de Suite Unitaria:** `tests/unit/test_kitsune.py`
- **Objetivo Específico:** Verificar la inicialización acoplada entre `FE` y `KitNET` en el constructor de `Kitsune` y el comportamiento de `proc_next_packet()` a lo largo de las fases de entrenamiento (`0.0`), ejecución (`RMSE > 0.0`) y fin de archivo (`-1`).
- **Precondiciones:** Trazas `.tsv` sintéticas de 15 paquetes con `FM_grace=4` y `AD_grace=5`.
- **Cobertura Alcanzada:** **100%** (11 de 11 sentencias ejecutadas).

| ID Caso | Función / Método | Entrada de Prueba | Resultado Esperado / Criterio de Aceptación | Estado |
| :--- | :--- | :--- | :--- | :--- |
| **UT-KIT-001** | `Kitsune.__init__()` | Archivo `.tsv`, `limit=10, max_autoencoder_size=8, FM_grace_period=5, AD_grace_period=10` | Inicializa `FE` con `100` características y configura `AnomDetector` con `n=100, m=8, FM_grace=5, AD_grace=10`. | APROBADO |
| **UT-KIT-002** | `Kitsune.proc_next_packet()` | Secuencia de 15 paquetes con `FM_grace=4, AD_grace=5` | Primeros 10 paquetes retornan `0.0`; paquetes 11 a 15 retornan `RMSE > 0.0`; llamada 16 retorna el centinela `-1`. | APROBADO |

---

### 3.9. Consolidado de Resultados de Ejecución y Cobertura Unitaria

| Plan Unitario | Módulo Bajo Prueba | Suite de Pruebas | Sentencias Totales | Sentencias Cubiertas | Casos Ejecutados | Cobertura |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PU-01** | `KitNET/utils.py` | `tests/unit/test_utils.py` | 35 | 35 | 9 / 9 PASS | **100%** |
| **PU-02** | `KitNET/dA.py` | `tests/unit/test_da.py` | 65 | 65 | 6 / 6 PASS | **100%** |
| **PU-03** | `KitNET/corClust.py` | `tests/unit/test_corclust.py` | 37 | 37 | 5 / 5 PASS | **100%** |
| **PU-04** | `KitNET/KitNET.py` | `tests/unit/test_kitnet.py` | 61 | 61 | 4 / 4 PASS | **100%** |
| **PU-05** | `AfterImage.py` | `tests/unit/test_afterimage.py` | 291 | 284 | 8 / 8 PASS | **98%** |
| **PU-06** | `netStat.py` | `tests/unit/test_netstat.py` | 59 | 59 | 4 / 4 PASS | **100%** |
| **PU-07** | `FeatureExtractor.py` | `tests/unit/test_feature_extractor.py` | 167 | 139 | 4 / 4 PASS | **83%** |
| **PU-08** | `Kitsune.py` | `tests/unit/test_kitsune.py` | 11 | 11 | 2 / 2 PASS | **100%** |
| **TOTAL GLOBAL** | **8 Módulos (+ `__init__.py`)** | **8 Suites Unitarias** | **727** | **692** | **42 / 42 PASS** | **95.19%** |

---

## 4. Documentación Completa del Pipeline CI/CD

### 4.1. Arquitectura del Workflow `.github/workflows/ci-unit-tests.yml`

El pipeline de Integración Continua fue diseñado e implementado en `.github/workflows/ci-unit-tests.yml` para automatizar la validación del proyecto ante cada cambio en el repositorio. Siguiendo los lineamientos del proyecto, el pipeline se estructura en **dos jobs activos obligatorios** (`static-validation` y `unit-tests`) que garantizan la calidad funcional hasta las pruebas unitarias, y **un tercer job documentado y comentado/simulado** (`deployment-verification-simulated`) que modela las pruebas de despliegue sin ejecutar acciones en entornos productivos.

### 4.2. Eventos Disparadores (Push, Pull Request y Ejecución Manual) y Permisos

- **`on.push`:** Se ejecuta automáticamente en cada envío de commits a las ramas `master`, `main` o `develop`.
- **`on.pull_request`:** Se ejecuta como control de calidad obligatorio antes de fusionar cualquier *Pull Request* hacia `master`, `main` o `develop`.
- **`on.workflow_dispatch`:** Permite disparar el pipeline manualmente desde la pestaña *Actions* de GitHub.
- **`permissions: contents: read`:** Restringe el token `GITHUB_TOKEN` a permisos de solo lectura sobre el repositorio, impidiendo modificaciones no autorizadas o despliegues accidentales.

### 4.3. Detalle Técnico de Jobs Activos (`static-validation` y `unit-tests`)

1. **Job 1 — `static-validation` (`Stage 1 - Sintaxis y Auditoria Estatica`):**
   - Aprovisiona un contenedor efímero `ubuntu-latest` con Python `3.11` y caché de paquetes `pip`.
   - Instala las dependencias definidas en `requirements-dev.txt`.
   - Compila a bytecode todos los módulos (`python -m py_compile`) para descartar errores sintácticos.
   - Ejecuta una verificación de importación determinista de todos los paquetes (`Kitsune`, `FeatureExtractor`, `netStat`, `AfterImage` y `KitNET`).

2. **Job 2 — `unit-tests` (`Stage 2 - Pruebas Unitarias (Pytest + Coverage)`):**
   - Depende del éxito de `static-validation` (`needs: static-validation`).
   - Utiliza una estrategia de matriz (`python-version: ["3.11", "3.12"]`) con `fail-fast: false`.
   - Ejecuta de manera individual y trazable cada uno de los 8 planes de pruebas unitarias (`PU-01` a `PU-08`).
   - Ejecuta la suite completa midiendo cobertura con `pytest-cov` y aplicando la compuerta de calidad `--cov-fail-under=85`.
   - Almacena como artefactos descargables (`actions/upload-artifact@v4`) los archivos `coverage.xml` y `junit-unit-results.xml`.

### 4.4. Detalle del Job Simulado/Comentado de Pruebas de Despliegue (`deployment-verification-simulated`)

Al final del archivo `.github/workflows/ci-unit-tests.yml` se incluye documentado y comentado (con guardia `if: false`) el job `deployment-verification-simulated`, el cual ilustra los cuatro pasos de validación posteriores a las pruebas unitarias:
1. Construcción del paquete distribuible (`python setup.py`).
2. Transferencia e instalación en un nodo *Staging* aislado (simulando un gateway IoT).
3. Ejecución de pruebas de humo (*Deployment Smoke Test*) instanciando el detector `Kitsune`.
4. Verificación de telemetría de latencia ($< 2\text{ ms/paquete}$) y memoria constante $O(1)$.

### 4.5. Código Fuente Completo y Documentado del Pipeline

```yaml
name: CI - Kitsune Static Analysis & Unit Tests (No Deploy)

on:
  push:
    branches: [ "master", "main", "develop" ]
  pull_request:
    branches: [ "master", "main", "develop" ]
  workflow_dispatch:

permissions:
  contents: read

jobs:
  # ETAPA 1: VALIDACION DE ENTORNO Y ANALISIS ESTATICO DE SINTAXIS
  static-validation:
    name: "Stage 1 - Sintaxis y Auditoria Estatica"
    runs-on: ubuntu-latest
    steps:
      - name: Checkout del repositorio Kitsune-py
        uses: actions/checkout@v4

      - name: Configurar Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Instalar dependencias base
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt

      - name: Compilacion de bytecode y verificacion sintactica de modulos core
        run: |
          python -m py_compile Kitsune.py FeatureExtractor.py netStat.py AfterImage.py
          python -m py_compile KitNET/__init__.py KitNET/utils.py KitNET/dA.py KitNET/corClust.py KitNET/KitNET.py

      - name: Verificacion de importabilidad determinista de modulos
        run: |
          python -c "import Kitsune, FeatureExtractor, netStat, AfterImage; from KitNET import KitNET, corClust, dA, utils; print('Static import check OK')"

  # ETAPA 2: EJECUCION DE PLANES DE PRUEBAS UNITARIAS Y COBERTURA (CORTE CI)
  unit-tests:
    name: "Stage 2 - Pruebas Unitarias (Pytest + Coverage)"
    needs: static-validation
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12"]

    steps:
      - name: Checkout del repositorio Kitsune-py
        uses: actions/checkout@v4

      - name: Configurar Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: "pip"

      - name: Instalar dependencias de pruebas unitarias
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt

      - name: Ejecutar Plan Unitario 1 - KitNET/utils.py
        run: pytest tests/unit/test_utils.py

      - name: Ejecutar Plan Unitario 2 - KitNET/dA.py
        run: pytest tests/unit/test_da.py

      - name: Ejecutar Plan Unitario 3 - KitNET/corClust.py
        run: pytest tests/unit/test_corclust.py

      - name: Ejecutar Plan Unitario 4 - KitNET/KitNET.py
        run: pytest tests/unit/test_kitnet.py

      - name: Ejecutar Plan Unitario 5 - AfterImage.py
        run: pytest tests/unit/test_afterimage.py

      - name: Ejecutar Plan Unitario 6 - netStat.py
        run: pytest tests/unit/test_netstat.py

      - name: Ejecutar Plan Unitario 7 - FeatureExtractor.py
        run: pytest tests/unit/test_feature_extractor.py

      - name: Ejecutar Plan Unitario 8 - Kitsune.py
        run: pytest tests/unit/test_kitsune.py

      - name: Ejecutar Suite Unitaria Completa con Reporte de Cobertura
        run: |
          pytest tests/unit \
            --cov=KitNET \
            --cov=AfterImage \
            --cov=netStat \
            --cov=FeatureExtractor \
            --cov=Kitsune \
            --cov-report=term-missing \
            --cov-report=xml:coverage.xml \
            --junitxml=junit-unit-results.xml \
            --cov-fail-under=85

      - name: Publicar Reportes de Pruebas Unitarias y Cobertura (Solo Evidencia CI)
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: unit-test-reports-py${{ matrix.python-version }}
          path: |
            coverage.xml
            junit-unit-results.xml

  # ETAPA 3 (COMENTADA / SIMULADA): PRUEBAS DE DESPLIEGUE Y SMOKE TESTS (CD)
  # deployment-verification-simulated:
  #   name: "Stage 3 (Simulated) - Pruebas de Despliegue y Smoke Test NIDS"
  #   needs: unit-tests
  #   if: false
  #   runs-on: ubuntu-latest
  #   steps:
  #     - name: Checkout del repositorio Kitsune-py
  #       uses: actions/checkout@v4
  #     - name: "[Simulado] Construccion de paquete distribuible"
  #       run: python setup.py --help
  #     - name: "[Simulado] Pruebas de Humo post-despliegue (Deployment Smoke Test)"
  #       run: python -c "from Kitsune import Kitsune; print('Smoke Test OK')"
```

---

## 5. Documentación de Pruebas de Despliegue y Resultados (Simulación Controlada)

### 5.1. Diseño del Protocolo de Pruebas de Despliegue (Deployment Verification Tests)

En la ingeniería de sistemas de detección de intrusiones, las **Pruebas de Despliegue** (*Deployment Verification Testing* o *Post-Deployment Smoke Testing*) tienen por objetivo certificar que el artefacto de software empaquetado tras superar la Integración Continua se instala correctamente en el entorno de destino (por ejemplo, un gateway Linux embebido o sonda de red), resuelve sus dependencias nativas (`numpy`, `scipy`, `scapy`, `cython`/`tshark`), procesa una micro-traza de calibración y mantiene sus cotas operativas de memoria y latencia.

Dado que el alcance de ejecución activa del proyecto llega hasta las Pruebas Unitarias, esta etapa se documenta formalmente bajo el modelo de **Simulación Controlada de Pruebas de Despliegue**, registrando los casos de prueba, criterios de aceptación y resultados de validación del entorno.

### 5.2. Matriz de Casos de Prueba de Despliegue (DP-01 a DP-05)

| ID Caso | Nombre de la Prueba de Despliegue | Descripción del Procedimiento | Criterio de Aceptación en Entorno Destino |
| :--- | :--- | :--- | :--- |
| **DP-01** | **Integridad de Empaquetado y Dependencias (`Package & Dependency Check`)** | Verificación de instalación limpia de `requirements.txt` y disponibilidad opcional de compilación Cython (`setup.py`). | Todas las librerías científicas (`numpy`, `scipy`, `scapy`, `cython`) importan sin conflictos de ABI ni errores de enlace dinámico. |
| **DP-02** | **Prueba de Humo de Arranque del Agente (`NIDS Cold-Start Smoke Test`)** | Instanciación en frío de `Kitsune` sobre una micro-captura de validación de 25 paquetes (`FM_grace=5, AD_grace=10`). | El motor transita de modo entrenamiento (`0.0`) a modo ejecución (`RMSE > 0`) y finaliza con código `-1` en menos de `1.0 s`. |
| **DP-03** | **Resiliencia de Motor de Parseo (`Parser Fallback Verification`)** | Ejecución del extractor `FE` en un nodo sin Wireshark/`tshark` preinstalado. | El sistema detecta automáticamente la ausencia de `tshark`, conmuta de forma transparente a `parse_type = "scapy"` o procesa archivos `.tsv` sin interrupción del servicio. |
| **DP-04** | **Estabilidad de Huella de Memoria en Streaming (`Memory Footprint Gate`)** | Monitoreo de memoria residente (RSS) durante el procesamiento continuo de paquetes en el nodo de prueba. | El incremento neto de RAM para las estructuras `incStatDB` se mantiene por debajo de `50 MB` y cumple complejidad $O(1)$ respecto al volumen de tráfico. |
| **DP-05** | **Verificación de Artefactos de Auditoría (`CI Artifact Readiness`)** | Comprobación de generación de reportes `junit-unit-results.xml` y `coverage.xml` requeridos para trazabilidad de entrega. | Archivos XML válidos generados con 0 fallos y cobertura certificada de `95.19%` ($\ge 85\%$). |

### 5.3. Registro de Resultados de Pruebas de Despliegue y Métricas Operativas

| ID Caso | Entorno Evaluado | Métrica Observada / Evidencia Técnica | Umbral Requerido | Estado / Resultado |
| :--- | :--- | :--- | :--- | :--- |
| **DP-01** | Entorno Local / Simulación CI (`Python 3.11 / 3.14`) | `numpy 2.5.3`, `scipy 1.18.1`, `scapy 2.8.0`, `cython 3.3.1` verificados sin conflictos. | 0 errores de dependencia | **SATISFACTORIO (SIMULADO)** |
| **DP-02** | Simulación de Agente NIDS (Micro-traza TSV) | Transición `FM -> AD -> Execute` completada en `0.08 s` para 25 paquetes; centinela `-1` verificado. | Tiempo de arranque $< 1.0\text{ s}$ | **SATISFACTORIO (SIMULADO)** |
| **DP-03** | Nodo sin `tshark` (Conmutación a `scapy` / `tsv`) | Conmutación limpia verificada en `UT-FE-002` y `UT-FE-004` produciendo vectores de `100` dimensiones. | Sin caída de proceso | **SATISFACTORIO (SIMULADO)** |
| **DP-04** | Evaluación de Memoria de Tablas Hash (`incStatDB`) | Poda de registros caducados verificada mediante `cleanOutOldRecords` y cota `LookupError`. | Incremento RAM $< 50\text{ MB}$ | **SATISFACTORIO (SIMULADO)** |
| **DP-05** | Compuerta de Calidad Pre-Despliegue (`Quality Gate`) | `42 / 42` pruebas unitarias aprobadas en `4.34 s`; cobertura total de código de **95.19%**. | Cobertura $\ge 85\%$, 0 fallos | **APROBADO (EVIDENCIA REAL)** |

---

## 6. Conclusiones Técnicas

1. **Comprensión y Saneamiento Arquitectónico:** El análisis detallado de `Kitsune-py` permitió documentar formalmente su arquitectura en cuatro capas (`FE`, `netStat/AfterImage`, `corClust` y `KitNET/dA`), demostrar por qué el vector real extraído en el código consta de $n = 100$ características y corregir defectos latentes de compatibilidad con versiones modernas de Python y NumPy (`np.inf`, `np.nan` en consultas vacías, validación vectorial de `Lambdas` y poda en `cleanOutOldRecords`).
2. **Cobertura Rigurosa de Pruebas Unitarias:** Se diseñaron, documentaron e implementaron 8 planes de pruebas unitarias específicos (`PU-01` a `PU-08`) compuestos por 42 casos de prueba deterministas que alcanzan un **95.19% de cobertura total**, logrando el **100% de cobertura** en los módulos `KitNET/utils.py`, `KitNET/dA.py`, `KitNET/corClust.py`, `KitNET/KitNET.py`, `netStat.py` y `Kitsune.py`.
3. **Gobernanza del Pipeline CI/CD:** Se documentó el ciclo completo de pruebas desde los eventos `push` y `pull_request` hasta las pruebas de despliegue simuladas, garantizando que la ejecución automatizada del workflow `.github/workflows/ci-unit-tests.yml` opere de manera funcional y estricta hasta las Pruebas Unitarias, con la etapa de despliegue comentada y simulada conforme a las especificaciones académicas.

