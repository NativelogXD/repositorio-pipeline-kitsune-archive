# Plan de Pruebas Unitarias Específico `PU-01`: Módulo `KitNET/utils.py`

## 1. Identificación y Metadatos
* **ID del Plan:** `PU-01-UTILS`
* **Módulo Bajo Prueba:** [`KitNET/utils.py`](../../KitNET/utils.py)
* **Suite de Prueba:** [`tests/unit/test_utils.py`](../../tests/unit/test_utils.py)
* **Etapa del Pipeline CI:** `Stage 2 - Pruebas Unitarias (Pytest + Coverage)`
* **Cobertura Alcanzada:** **100%** (35/35 líneas ejecutadas)

## 2. Intuición Técnica y Alcance
El módulo `KitNET/utils.py` provee los bloques matemáticos elementales de los autoencoders y del ajuste estadístico de anomalías:
1. **Funciones Estadísticas (`pdf`, `invLogCDF`):** Evalúan la densidad gaussiana $\mathcal{N}(\mu, \sigma^2)$ y la función de supervivencia logarítmica $\log(1 - \Phi(z))$ empleada para estimar probabilidades de anomalía.
2. **Funciones de Activación y Gradientes (`sigmoid`, `dsigmoid`, `tanh`, `dtanh`, `softmax`, `ReLU`, `dReLU`):** Operan vectorialmente sobre tensores de NumPy durante la propagación hacia adelante y hacia atrás.
3. **Ventana Deslizante Circular (`rollmean`):** Calcula la media móvil de tamaño $k$ en tiempo $O(1)$ de inserción y $O(k)$ de reducción usando un puntero circular.

## 3. Precondiciones y Criterios de Aceptación
* Entradas vectorizadas en `np.float64`.
* Estabilidad numérica ante entradas de gran magnitud (ej. desplazamientos $+1000$ en `softmax` sin producir `NaN` por desbordamiento exponencial).
* Tolerancia numérica relativa `rtol <= 1e-7`.

## 4. Especificación de Casos de Prueba Unitarios

| ID Caso | Función / Clase | Descripción y Datos de Entrada | Oráculo / Resultado Esperado |
| :--- | :--- | :--- | :--- |
| **`UT-UTILS-001`** | `pdf(x, mu, sigma)` | Evaluación puntual en $x=0, \mu=0, \sigma=1$. | Valor exacto $\frac{1}{\sqrt{2\pi}} \approx 0.3989422804$. |
| **`UT-UTILS-002`** | `pdf(x, mu, sigma)` | Evaluación vectorizada en $\mu \pm [0.5, 1.0, 2.5, 4.0]$ con $\mu=5, \sigma=2$. | Invariante de paridad simétrica: $\text{pdf}(\mu - \delta) = \text{pdf}(\mu + \delta)$. |
| **`UT-UTILS-003`** | `invLogCDF(x, mu, sigma)` | Vector $x=[0, 1, 2, 3]$ con $\mu=1.0, \sigma=0.5$. | Igualdad numérica con `scipy.stats.norm.logcdf(-(x - mu)/sigma)`. |
| **`UT-UTILS-004`** | `sigmoid(x)` | Vector $x=[-10, -1, 0, 1, 10]$. | $\sigma(0)=0.5$, imagen contenida en $(0,1)$ y derivada discreta positiva. |
| **`UT-UTILS-005`** | `dsigmoid(y)` | Vector de activaciones $y=[0.1, 0.25, 0.5, 0.75, 0.9]$. | Identidad exacta $y(1-y)$ y valor máximo $0.25$ en $y=0.5$. |
| **`UT-UTILS-006`** | `tanh(x)`, `dtanh(y)` | Vector $x=[-2.0, -0.5, 0.0, 0.5, 2.0]$. | Imparidad $\tanh(-x) = -\tanh(x)$ y $\text{dtanh}(y) = 1 - y^2$. |
| **`UT-UTILS-007`** | `softmax(x)` | Vector 1D `[1, 2, 3]`, `[1001, 1002, 1003]` y matriz 2D `(2, 3)`. | Suma en el simplex igual a $1.0$ e invarianza al desplazamiento constante. |
| **`UT-UTILS-008`** | `ReLU(x)`, `dReLU(x)` | Vector $x=[-5.0, -0.01, 0.0, 0.01, 5.0]$. | Salidas `[0, 0, 0, 0.01, 5.0]` y subgradiente `[0, 0, 0, 1, 1]`. |
| **`UT-UTILS-009`** | `rollmean(k=3)` | Inserción secuencial de `3.0, 6.0, 9.0, 12.0`. | Secuencia de medias `1.0, 3.0, 6.0, 9.0` y rotación circular del índice `0 -> 1 -> 2 -> 0 -> 1`. |

## 5. Comando de Verificación
```powershell
python -m pytest tests/unit/test_utils.py --cov=KitNET.utils --cov-report=term-missing
```

