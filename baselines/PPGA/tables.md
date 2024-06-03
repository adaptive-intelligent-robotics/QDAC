**Table 1**

| **Algorithm**      | **Exp.**    | **QD Score**       | **Coverage** | **Best Reward** |
|--------------------|-------------|--------------------|--------------|-----------------|
| PPGA               | Humanoid    | $7.24 \times 10^6$ | 77.75%       | 8758            |
|                    | Walker2d    | $7.26 \times 10^6$ | 66.76%       | 3978            |
|                    | Halfcheetah | $2.71 \times 10^7$ | 90.51%       | 7673            |
|                    | Ant         | $2.3 \times 10^7$  | 51.65%       | 5997            |
| QDPG               | Humanoid    | $2.1 \times 10^6$  | 96.94%       | 2060            |
|                    | Walker2d    | $2.3 \times 10^5$  | 82.25%       | 2418            |
|                    | Halfcheetah | $9.46 \times 10^5$ | 97.00%       | 1286            |
|                    | Ant         | $2.85 \times 10^7$ | 61.47%       | 3595            |
| CMA-MAEGA(TD3, ES) | Humanoid    | $9.06 \times 10^5$ | 96.44%       | 1106            |
|                    | Walker2d    | $1.7 \times 10^6$  | 69.72%       | 1252            |
|                    | Halfcheetah | X                  | X            | X               |
|                    | Ant         | $1.66 \times 10^7$ | 46.03%       | 1625            |
_Table 1 compares the performance of our algorithm PPGA to QDPG and CMA-MAEGA(TD3, ES) on all four tasks._


**Table 2**

| **Algorithm** | **Exp.** | **Wall Clock Time (Hrs)** | **QD Score**       | **Coverage** | **Best Reward** |
|---------------|----------|---------------------------|--------------------|--------------|-----------------|
| PPGA          | Humanoid | 22.86                     | $7.24 \times 10^6$ | 77.75%       | 8758            |
| PGA-ME        | Humanoid | 24                        | $2.21 \times 10^6$ | 98.76%       | 1920            |
_Table 2 compares the performance of PPGA to PGA-ME on humanoid given equivalent wall-clock time. This corresponds to 
4.8 million controller evaluations for PGA-ME, or 8X the number of controller evaluations as PPGA._


**Table 3** 

| **Algorithm**      | **Humanoid** | **Walker2d** | **Halfcheetah** | **Ant** |
|--------------------|--------------|--------------|-----------------|---------|
| PPGA               | 22.8         | 7.5          | 14.3            | 20.4    |
| PGAME              | 11.5         | 1.8          | 5.9             | 5.1     |
| sep-CMA-MAE        | 2.3          | 2.3          | 4.5             | 3.3     |
| QDPG               | 6.5          | 6.7          | 9.41            | 6.2     |
| CMA-MAEGA(TD3, ES) | 4            | 4.4          | X               | 8.3     |

_Table 3 compares the wall clock time of PPGA to baselines. The results for PPGA, PGAME, and sep-CMA-MAE are based on 
the experiments reported in Figure 3 of the manuscript, while the results for QDPG and CMA-MAEGA(TD3, ES) were recently collected._
