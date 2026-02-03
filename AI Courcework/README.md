# AI Coursework — CSF 407 (Artificial Intelligence)

This folder contains coursework completed as part of **CSF 407 – Artificial Intelligence**.  
The primary focus of this work is the application of **Genetic Algorithms (GA)** to solve a real-world **engineering optimization problem**.

The project integrates:
- A Python-based optimization implementation
- A formal academic presentation
- A peer-reviewed research paper as theoretical foundation

---

## 📌 Project Title  
**Optimal Heat Exchanger Design using Genetic Algorithms**

---

## 📂 Folder Contents

| File | Description |
|----|----|
| `aicode final.py` | Python implementation of the Genetic Algorithm for heat exchanger optimization |
| `CSF 407 – ARTIFICIAL INTELLIGENCE.pptx` | Course presentation describing problem formulation, GA approach, and results |
| `1-s2.0-S1359431107002839-main.pdf` | Reference research paper forming the theoretical basis of the project |

---

## 🧠 Problem Background

Shell-and-tube heat exchangers are widely used in chemical and process industries.  
Their design involves balancing:
- **Capital investment** (equipment size, surface area)
- **Operating cost** (pumping power, pressure drops)

Traditional design methods rely on iterative heuristics and designer experience, which do not guarantee an economically optimal solution.  
This project explores **Genetic Algorithms** as a systematic optimization technique to minimize **total discounted lifecycle cost**.

---

## 🎯 Objective

To identify an optimal combination of design parameters that minimizes the **total discounted operating cost** of a shell-and-tube heat exchanger while satisfying physical and operational constraints.

---

## ⚙️ Optimization Variables

The following parameters are optimized:
- **Shell diameter (Ds)**
- **Tube outer diameter (Do)**
- **Baffle spacing (B)**

Each candidate solution represents a feasible heat exchanger configuration within predefined bounds.

---

## 🧬 Genetic Algorithm Methodology

The implemented Genetic Algorithm includes:

- **Population initialization** using randomly generated feasible solutions  
- **Fitness evaluation** based on inverse discounted operating cost  
- **Selection** of top-performing (elite) solutions  
- **Crossover** via parameter pooling from elite solutions  
- **Mutation** through small random perturbations  
- **Constraint handling** to enforce physical feasibility  
- **Iterative evolution** across generations until convergence  

The GA searches a large, non-linear design space efficiently without requiring gradient information.

---

## 📊 Fitness Function

The fitness function is derived from a detailed cost model that computes:

- Heat duty and heat transfer surface area  
- Fluid velocities and Reynolds numbers  
- Friction factors and pressure drops  
- Annual pumping power requirements  
- Total discounted operating cost over equipment lifetime  

Fitness is defined as: fitness = |1 / total_discounted_cost|


Lower-cost designs therefore receive higher fitness values.

---

## 📈 Output and Results

For each generation, the algorithm outputs:
- Best fitness score
- Corresponding design parameters (Ds, Do, B)
- Associated cost value

The algorithm terminates when:
- A sufficiently optimal solution is found, or
- A predefined generation limit is reached

The results demonstrate the effectiveness of GA-based optimization compared to conventional design approaches.

---

## 📚 Research Foundation

The implementation and cost formulation are based on the following peer-reviewed study:

**Caputo, A. C., Pelagagge, P. M., & Salini, P. (2008)**  
*Heat exchanger design based on economic optimisation*  
Applied Thermal Engineering, 28, 1151–1159.

The accompanying presentation summarizes:
- Problem formulation
- Optimization framework
- Genetic Algorithm design
- Result interpretation and comparison

---

## 🛠 Technologies Used

- Python  
- Genetic Algorithms  
- Mathematical modeling  
- Engineering optimization  
- Randomized and heuristic search  

---

## 🎯 Key Learnings

- Applying AI-based heuristic optimization to physical systems  
- Translating engineering equations into computational fitness functions  
- Managing constraints in evolutionary algorithms  
- Understanding trade-offs between capital and operating costs  
- Structuring AI solutions for real-world, non-convex problems  

---

## 🔮 Possible Extensions

- Modularizing the GA implementation for clarity and reuse  
- Visualizing convergence and fitness evolution  
- Comparing GA with other optimization techniques (PSO, Simulated Annealing)  
- Extending the cost model to include manufacturing constraints  

---

## 📖 Academic Context

This work was completed as part of **CSF 407 – Artificial Intelligence**,  
demonstrating the practical application of **evolutionary computation and heuristic search** methods to an engineering optimization problem.



