# Combinatorial Causal Bandits without Graph Skeleton
[![](https://tokei.rs/b1/github/fengtony686/CCB-without-Graph)](https://github.com/fengtony686/CCB-without-Graph)      
This repository contains code of numerical experiments for paper:         
**Combinatorial Causal Bandits without Graph Skeleton**  
In submission  
[Shi Feng](https://fengshi.link/), [Nuoya Xiong](https://xiongny.github.io/index.html), [Wei Chen](https://www.microsoft.com/en-us/research/people/weic/)          
[[ArXiv Version](https://arxiv.org/abs/2301.13392)]

## Usage
The model is a parallel binary linear model (BLM). Parameters of the BLM are shown as below:
<center>
    <img src="https://github.com/fengtony686/CCB-without-Graph/raw/main/results/blm.png" width="300"/>
</center>

If you want to compare regrets of BGLM-OFU-Unknown, BLM-LR-Unknown, UCB and $\epsilon$-greedy algorithms on this BLM, you need to run
```
python main.py
```
You can find our running samples in `./results/` directory.

## Contact

If you have any questions, feel free to contact us through email (shifeng-thu@outlook.com) or Github issues. Enjoy!
