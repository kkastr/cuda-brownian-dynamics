# Brownian dynamics simulation written in CUDA

<div align="center">
<img src="./plots/bd_anim.gif">
</div>

Brownian motion is the result of stochastic fluctuations in a given quantity. This type of motion has rich mathematical structure and is ubiquitous in many fields of study such as physics, biology, economics, and social science.

In physics for instance, particles undergoing Brownian motion (Brownian particles) are often used as models for biological processes as they are intimately linked to the concept of diffusion. Mathematically exploring the diffusion of Brownian particles yields a method for computing their diffusion coefficient which, loosely speaking, denotes the "spread" in the particles position within a given system affording inside on the physical behaviour of said system.

This repo contains simulation code for particles evolving with Brownian dynamics inside of a box, written in CUDA. In physics, we often use Brownian particles in complex geometries as models for complex biological systems and this implemenation allows for simulating many thousands of these particles at once, in addition to any further interactions that may be needed for the physical system under investigation.

<div align="center">
<img src="./plots/msd.png" height="335">
</div>

The figure above is a typical measure of Brownian motion, the mean square displacement (MSD) of the particles. Theoretically, the MSD can be used to show that the displacement of Brownian particles is proportional to the square root of the elapsed time. In a standard setup, in which all relevant simulation quantities (temperature etc.) are set to unity, the simulation diffusion coefficient should also be unity, as is expected from theory. As can be seen, the simulation yields good agreement with the theoretical result for the diffusion of Brownian particles, indicating that the particles in the simulation are indeed undergoing Brownian motion.

## Code

To use this code you will need to have CUDA and `nvcc` installed on your machine. Once these requirements are met, you can compile on the terminal:

```bash
nvcc bd-sim.cu
```
