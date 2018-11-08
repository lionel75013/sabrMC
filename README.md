# sabrMC
Unbiased SABR model simulation in the manner of Bin Chen, Cornelis W. Oosterlee and Hans van der Weide (2011).

The Sigma Alpha Beta Rho model (SABR) first designed by Hagan & al. is very popular and used extensively by practitioners
for interest rates derivatives. In this framework, volatility is stochastic following a geometric brownian motion
with no drift, whereas the forward rate dynamics are modeled with a CEV process. However, despite the simplicity of its formulation, 
it does not allow for exact closed form analytical solutions. As a matter of fact,  Hagan used singular perturbation techniques to obtain the price of european option under the SABR model.

Moreover as pointed by early authors Andersen (1995) and Andersen & Andreasen (2000) Euler-Maruyama and Milstein discretization scheme 
are biased for the CEV process, and monte carlo simulations will exhibit significant bias even with a high number of simulated paths.

Chen & al. (2011) extend the methodologies of Willard (1997), Broadie & Kaya (2006), Andersen (2008)and  Islah (2009) to provide an unbiased
scheme to simulate and discretize the SABR process. This method is a mix of  multiple techniques: a direct inversion scheme of the non central
 chi-squared distribution, the QE method of Andersen and small disturbance expansion. More information on the theory and results can be found in the article: [Monte Carlo Simulation of the SABR process](http://underaudit.com/2018/01/30/monte-carlo-simulation-of-the-sabr-process/).

The implementation I have provided, tries to vectorize the problem as much as possible, but some amount of iteration is required when dealing
with the conditional application of the QE scheme or direct inversion. It also does not implement the so-called "Enhanced direct inversion procedure"
of formula (3.12). Nor does the direct inversion scheme use a newton type root finder.  I leave this for a later time.



Implementing the details of a research paper is hard and subject to interpretation error. As such, this code is for educational purpose only, and it might still undergo some changes/corrections in the future.

Any comments or insight on the code is very welcome.

References
----------
 * "Efficient Techniques for Simulation of Interest Rate Models Involving Non-Linear Stochastic Differential Equations"
   Leif B. G. Andersen (1995)
 * "Volatility skews and extensions of the libor market model"
   L. Andersen, J. Andreasen (2000)
 * "Managing Smile Risk",
   Patrick S. Hagan, Deep Kumar, Andrew S. Lesniewski,and Diana E. Woodward (2002)
 * "Efficient simulation of the heston stochastic volatility model"
   Andersen L. Journal of Computational Finance 11:3 (2008) 1–22.
 * "Simulation of the CEV process and the local martingale property."
	A. E. Lindsay, D. R. Brecher (2010)
 * "Efficient unbiased simulation scheme for the SABR stochastic volatility model"
       Bin Chen, Cornelis W. Oosterl, Hans van der Weide (2011)

Usage
-----
This package requires numpy and scipy. Its main method is the sabrMC function which contains all relevant parameters.
 
Attribution
-----------
L. Ouaknin (2018)
