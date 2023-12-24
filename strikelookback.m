% Fixed strike lookback call option
% Uses Monte Carlo with antithetic variates
randn('state',100)
%%%%%% Problem and method parameters %%%%%%%%%
S=1; E=1; sigma = 0.3; r = 0.05; T = 1;
Dt = 1e-3;N= T/Dt;M= 1e4;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
V = zeros(M,1);
Vanti = zeros(M,1);
for i = 1:M
samples = randn(N,1);
% standard Monte Carlo
Svals = S*cumprod(exp((r-0.5*sigma^2)*Dt+sigma*sqrt(Dt)*samples));
Smax = max(Svals)
V(i) = max(Smax-E,0);
% antithetic path
Svals2 = S*cumprod(exp((r-0.5*sigma^2)*Dt-sigma*sqrt(Dt)*samples));
Smax2 = max(Svals2)
V2=0
V2 = max(Smax2-E,0);
Vanti(i) = 0.5*(V(i) + V2);
end
aM = mean(V); bM = std(V);
conf = [aM - 1.96*bM/sqrt(M), aM + 1.96*bM/sqrt(M)]
aManti = mean(Vanti); bManti = std(Vanti);
confanti = [aManti - 1.96*bManti/sqrt(M), aManti + 1.96*bManti/sqrt(M)]