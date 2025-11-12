clearvars;
close all; clc;

%% Parameters
% Cost parameters
VE = 80; % entry cost
VEx = 50; % scrap value
Phi = 5; % fixed costs/contunuation value

% Households
rho = 0.04; % discount factor
theta = 1.5; % 1/elasticity of intertemporal substitution
L = 1e3; % labour force

% Market environment
xi = 4.13396765028375; % elasticity of substitution

% Technology diffusion parameters
mu1 = 0.01; % drift
mu2 = 0; % OU coefficient, e^(-mu2) = rho in AR(1)
sigma = 0.02; % diffusion
% gBM = true; % gBM/gOU switch - replaced with a more general formula

% Productivity space
Iz = 400; zmin = 1e-3; zmax = 1; % grid density, limits for productivity
z = linspace(zmin,zmax,Iz)'; % productivity grid
dz = (zmax-zmin)/Iz;

% Entry density parameters
muE = -0.8*zmax;
sigmaE = abs(muE/2);

%% Functions

p = @(z,Z) Z./z; % function for the cross section of prices
y = @(z,Z) Z*L .* (z./Z).^xi; % function for the cross section of outputs

% Entrant productivity distribution densities: fzE_LN - log-normal, fzE_G -
% gamma-distribution
fzE_LN = @(x) 1/(2*pi).^0.5/sigmaE./x .* exp(-((log(x)-muE).^2)/2/sigmaE^2);
fzE_G = @(x) (muE/sigmaE^2)^((muE/sigmaE)^2)/gamma((muE/sigmaE)^2) * ...
    x.^((muE/sigmaE)^2 - 1) .* exp(-muE.*x/sigmaE^2);

%% Initialisation
R = rho; % rate of return on creating a firm

% Density of the entrant firm mass distributions
fzE = fzE_LN(z);
fzE = fzE/sum(fzE)/dz;
fzE(Iz) = 1/dz - sum(fzE(1:Iz-1));

% Aggregate productivity shock (MIT, at least for now)
uQ = 0;

%% KB (HJB) and KF operator matrices
% Matrix for the first derivative
d1Mx = spdiags([(-mu1 - mu2*log(z) + sigma^2/2 > zeros(Iz,1)), ...
            (-mu1 - mu2*log(z) + sigma^2/2 < zeros(Iz,1)) - (-mu1 - mu2*log(z) + sigma^2/2 > zeros(Iz,1)), ...
            -(-mu1 - mu2*log(z) + sigma^2/2 < zeros(Iz,1))],-1:1,Iz,Iz);
d1Mx = d1Mx';
d1Mx(1,1) = (-mu1 - mu2*log(z(1)) + sigma^2/2 < 0) - (-mu1 - mu2*log(z(1)) + sigma^2/2 > 0);
d1Mx(Iz,Iz) = (-mu1 - mu2*log(z(Iz)) + sigma^2/2 < 0) - (-mu1 - mu2*log(z(Iz)) + sigma^2/2 > 0);

% Matrix for the second derivative
d2Mx = spdiags(repmat([1 -2 1],Iz,1),-1:1,Iz,Iz);
d2Mx(1,1) = -1; d2Mx(Iz,Iz) = -1;

Kb = spdiag(-(mu1 + mu2*log(z) - sigma^2/2).*z)*d1Mx/dz + sigma^2/2*spdiag(z.^2)*d2Mx/dz^2;

% KF operator for the productivity distribution
Kf = Kb';
clear d1Mx d2Mx;

%% Solution - stationary equilibrium
fprintf('Calculating the stationary equilibrium...\n\n')
[Vars,errCode] = E_Ex_sGE(z,VE,VEx,Phi,Kf,Kb,fzE,L,R,xi,uQ);

%% Parsing the variable stack, calculating aggregates

M = Vars(1);
MER = Vars(2);
MExR = Vars(3);
V = Vars(4:Iz+3);
Vc = Vars(Iz+4:2*Iz+3);
fz = Vars(2*Iz+4:3*Iz+3);
fzMap = griddedInterpolant(z,fz,'makima');

% Productivity composite
Z = pmean(z,xi-1,fz*dz);

% Output, consumption
Y = exp(uQ)*M^(1/(xi-1))*Z(1)*L;
C = Y - (MER*VE + Phi - MExR*VEx)*M;

%% Distributions and aggregates

ixEx = find(Ind(V) > 1e-6, 1) - 1; % 

ps = p(z, M^(1/(xi-1))*Z); ps(1:ixEx) = sparse(ixEx,1);
ys = y(z, M^(1/(xi-1))*Z); ys(1:ixEx) = sparse(ixEx,1);

%% Stationary distribution - graphs
figure;
tiledlayout(2, 2, "Padding", "tight", "TileSpacing", "tight");

nexttile;
plot(z,V+VEx,'linewidth',1.2,'color',[1 0.3 0]);
line([0 z(end)],[VEx VEx],'linestyle','--','color',[0 0 0]);
ylim([0 inf]);

xlabel('$z$','interpreter','latex');
ylabel('$V\!\left(z\right)$','interpreter','latex');
title('Value function~$V\!\left(z\right)$', ...
    'interpreter','latex','fontsize',12);

nexttile; hold on;
plot(z(ixEx+1:Iz),ys(ixEx+1:Iz),'linewidth',1.2,'color',[1 0.3 0]);
line([z(1) z(ixEx+1)],[0 0],'linewidth',1.2,'color',[1 0.3 0]);
line([z(ixEx+1) z(ixEx+1)],[0 ys(ixEx+1)], ...
    'linestyle','--','color',[1 0.3 0]);
hold off;

xlabel('$z$','interpreter','latex');
ylabel('$y\!\left(z\right)$','interpreter','latex');
title('Output~$y\!\left(z\right)$','interpreter','latex','fontsize',12);

nexttile;
% Unused so far

nexttile; hold on;
plot(z(ixEx+1:Iz),ys(ixEx+1:Iz)./z(ixEx+1:Iz),'linewidth',1.2,'color',[1 0.3 0]);
line([z(1) z(ixEx+1)],[0 0],'linewidth',1.2,'color',[1 0.3 0]);
line([z(ixEx+1) z(ixEx+1)],[0 ys(ixEx+1)/z(ixEx+1)], ...
    'linestyle','--','color',[1 0.3 0]);
hold off;

xlabel('$z$','interpreter','latex');
ylabel('$l\!\left(z\right)$','interpreter','latex');
title('Labour~$l\!\left(z\right)$', ...
    'interpreter','latex','fontsize',12);

figure;
tiledlayout(2, 2, "Padding", "tight", "TileSpacing", "tight");

nexttile; hold on;
plot(z,fz,'linewidth',1.2,'color',[1 0.3 0]);
plot(z,fzE_LN(z),'linestyle','--','color',[0 0 0]); hold off;

xlabel('$z$','interpreter','latex');
ylabel('$f_m\!\left(z\right)$','interpreter','latex');
title('Productivity distribution~$f_m\!\left(z\right)$', ...
    'interpreter','latex','fontsize',12);

nexttile; hold on;
plot(log(ys),log(abs(fz)), ...
    'linewidth',1.2,'color',[1 0.3 0]);
xlim([0.99*log(ys(ixEx+1)) 1.01*log(ys(Iz))]);

xlabel('$\ln{y\!\left(\tilde{z}\right)}$','interpreter','latex');
ylabel('$\ln{g_m\!\left(\tilde{z}\right)}$','interpreter','latex');
title('Size distribution~(output,~$\ln$--$\ln$)', ...
    'interpreter','latex','fontsize',12);

nexttile;
% Unused so far

nexttile; hold on;
plot(log(ys./z),log(abs(fz)), 'linewidth',1.2,'color',[1 0.3 0]);
xlim([0.99*log(ys(ixEx+1)./z(ixEx+1)) 1.01*log(ys(Iz)./z(Iz))]);
hold off;

xlabel('$\ln{l\!\left(\tilde{z}\right)}$','interpreter','latex');
ylabel('$\ln{g_m\!\left(\tilde{z}\right)}$','interpreter','latex');
title('Size distribution~(employment,~$\ln$--$\ln$)', ...
    'interpreter','latex','fontsize',12);

%% Power mean (a.k.a. Hölder mean)
function[x] = pmean(v,pwr,varargin)
    if sum(v<0)>0 x=NaN; return;
    end
    
    wgts=1/length(v); % weights, all equal 1/(v's length) by default
    if ~isempty(varargin) && isnumeric(varargin{1})
        if  length(varargin{1})==1 || length(varargin{1})==length(v)
            wgts=varargin{1};
        end
    end
    
    [rv,~] = size(v);
    
    x = repmat(sum(wgts.*(v.^pwr)).^(1/pwr),rv,1);
end

%% Simple continuous indicator function
function[x] = Ind(v,eInd)
    if (nargin~=2 || isempty(eInd)) eInd = 1e8;
    end
    
    x = v.^2./(v.^2 + exp(-eInd*v));
end
function[x] = Ind_p1(v,varargin)
    if (nargin~=2 || isempty(eInd)) eInd = 1e8;
    end
    
    x = sparse(size(v,1),size(v,2)); %(2./x+eInd).*sgn(x)./(x.^2.*exp(eInd*x) + 1);
end

%% Simple sparse diagonalisation
function[x] = spdiag(v)
    [rv,cv] = size(v);
    
    if (cv==1)
        x = spdiags(v,0,rv,rv);
        return;
    elseif (rv==1)
        x = spdiags(v',0,cv,cv);
        return;
    else x=v; return;
    end
end

%% Slice diagonalisation (row-wise or column-wise)
function[x] = slcdiag(v,dim)
    if (dim==1)
        [rv,cv] = size(v); vl = v;
        x = sparse((1:numel(v))', ...
            kron((1:cv)',ones(rv,1)), ...
            vl(:));
        return;
    elseif (dim==2)
        [rv,cv] = size(v); vl = v';
        x = sparse(kron((1:rv)',ones(cv,1)), ...
            (1:numel(v))', ...
            vl(:));
        return;
    else x=v; return;
    end
end

%% Exclusion of shifts in the exit region
function[x] = exclIndices(initIxs, ixMax, step)
    if length(initIxs)==0 x=[]; return;
    end
    
    x=sparse(ceil(ixMax/step)*length(initIxs),1);
    currLen=1;
    for i=1:length(initIxs)
        x(currLen:currLen+ceil((ixMax-initIxs(i))/step)-1) = (initIxs(i):step:ixMax)';
        currLen = currLen + length(initIxs(i):step:ixMax) - 1;
    end
    
    x=x(1:currLen);
    x=sort(x);
end
