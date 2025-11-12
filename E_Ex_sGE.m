% Solver for the stationary general equilibrium in an economy with
% entry, exit, monopolistic competition and idisyncratic productivities
% described by a regular diffusion process.
% Solving for m, V(z), fz(z) simultaneously using the Newton-Raphson method.
% Solution for V satisfies an LCP, which is solved through introducing the
% complementary variable Vc and finding the root of sqrt(V^2+Vc^2) - V -
% Vc, as per a simplified version of the method from Fischer (1995)

function[Vars,errCode] = sGE_solver(z,VE,VEx,Phi,Kf,Kb,fzE,L,R,xi,uQ,varargin)
    Vars = [];
    errCode = 0;
    errTol = eps^0.4; iMax = 5e2;
    verbose = true;
    
    % Parsing optional parameters
    if(nargin<11) errCode=1; return;
    end
    
    % Productivity mesh density
    Iz = length(z); dz = (z(Iz)-z(1))/Iz;
    Vars = sparse(3*Iz+3,1);
    
    % Limits for M
    MU = (L*z(Iz)/xi/(VE*R + Phi)).^((xi-1)/(xi-2));
    ML = (L*z(1)/xi/(VE*R + Phi)).^((xi-1)/(xi-2));
    
    Vars=[ML;0;0;z-mean(z);z-mean(z);fzE]; %(mL*mU)^0.5

    if(nargin>=12) Vars=varargin{1}; end
    if(nargin>=13) errTol=varargin{2}; end
    if(nargin>=14) iMax=varargin{3}; end
    if(nargin>=15) verbose=varargin{4}; end
    
    % Initialisation
    M = Vars(1);
    MER = Vars(2);
    MExR = Vars(3);
    V = Vars(4:Iz+3);
    Vc = Vars(Iz+4:2*Iz+3);
    fz = Vars(2*Iz+4:3*Iz+3);
    
    metric=1; i=0; % Error norm and iteration counter
    
    % Jacobian - static component
    Js = [sparse(1,3), fzE'*dz, sparse(1,2*Iz);
          sparse(1,2*Iz+3), ones(1,Iz)*dz;
          sparse(1,2), 1, sparse(1,3*Iz);
          sparse(Iz,3), R*speye(Iz) - Kb, -speye(Iz), sparse(Iz,Iz);
          sparse(Iz,3), -speye(Iz), -speye(Iz), sparse(Iz,Iz);
          sparse(Iz,3*Iz+3)];
    Jd = sparse(3*Iz+3,3*Iz+3);
    
    if(verbose) tic
    end
    
    while true
        % Parsing the variable stack
        M = Vars(1);
        MER = Vars(2);
        MExR = Vars(3);
        V = Vars(4:Iz+3);
        Vc = Vars(Iz+4:2*Iz+3);
        fz = Vars(2*Iz+4:3*Iz+3);
        
        v = exp(uQ)*pmean(z,xi-1,M*fz.*Ind(V)*dz).^(2-xi)*L/xi.*z.^(xi-1) - Phi;
        
        % Error vector/Merit function
        err = [fzE'*V*dz - (VE-VEx); % Free entry condition
               sum(fz)*dz - 1; % Productivity distribution normalisation
               MExR - ... % Exit rate
                   ones(1,Iz) * ...
                   ((1-Ind(V)).*(Kf*fz + MER.*fzE))*dz;
               R*V + R*VEx - v - Kb*V - Vc; % Stationary HJB equation
               LCP_obj(V,Vc) - V - Vc; % LC condition
%                ((V-Vc).^2 - V.*abs(V) - Vc.*abs(Vc))/2;
               -(1-Ind(V)).*fz - ... % Stationary KF equation
                   Ind(V).*(Kf*fz + MER*fzE)];
        
        % Loop exit
        metric = max(abs(err)); i=i+1;
        if(verbose)
            fprintf('Iteration = %d; firm mass = %f; entry rate = %f;\n', i, Vars(1), Vars(2));
        end
        if(metric<=errTol || i>=iMax) break;
        end
        
        % Jacobian - dynamic component
        Jd = [sparse(3,3*Iz+3);
%               sparse(1,1), -(1-Ind(V))'*fzE*dz, sparse(1,1), ...
%                   ((Bz*fz + mER*fzE).*Ind_p1(V))'*dz, sparse(1,Iz), ...
%                   -(1-Ind(V))'*Bz*dz;
              (xi-2)/(xi-1)*(v+Phi)/M, sparse(Iz,2*Iz+2), ...
                  sparse(Iz,Iz);
                  %(xi-2)/(xi-1)*spdiag((v+Phi).*(z./pmean(z,xi-1,fz*dz)).^(xi-1)*dz);
              sparse(Iz,3), ...
                  spdiag(V./LCP_obj(V,Vc)), ...
                  spdiags(Vc./LCP_obj(V,Vc),0,Iz,2*Iz);
%               sparse(Iz,3), ...
%                   spdiag(V-Vc-abs(V)), ...
%                   spdiags(Vc-V-abs(Vc),0,Iz,2*Iz);
              sparse(Iz,1), -Ind(V).*fzE, sparse(Iz,1), ...
                  sparse(Iz,Iz), ... %spdiag((fz - Bz*fz - mER*fzE).*Ind_p1(V)), ...
                  sparse(Iz,Iz), ...
                  -speye(Iz) + spdiag(Ind(V))*(speye(Iz)-Kf)];
        
        Vars = Vars - (Js+Jd)\err;
    end
    
    if isnan(metric) || metric>errTol
        errCode=2;
    end
    
    if verbose
        if isnan(metric) || metric>errTol
            fprintf('--------\n');
            fprintf('The algorithm failed to converge in %d iterations!\n', i);
            toc
        else
            fprintf('--------\n');
            fprintf('The algorithm converged in %d iterations.\n', i);
            toc
            fprintf('\n\n');
        end
    end
end

%% Square power mean (Fisher-Marquart) LCP objective - used in setting up
% the error vector and the Jacobian in the implementation of
% the Newton-Raphson method
function[v] = LCP_obj(x,y)
    v = (x.^2 + y.^2).^0.5;
end
function[v] = LCP_obj_p1(x,y)
    v = x./(x.^2 + y.^2).^0.5;
end

%% Simple continuous indicator function
function[x] = Ind(v,eInd)
    if (nargin~=2 || isempty(eInd)) eInd = 1e8;
    end
    
    x = v.^2./(v.^2 + exp(-eInd*v));
end
function[x] = Ind_p1(v,eInd)
    eInd = 1e8;
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
