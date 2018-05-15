function[Sigmaout] = divideconquer(Y,g,k,BURNIN,MCMC,thin,rho)
%% Function to implement the divide and conquer algorithm (https://arxiv.org/pdf/1612.02875.pdf)
%% High dimensional Bayesian Factor Models %%

%% Written by Gautam Sabnis (gautamsabnis15@gmail.com) and Debdeep Pati (debdeep.isi@gmail.com) %% 


%% Model: Y = \Lambda\eta + \epsilon, \epsilon \sim N(0, \Omega) %%
%%        \Lambda ~ MGPS shrinkage prior (Sparse Bayesian Infinite Factor Models, Bhattacharya & Dunson 2011)%%
%% Hierarchical model on latent factors: \eta = \sqrt(\rho)X + \sqrt(1 - \rho)Z(:,:,m), X, Z(:,:,m) \sim N(0,I) %% 
%%        X: Impure factors, Z(:,:,m): Machine/group specific pure factor    %% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Input: Y = Observation, a p*1 vector %%
%%        g = number of groups %%
%%        k = number of factors %%
%%        BURNIN = Number of burnin MCMC samples %%
%%        MCMC = Number of posterior draws to be saved %%
%%        thin = thinning parameter of the chain %%
%%        rho = correlation %%


%% Output: Sigmaout = p x p Estimated Covariance Matrix %%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


tic;
[n,p] = size(Y);
nnzcol = zeros(p,1); 
for j = 1:p
    nnzcol(j) = nnz(Y(:,j)); %non zero entries in each column
end

zerocol = find(~nnzcol); %zero columns

Y = Y(:,setdiff(1:p,zerocol)); %remove the zero columns

P = p/g; K = k/g;
Sigmaout = zeros(p,p); 

%-----define global constants-----%
N = BURNIN + MCMC; effsamp = (N - BURNIN)/thin; 


%-----Randomly divide Y across dimensions to create Yd-----%
Yd = zeros(n,P,g); 
varind = randperm(p);

for m = 1:g
    Yd(:,:,m) = Y(:,varind((m-1)*P + 1:m*P));
end

%----Scale and center data----%
Md = mean(Yd); VYd = var(Yd); 
Yd = bsxfun(@minus, Yd, Md); 
Yd = bsxfun(@times, Yd, 1./sqrt(VYd));

%-----Define hyperparameter values-----%
as = 1; bs = 0.3; % gamma hyperparameters for residual precision
df = 3;           % gamma hyperparameters for t_{ij}
ad1 = 2; bd1 = 1; % gamma hyperparameters for delta_1
ad2 = 2; bd2 = 1; % gamma hyperparameters delta_h, h >= 2


%----Initial values----%
ps = gamrnd(as,1/bs,[P,1,g]);
Lambda = zeros(P,K,g); eta = zeros(n,K,g);
Z = zeros(n,K,g); X = normrnd(0,1,[n,K]); 

psijh = gamrnd(df/2,2/df,[P,K,g]); %local shrinkage coefficients
delta = zeros(K,1,g); 
Omega = zeros(P,P,g);   %Sigma = diagonal residual covariance
tauh = zeros(K,1,g); 
Plam = zeros(P,K,g);    %precision of factor loadings

for m = 1:g
    Z(:,:,m) = normrnd(0,1,[n,K]); 
    eta(:,:,m) = sqrt(rho)*X + sqrt(1 - rho)*Z(:,:,m);
    delta(:,1,m) = ...
        [gamrnd(ad1,bd1);gamrnd(ad2,bd2,[K-1,1])];  %global shrinkage coefficients multipliers
    Omega(:,:,m) = diag(ps(:,1,m)); 
    tauh(:,1,m) = cumprod(delta(:,1,m));            %global shrinkage coefficients
    Plam(:,:,m) = bsxfun(@times, psijh(:,:,m),tauh(:,:,m)'); %precision of loading rows 
end
%------start Gibbs Sampling------%

for iter = 1:N
    
    
    %Update eta|rest. Two step process.
    
    %-----I) Update Z|rest -----%
    
    for m = 1:g
        Zmsg = bsxfun(@times, Lambda(:,1:K,m),diag(Omega(:,:,m)));
        Zprec = eye(K) + (1 - rho)*Zmsg'*Lambda(:,1:K,m); 
        Lz = cholcov(Zprec); 
        for i = 1:n
            Rz = Yd(i,:,m)' - sqrt(rho)*Lambda(:,1:K,m)*X(i,1:K)';
            bz = sqrt(1 - rho)*Zmsg'*Rz; 
            vz = Lz\bz; mz = Lz'\vz; zz = normrnd(0,1,[K,1]);
            yz = Lz'\zz; 
            Z(i,1:K,m) = mz + yz; 
        end
    end
    
    
    %-----II) Update X|rest ------%
    sumx1 = 0;
    for m = 1:g
        Xmsg = bsxfun(@times, Lambda(:,1:K,m),diag(Omega(:,:,m)));
        sumx1 = sumx1 + Xmsg'*Lambda(:,1:K,m); 
    end
    Xprec = g*eye(K) + rho*sumx1;
    Lx = cholcov(Xprec); 
    for i = 1:n
        sumx2 = 0;
        for m = 1:g
            Rx = Yd(i,:,m)' - sqrt(1 - rho)*Lambda(:,1:K,m)*Z(i,1:K,m)';
            sumx2 = sumx2 + bsxfun(@times, Lambda(:,1:K,m), diag(Omega(:,:,m)))'*Rx;
        end
        bx = sqrt(rho)*sumx2;
        vx = Lx\bx; mx = Lx'\vx; zx = normrnd(0,1,[K,1]);
        yx = Lx'\zx;
        X(i,1:K) = mx + yx; 
    end
    
    %-----Update eta|rest using I) and II)-----%
    for m = 1:g
        eta(:,1:K,m) = sqrt(rho)*X(:,1:K) + sqrt(1-rho)*Z(:,1:K,m);
    end
    
    %-----Update Lambda|rest-----%
    for m = 1:g
        eta2 = eta(:,1:K,m)'*eta(:,1:K,m); 
        
        for j = 1:P
            Qlam = diag(Plam(j,1:K,m)) + ps(j,:,m)*eta2; blam = ps(j,:,m)*(eta(:,1:K,m)'*Yd(:,j,m));
            Llam = chol(Qlam,'lower'); zlam = normrnd(0,1,K,1); 
            vlam = Llam\blam; mlam = Llam'\vlam; ylam = Llam'\zlam; 
            Lambda(j,1:K,m) = (ylam + mlam)';
        end
    end
    
    %-----Update psi_jh|rest-----%
    for m = 1:g
        psijh(:,1:K,m) = gamrnd(df/2 + 0.5, 1./(df/2 + 0.5*bsxfun(@times, Lambda(:,1:K,m).^2, ...
            tauh(1:K,:,m)')));
    end
    
    %-----Update delta|rest & tauh|rest-----%
    for m = 1:g
        mat = bsxfun(@times,psijh(:,1:K,m),Lambda(:,1:K,m).^2);
        ad = ad1 + 0.5*P*K; bd = bd1 + 0.5*(1/delta(1,:,m))*sum(tauh(1:K,:,m).*sum(mat)');
        delta(1,:,m) = gamrnd(ad,1/bd); tauh = cumprod(delta); 
        
        for h = 2:K
            ad = ad2 + 0.5*P*(K-h+1); bd = bd2 + 0.5*(1/delta(h))*sum(tauh(h:end,:,m).*...
                sum(mat(:,h:end))');
            delta(h,:,m) = gamrnd(ad,1/bd); tauh(:,:,m) = cumprod(delta(:,:,m));
        end
    end
    
    %----Update Sigma|rest----%
    for m = 1:g
        Ytil = Yd(:,:,m) - eta(:,:,m)*Lambda(:,:,m)';
        ps(:,:,m) = gamrnd(as + 0.5*n, 1./(bs + 0.5*sum(Ytil.^2)));
        Omega(:,:,m) = diag(1./ps(:,:,m)); 
    end
    
    %----Update precision parameters----%
    for m = 1:g
        Plam(:,:,m) = bsxfun(@times, psijh(:,:,m),tauh(:,:,m)');
    end
    
    
    if mod(iter, thin)== 0 && iter > BURNIN
                    
                    for rind = 1:g
                        for cind = 1:g
                            
                            if rind == cind
                                Sigma((rind - 1)*P + 1:rind*P,(cind-1)*P + 1:cind*P) = Lambda(:,:,rind)*Lambda(:,:,rind)' + Omega(:,:,rind); 
                                
                            else
                                Sigma((rind - 1)*P + 1:rind*P,(cind-1)*P + 1:cind*P) = rho*Lambda(:,:,rind)*Lambda(:,:,cind)';
                            end
                        end
                    end
    
                    Sigmaout = Sigmaout + Sigma/effsamp;
                    Sigmaout = (Sigmaout + Sigmaout')/2;
    end
end
           
        
time = toc; 
fprintf('Execution time %d Gibbs iteration with (n,p) = (%d,%d) is %f seconds',N,n,p,time);

        
            
