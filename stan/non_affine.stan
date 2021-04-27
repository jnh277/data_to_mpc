data {
    int<lower=0> N;
    vector[N] y;
    vector[N] u;
    vector[4] prior_mu;     // prior means of parameters stored a, b, q, r
    vector[4] prior_std;    // prior stds of parameters
    real prior_state_mu;    // first timestep state prior mean
    real prior_state_std;   // first timestep state prior std
}
parameters {
    real<lower=-1.0,upper=1.0> a;       // state parameter
    real<lower=0.0> r;                  // measurement noise std
    real<lower=0.0> q;                  // process noise std
    real b;
//    real z0;
    vector[N] z;
}
//transformed parameters {
//    vector[N] z;
//    z[1] = z0;
//    for (n in 2:N) {
//        z[n] = a*z[n-1] + b * sin(u[n-1]) + e[n-1];
//    }
//
//
//}

model {
    // noise stds priors
    q ~ normal(prior_mu[3], prior_std[3]);
    r ~ normal(prior_mu[4], prior_std[4]);

    // prior on parameter
    a ~ normal(prior_mu[1], prior_std[1]);
    b ~ normal(prior_mu[2], prior_std[2]);

    // initial state prior
    z[1] ~ normal(0,5);
    // state likelihood
//    e ~ lognormal(-3, q);
    z[2:N] ~ uniform(a*z[1:N-1] + b * sin(u[1:N-1])-q, a*z[1:N-1] + b * sin(u[1:N-1])+q);

    // measurement likelihood
    y ~ normal(z, r);

}