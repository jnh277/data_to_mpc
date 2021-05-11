data {
    int<lower=0> N;
    vector[N] y;
    vector[N] u;
    vector[4] prior_mu;     // prior means of parameters stored a, b, q, r
    vector[4] prior_std;    // prior stds of parameters
    real prior_state_mu;    // first timestep state prior mean
    real prior_state_std;   // first timestep state prior std
    real <lower=0.1> nu;    // student t degrees of freedom
}
parameters {
    real<lower=-1.0,upper=1.0> a;       // state parameter
    real<lower=0.0> r;                  // measurement noise std
    real<lower=0.0> q;                  // process noise std
    real b;
    vector<lower=1e-8>[N] z;
}


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
    z[2:N] ~ normal(a*z[1:N-1] + b * sin(u[1:N-1]), z[1:N-1]*q);

    // measurement likelihood
    y ~ student_t(nu ,z, r);

}