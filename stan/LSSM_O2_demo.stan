data {
    int<lower=0> N;
    vector[N] y; // 1d measurement
    vector[N] u; // 1d input
}
parameters {
    real<lower=-1.0,upper=1.0> a11;       // state parameter
    real<lower=-1.0,upper=1.0> a12;       // state parameter
    real<lower=-1.0,upper=1.0> a21;       // state parameter
    real<lower=-1.0,upper=1.0> a22;       // state parameter
    real<lower=0.0> r;                  // measurement noise std
    real<lower=0.0> q11;                  // process noise std
    real<lower=0.0> q22;                  // process noise std
    // real b1;
    real b2;
    vector[N] z1;
    vector[N] z2;
}
model {
    // noise stds priors
    r ~ cauchy(0, 1.0);
    q11 ~ cauchy(0, 1.0); // noise on each state assumed independant
    q22 ~ cauchy(0, 1.0);

    // prior on parameter
    a11 ~ cauchy(0, 1.0);
    a12 ~ cauchy(0, 1.0);
    a21 ~ cauchy(0, 1.0);
    a22 ~ cauchy(0, 1.0); 

    // b1 ~ cauchy(0,1.0); // this is zero
    b2 ~ cauchy(0,1.0);

    // initial state prior
    z1[1] ~ normal(0,5);
    z2[1] ~ normal(0,0.05); // small prior on velocity (going to start the sim with zero speed every time)
    // state likelihood
    z1[2:N] ~ normal(a11*z1[1:N-1] + a12*z2[1:N-1], q11);
    z2[2:N] ~ normal(a21*z1[1:N-1] + a22*z2[1:N-1] + b2*u[1:N-1], q22); // input affects second state only
    // measurement likelihood
    y ~ normal(z1, r); // measurement of first state only

}