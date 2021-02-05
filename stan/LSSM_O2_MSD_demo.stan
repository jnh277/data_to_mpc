data {
    int<lower=0> N;
    int<lower=0> O;
    int<lower=0> D;
    matrix[D,N] y; // 2d measurement
    matrix[1,N] u; // 1d input
    real T;
}
parameters {
    real<lower=0.0> m;       // mass parameter
    real<lower=0.0> k;       // spring parameter
    // real<upper=0.0> k;    // unstalbe spring parameter
    real<lower=0.0> c;       // damper parameter
    real<lower=0.0> r;       // measurement noise std
    vector[D]<lower=0.0> q;     // process noise std
    real<lower=0.0> q22;     // process noise std
    matrix[O,N] x;
}
model {
    // noise stds priors
    r ~ cauchy(0, 1.0);
    q ~ cauchy(0, 1.0); // noise on each state assumed independant

    // prior on parameter
    m ~ cauchy(0, 1.0);
    k ~ cauchy(0, 1.0);
    c ~ cauchy(0, 1.0);

    // initial state prior
    z1[1] ~ normal(0,5);
    z2[1] ~ normal(0,0.05); // small prior on velocity (going to start the sim with zero speed every time)
    // state likelihood
    z1[2:N] ~ normal(T*z2[1:N-1], q11);
    z2[2:N] ~ normal(a21*z1[1:N-1] + a22*z2[1:N-1] + b2*u[1:N-1], q22); // input affects second state only
    // measurement likelihood
    y ~ normal(z1, r); // measurement of first state only

}