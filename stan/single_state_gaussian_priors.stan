data {
    int<lower=0> N;
    vector[N] y;
    vector[N] u;
}
parameters {
    real<lower=-1.0, upper=1.0> a;       // state parameter constrained stable
    real c;
    // real<lower=1E-8> r;                  // measurement noise std
    // real<lower=1E-8> q;                  // process noise std
    real b;
    // vector[N] z;
}
model {
    vector[N] z;
    // state likelihood
    z[2:N] = a*z[1:N-1] + b*u[1:N-1];
    // measurement likelihood
    y ~ normal(c*z,1);
}