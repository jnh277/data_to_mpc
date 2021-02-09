data {
    int<lower=0> N; // time horizon in steps
    int<lower=0> O; // order / state count
    int<lower=0> D; // number of measurements
    matrix[D,N] y; // 2d measurement
    matrix[1,N] u; // 1d input
    real T; // timestep length!
}
parameters {
    real<lower=0.0> m;       // mass parameter
    real<lower=0.0> k;       // spring parameter
    // real<upper=0.0> k;    // unstalbe spring parameter
    real<lower=0.0> b;       // damper parameter
    vector<lower=0.0>[D] r;  // measurement noise stds
    vector<lower=0.0>[O] q;  // process noise stds
    matrix[O,N] z;           // states
}
// transformed parameters {
//     matrix[O,O] A;
//     matrix[O,1] B;
//     A[1,1] = 0;
//     A[1,2] = T;
//     A[2,1] = -T*k/m;
//     A[2,2] = -T*b/m;
//     B[1,1] = 0;
//     B[2,1] = T/m;
// }

model {
    // noise stds priors (i think these will draw them from the )
    r ~ cauchy(0, 1.0);
    q ~ cauchy(0, 1.0); // noise on each state assumed independant

    // prior on parameters
    m ~ cauchy(0, 1.0);
    k ~ cauchy(0, 1.0);
    b ~ cauchy(0, 1.0);

    // initial state prior
    z[1,1] ~ normal(0,5);
    z[2,1] ~ normal(0,0.05); // small prior on velocity (going to start the sim with zero speed every time)
   
    // state likelihood (apparently much better to do univariate sampling twice)
    z[1,2:N] ~ normal(T*z[2,1:N-1], q[1]);
    z[2,2:N] ~ normal(-(k*T/m)*z[1,1:N-1] + -(b*T/m)*z[2,1:N-1] + (T/m)*u[1,1:N-1], q[2]); // input affects second state only
    // measurement likelihood
    y[1,:] ~ normal(z[1,:], r[1]); // measurement of first state only
    y[2,:] ~ normal(-(k/m)*z[1,1:N] - (b/m)*z[2,1:N] + u[1,:]/m, r[2]); // acceleration measurement?
}