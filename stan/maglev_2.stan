// stan model for the maglev

functions{
    matrix process_model_vec(matrix z, row_vector u, vector theta, real g){
    // theta = [k0, I0]
    // x = [ball position from magnet, ball velocity]
        int pdims[2] = dims(z);
        matrix[pdims[1],pdims[2]] dz;
        real k0 = theta[2];
        real I0 = theta[1];
        // row_vector[pdims[2]] dz2 =
        dz[1,:] = z[2,:];
        dz[2,:] = (g - k0 * u .* u ./((z[1,:] + I0).*(z[1,:] + I0)));   // 100 gives conversion to cms
        return dz;
    }

    matrix rk4_update(matrix z, row_vector u, vector theta, real g, real Ts){
        int pdims[2] = dims(z);
        matrix[pdims[1],pdims[2]] k1;
        matrix[pdims[1],pdims[2]] k2;
        matrix[pdims[1],pdims[2]] k3;
        matrix[pdims[1],pdims[2]] k4;
        k1 = process_model_vec(z, u, theta, g);
        k2 = process_model_vec(z+Ts*k1/2, u, theta, g);
        k3 = process_model_vec(z+Ts*k2/2, u, theta, g);
        k4 = process_model_vec(z+Ts*k3, u, theta, g);
        return z + Ts*(k1/6 + k2/3 + k3/3 + k4/6);
    }

}

data {
    int<lower=0> no_obs;
    row_vector[no_obs] y;                // measurement [x, y, theta]
    row_vector[no_obs] u;               // input
    real<lower=0> Ts;                   // time step
    real g;                             // gravity
    vector[2] theta_p_mu;               // prior means on theta
    vector[2] theta_p_std;              // prior stds on theta
    // vector[2] z0_mu;                    // initial state prior
    // vector[2] z0_std;
    vector[1] r_p_mu;                   // measurement noise prior mean
    vector[1] r_p_std;                  // measurement noise prior std
    vector[1] q_p_mu;                   // process noise prior mean
    vector[1] q_p_std;                  // process noise prior std
}
parameters {
    row_vector[no_obs] w;                       // noise realisation
    matrix[2,1] init;
    vector<lower=1e-8>[2] theta;             // the parameters  [I0, k0]
    vector<lower=1e-8>[1] r;                // iid measurement noise student's t scale parameter
    // vector<lower=1.0>[1] df_r;           // df parameter
    vector<lower=1e-8>[1] q;                // iid input noise injection scale parameter
    // vector<lower=1.0>[1] df_q;                // df parameter
}
transformed parameters {
    matrix[2, no_obs+1] mu;
    mu[:,1:1] = init;
    for (kk in 1:no_obs)
        mu[:,kk+1:kk+1] = rk4_update(mu[:,kk:kk], u[kk:kk] + w[kk:kk], theta, g, Ts);
}
model {
    // df_r ~ gamma(2, 0.1); // according to https://jrnold.github.io/bayesian_notes/robust-regression.html
    // df_q ~ gamma(2, 0.1); // according to https://jrnold.github.io/bayesian_notes/robust-regression.html
    r ~ normal(r_p_mu, r_p_std);
    q ~ normal(q_p_mu, q_p_std);
    w[1:no_obs] ~ normal(0,q[1]);
    theta ~ normal(theta_p_mu, theta_p_std);
    y[1:no_obs] ~ student_t(3.0,mu[1,1:no_obs],r[1]);
}


