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
    vector[2] q_p_mu;                   // process noise prior mean
    vector[2] q_p_std;                  // process noise prior std
}
parameters {
    matrix[2,no_obs+1] h;                     // hidden states
    vector<lower=0.0>[2] theta;             // the parameters  [I0, k0]
    vector<lower=1e-8>[1] r;                 // independent measurement noise variances
    vector<lower=1e-8>[2] q;                 // independent process noise variances
}
transformed parameters {
    matrix[2, no_obs] mu;
    matrix[1, no_obs] yhat;
    // process model
    mu = rk4_update(h[:,1:no_obs], u[1:no_obs], theta, g, Ts); // this option was used for results in paper
    // measurement model
//    yhat[1,:] = h[1,1:no_obs];
}
model {
    r ~ normal(r_p_mu, r_p_std);
    q ~ normal(q_p_mu, q_p_std);

    // parameter priors
    theta ~ normal(theta_p_mu, theta_p_std);

    // initial state prior (Not using this / not needed)
    // h[:,1] ~ normal(z0_mu, z0_std);

    // independent process likelihoods
    h[1,2:no_obs+1] ~ normal(mu[1,:], q[1]);
    h[2,2:no_obs+1] ~ normal(mu[2,:], q[2]);

    // independent measurement likelihoods
    y[:] ~ normal(h[1,1:no_obs], r[1]);
}


