/*
###############################################################################
#    Practical Bayesian Linear System Identification using Hamiltonian Monte Carlo
#    Copyright (C) 2020  Johannes Hendriks < johannes.hendriks@newcastle.edu.a >
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
###############################################################################
*/
// stan model for the QUBE pendulum

functions{
//    real matrix_normal_lpdf(matrix y, matrix mu, matrix LSigma){
//        int pdims[2] = dims(y);
//        matrix[pdims[1],pdims[2]] error_sc = mdivide_left_tri_low(LSigma, y - mu);
//        real p1 = -0.5*pdims[2]*(pdims[1]*log(2*pi()) + 2*sum(log(diagonal(LSigma))));
//        real p2 = -0.5*sum(error_sc .* error_sc);
//        return p1+p2;
//    }
    matrix process_model_vec(matrix z, row_vector u, vector theta, real g){
    // theta = [k0, I0]
    // x = [ball position from magnet, ball velocity]
        int pdims[2] = dims(z);
        matrix[pdims[1],pdims[2]] dz;
        real k0 = theta[1];
        real I0 = theta[2];
        row_vector[pdims[2]] dz2 = g - k0 * u/((I0 + z[1,:])^2)

        dz[1,:] = z[2,:];
        dz[2,:] = dz2;
        return dz;
    }

    matrix rk4_update(matrix z, row_vector u, vector theta, real g){
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
    matrix[3, no_obs] y;                // measurement [x, y, theta]
    row_vector[no_obs] u;               // input
    real<lower=0> Ts;                   // time step
    real g;                             // gravity
    vector[2] theta_p_mu;               // prior means on theta
    vector[2] theta_p_std;              // prior stds on theta
//    vector[4] z0_mu;                    // initial state prior
//    vector[4] z0_std;
    vector[1] r_p_mu;                   // measurement noise prior mean
    vector[1] r_p_std;                  // measurement noise prior std
    vector[2] q_p_mu;                   // process noise prior mean
    vector[2] q_p_std;                  // process noise prior std
//    vector[4] h0_mu;                    // prior on initial state
//    vector[4] z0;                       // initial state guess
}
parameters {
    matrix[2,no_obs+1] h;                     // hidden states
    vector<lower=0.0>[2] theta;             // the parameters  [k0, I0]
//    vector[4] z0;                           // initial state guess
    vector<lower=1e-8>[1] r;                 // independent measurement noise variances
    vector<lower=1e-8>[2] q;                 // independent process noise variances

}
transformed parameters {
    matrix[2, no_obs] mu;
    matrix[1, no_obs] yhat;
    // process model
    mu = rk4_update(h[:,1:no_obs], u[1:no_obs], theta, g); // this option was used for results in paper
    // measurement model
    yhat[1,:] = h[1,1:no_obs];
}
model {
    r ~ normal(r_p_mu, r_p_std);
    q ~ normal(q_p_mu, q_p_std);

    // parameter priors
    theta ~ normal(theta_p_mu, theta_p_std);

    // initial state prior (Not using this / not needed)
//    h[:,1] ~ normal(z0_mu, z0_std);      //

    // independent process likelihoods
    h[1,2:no_obs+1] ~ normal(mu[1,:], q[1]);
    h[2,2:no_obs+1] ~ normal(mu[2,:], q[2]);

    // independent measurement likelihoods
    y[1,:] ~ normal(yhat[1,:], r[1]);

}
//generated quantities {
//    cholesky_factor_cov[7] L;
//    real loglikelihood;
//    L = diag_pre_multiply(tau,Lcorr);
////    loglikelihood = matrix_normal_lpdf(append_row(h,y) | mu_c, diag_pre_multiply(tau,Lcorr));
//
//}


