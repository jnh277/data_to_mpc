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
//
//    }
    matrix process_model_vec(matrix z, row_vector u, vector theta, real Lr, real Mp, real Lp, real g){
    // theta = [Jr, Jp, Km, Rm, Dp, Dr]
    // x = [arm angle, pendulum angle, arm angle velocity, pendulum angle velocity]
        int pdims[2] = dims(z);
        matrix[pdims[1],pdims[2]] dz;

        real Jr = theta[1];
        real Jp = theta[2];
        real Km = theta[3];
        real Rm = theta[4];
        real Dp = theta[5];
        real Dr = theta[6];

        row_vector[pdims[2]] m11 = Jr + Mp * Lr^2 + 0.25 * Mp * Lp^2 - 0.25 * Mp * Lp^2 * (cos(z[2,:]) .* cos(z[2,:]));
        row_vector[pdims[2]] m12 = 0.5 * Mp * Lp * Lr * cos(z[2,:]);
        real m22 = (Jp + 0.25 * Mp * Lp^2);
        row_vector[pdims[2]] sc = m11 * m22 - m12 .* m12;      // denominator of 2x2 inverse of mass matrix

        row_vector[pdims[2]] tau = (Km * (u - Km * z[3,:])) / Rm;

        row_vector[pdims[2]] d1 = tau - Dr * z[3,:] - 0.5 * Mp * Lp^2 * (sin(z[2,:]) .* cos(z[2,:]) .*z[3,:].* z[4,:]) + 0.5 * Mp * Lp * Lr * (sin(z[2,:]) .* z[4,:] .* z[4,:]);
        row_vector[pdims[2]] d2 = - Dp * z[4,:] + 0.25 *Mp * Lp^2 * (cos(z[2,:]) .* sin(z[2,:]) .* z[3,:].*z[3,:]) - 0.5 * Mp * Lp * g * sin(z[2,:]);

        dz[1,:] = z[3,:];
        dz[2,:] = z[4,:];
        dz[3,:] = (m22 * d1 - m12 .* d2) ./ sc;
        dz[4,:] = (m11 .* d2 - m12 .* d1) ./ sc;
        return dz;
    }

    matrix rk4_update(matrix z, row_vector u, vector theta, real Lr, real Mp, real Lp, real g, real Ts){
        int pdims[2] = dims(z);
        matrix[pdims[1],pdims[2]] k1;
        matrix[pdims[1],pdims[2]] k2;
        matrix[pdims[1],pdims[2]] k3;
        matrix[pdims[1],pdims[2]] k4;
        k1 = process_model_vec(z, u, theta, Lr, Mp, Lp, g);
        k2 = process_model_vec(z+Ts*k1/2, u, theta, Lr, Mp, Lp, g);
        k3 = process_model_vec(z+Ts*k2/2, u, theta, Lr, Mp, Lp, g);
        k4 = process_model_vec(z+Ts*k3, u, theta, Lr, Mp, Lp, g);

        return z + Ts*(k1/6 + k2/3 + k3/3 + k4/6);
    }

}

data {
    int<lower=0> no_obs;
    matrix[3, no_obs] y;                // measurement [x, y, theta]
    row_vector[no_obs] u;               // input
    real<lower=0> Ts;                   // time step
    real Lr;                            // arm length
    real Mp;                            // pendulum mass
    real Lp;                            // pendulum length
    real g;                             // gravity
    vector[6] theta_p_mu;               // prior means on theta
    vector[6] theta_p_std;              // prior stds on theta
//    vector[4] z0_mu;                    // initial state prior
//    vector[4] z0_std;
    vector[3] r_p_mu;                   // measurement noise prior mean
    vector[3] r_p_std;                  // measurement noise prior std
    vector[4] q_p_mu;                   // process noise prior mean
    vector[4] q_p_std;                  // process noise prior std
//    vector[4] h0_mu;                    // prior on initial state
//    vector[4] z0;                       // initial state guess
}
parameters {
    matrix[4,no_obs+1] h;                     // hidden states
    vector<lower=0.0>[6] theta;             // the parameters  [Jr, Jp, Km, Rm, Dp, Dr]
//    vector[4] z0;                           // initial state guess
    vector<lower=0.0>[3] r;                 // independent measurement noise variances
    vector<lower=0.0>[4] q;                 // independent process noise variances

}
transformed parameters {
    matrix[4, no_obs] mu;
    matrix[3, no_obs] yhat;

    // process model
    mu = rk4_update(h[:,1:no_obs], u[1:no_obs], theta, Lr, Mp, Lp, g, Ts); // this option was used for results in paper

    // measurement model
    yhat[1:2,:] = h[1:2,1:no_obs];
    yhat[3,:] = (u - theta[3] * h[3, 1:no_obs]) / theta[4];

}
model {
    r ~ normal(r_p_mu, r_p_std);
    q ~ normal(q_p_mu, q_p_std);

    // parameter priors
    theta ~ normal(theta_p_mu, theta_p_std);

    // initial state prior
    h[:,1] ~ normal(z0_mu, z0_std);      //

    // independent process likelihoods
    h[1,2:no_obs+1] ~ normal(mu[1,:], q[1]);
    h[2,2:no_obs+1] ~ normal(mu[2,:], q[2]);
    h[3,2:no_obs+1] ~ normal(mu[3,:], q[3]);
    h[4,2:no_obs+1] ~ normal(mu[4,:], q[4]);

    // independent measurement likelihoods
    y[1,:] ~ normal(yhat[1,:], r[1]);
    y[2,:] ~ normal(yhat[2,:], r[2]);
    y[3,:] ~ normal(yhat[3,:], r[3]);

}
//generated quantities {
//    cholesky_factor_cov[7] L;
//    real loglikelihood;
//    L = diag_pre_multiply(tau,Lcorr);
////    loglikelihood = matrix_normal_lpdf(append_row(h,y) | mu_c, diag_pre_multiply(tau,Lcorr));
//
//}


