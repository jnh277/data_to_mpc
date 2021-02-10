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
    real matrix_normal_lpdf(matrix y, matrix mu, matrix LSigma){
        int pdims[2] = dims(y);
        matrix[pdims[1],pdims[2]] error_sc = mdivide_left_tri_low(LSigma, y - mu);
        real p1 = -0.5*pdims[2]*(pdims[1]*log(2*pi()) + 2*sum(log(diagonal(LSigma))));
        real p2 = -0.5*sum(error_sc .* error_sc);
        return p1+p2;

    }
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
    matrix discrete_update_vec(matrix z, row_vector u, vector theta, real Lr, real Mp, real Lp, real g, real Ts){
        int pdims[2] = dims(z);
        matrix[pdims[1],pdims[2]] z_next;
        z_next = z;
        for (n in 1:2){
            z_next = z_next + Ts/2 * process_model_vec(z_next, u, theta, Lr, Mp, Lp, g);
        }
        return z_next;
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

    matrix multi_rk4_update(matrix z, row_vector u, vector theta, real Lr, real Mp, real Lp, real g, real Ts){
        int pdims[2] = dims(z);
        matrix[pdims[1],pdims[2]] z_next;
        z_next = z;
        for (n in 1:2){
            z_next = rk4_update(z_next, u, theta, Lr, Mp, Lp, g, Ts/2);
        }
        return z_next;
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
//    vector[4] z0;                       // initial state guess
}
parameters {
    matrix[4,no_obs+1] h;                     // hidden states
    vector<lower=0.0>[6] theta;             // the parameters  [Jr, Jp, Km, Rm, Dp, Dr]
//    vector[4] z0;                           // initial state guess
    // components of joint covariance
    vector<lower=0,upper=pi()/2>[7] tau_unif;
    cholesky_factor_corr[7] Lcorr;
    // horeshoe hyperparameters for theta
    vector<lower=0.0>[6] theta_hyper;
    real<lower=0.0> shrinkage_param;

}
transformed parameters {
    matrix[4, no_obs] mu;
    matrix[3, no_obs] yhat;
    matrix[7, no_obs] mu_c;
//    matrix[3,no_obs] yhat;
    vector<lower=0>[7] tau = 2.5 * tan(tau_unif);       // L diag scaling


    // process model
//    mu = discrete_update_vec(h[:,1:no_obs],u[1:no_obs], theta, Lr, Mp, Lp, g, Ts);
    mu = rk4_update(h[:,1:no_obs], u[1:no_obs], theta, Lr, Mp, Lp, g, Ts); // this option was used for results in paper
//    mu = multi_rk4_update(h[:,1:no_obs], u1[1:no_obs], u2[1:no_obs-1], m, J, l, a, r1, r2, Ts);

    // measurement model
    yhat[1:2,:] = h[1:2,1:no_obs];
    yhat[3,:] = (u - theta[3] * h[3, 1:no_obs]) / theta[4];

    // combine
    mu_c[1:4,1:no_obs] = mu;
    mu_c[5:7,:] = yhat;
}
model {
    Lcorr ~ lkj_corr_cholesky(2);

    // parameter priors
    theta_hyper ~ cauchy(0.0, 1.0);
    shrinkage_param ~ cauchy(0.0, 1.0);
    theta ~ normal(0.0, theta_hyper * shrinkage_param);

    // initial state prior (don't use this for now)
    h[:,1] ~ normal(0, 2);      //


    // combined distribution
    target += matrix_normal_lpdf(append_row(h[:,2:no_obs+1],y) | mu_c, diag_pre_multiply(tau,Lcorr));

}
generated quantities {
    cholesky_factor_cov[7] L;
    real loglikelihood;
    L = diag_pre_multiply(tau,Lcorr);
//    loglikelihood = matrix_normal_lpdf(append_row(h,y) | mu_c, diag_pre_multiply(tau,Lcorr));

}


