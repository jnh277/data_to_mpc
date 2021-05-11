data {
    int<lower=0> N;
    vector<lower=0.0>[N] y;
    vector[N] x;
    }

parameters {
    real mu;
    real<lower=1e-6> sigma;
    real<lower=0.0> theta;
}

model {
    sqrt(y - theta) ~ normal(x + mu, sigma);
    target += -log(sqrt(2*sqrt(y-theta)));

}