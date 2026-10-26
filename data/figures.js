window.FIGURES = {
 "families": {
  "lr": {
   "label": "Linear regression",
   "blurb": "A 0-Markov family: each task is a weight vector in \u211d\u2078, observations are noisy linear responses. PMC recovers a distribution over the weights themselves.",
   "latent": "Regression coefficient",
   "diversity_label": "M",
   "diversity": [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536
   ],
   "prompt_length": 32,
   "sources": [
    {
     "key": "prior",
     "label": "Prior",
     "kind": "prior"
    },
    {
     "key": "discrete",
     "label": "Posterior \u00b7 in-distribution",
     "kind": "posterior"
    },
    {
     "key": "gaussian",
     "label": "Posterior \u00b7 out-of-distribution",
     "kind": "posterior"
    },
    {
     "key": "random",
     "label": "Posterior \u00b7 random",
     "kind": "posterior"
    }
   ]
  },
  "bau": {
   "label": "Balls-and-urns",
   "blurb": "A 0-Markov family over the simplex: each task is an urn composition, observations are draws from it. The population prior is Dirichlet, so every marginal is a Beta.",
   "latent": "Color probability",
   "diversity_label": "M",
   "diversity": [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096
   ],
   "prompt_length": 32,
   "sources": [
    {
     "key": "prior",
     "label": "Prior",
     "kind": "prior"
    },
    {
     "key": "data_memorising",
     "label": "Posterior \u00b7 in-distribution",
     "kind": "posterior"
    },
    {
     "key": "data_generalising",
     "label": "Posterior \u00b7 out-of-distribution",
     "kind": "posterior"
    }
   ]
  },
  "markov": {
   "label": "Markov chains",
   "blurb": "A 1-Markov family: each task is a K=10 transition matrix. PMC returns a distribution over every row of that matrix, shown as a matrix of marginals.",
   "latent": "Transition probability",
   "diversity_label": "M",
   "diversity": [
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048
   ],
   "prompt_length": 32,
   "sources": [
    {
     "key": "prior",
     "label": "Prior",
     "kind": "prior"
    },
    {
     "key": "posterior_in_distribution",
     "label": "Posterior \u00b7 in-distribution",
     "kind": "posterior"
    },
    {
     "key": "posterior_out_of_distribution",
     "label": "Posterior \u00b7 out-of-distribution",
     "kind": "posterior"
    }
   ]
  }
 },
 "marginals": {
  "lr|1|prior": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T1_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|1|discrete": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T1_discrete_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|1|gaussian": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T1_gaussian_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|1|random": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T1_random_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|2|prior": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T2_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|2|discrete": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T2_discrete_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|2|gaussian": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T2_gaussian_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|2|random": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T2_random_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|4|prior": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T4_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|4|discrete": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T4_discrete_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|4|gaussian": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T4_gaussian_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|4|random": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T4_random_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|8|prior": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T8_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|8|discrete": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T8_discrete_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|8|gaussian": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T8_gaussian_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|8|random": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T8_random_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|16|prior": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T16_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|16|discrete": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T16_discrete_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|16|gaussian": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T16_gaussian_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|16|random": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T16_random_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|32|prior": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T32_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|32|discrete": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T32_discrete_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|32|gaussian": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T32_gaussian_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|32|random": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T32_random_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|64|prior": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T64_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|64|discrete": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T64_discrete_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|64|gaussian": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T64_gaussian_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|64|random": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T64_random_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|128|prior": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T128_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|128|discrete": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T128_discrete_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|128|gaussian": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T128_gaussian_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|128|random": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T128_random_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|256|prior": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T256_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|256|discrete": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T256_discrete_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|256|gaussian": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T256_gaussian_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|256|random": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T256_random_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|512|prior": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T512_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|512|discrete": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T512_discrete_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|512|gaussian": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T512_gaussian_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|512|random": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T512_random_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|1024|prior": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T1024_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|1024|discrete": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T1024_discrete_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|1024|gaussian": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T1024_gaussian_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|1024|random": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T1024_random_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|2048|prior": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T2048_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|2048|discrete": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T2048_discrete_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|2048|gaussian": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T2048_gaussian_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|2048|random": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T2048_random_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|4096|prior": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T4096_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|4096|discrete": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T4096_discrete_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|4096|gaussian": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T4096_gaussian_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|4096|random": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T4096_random_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|8192|prior": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T8192_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|8192|discrete": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T8192_discrete_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|8192|gaussian": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T8192_gaussian_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|8192|random": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T8192_random_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|16384|prior": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T16384_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|16384|discrete": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T16384_discrete_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|16384|gaussian": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T16384_gaussian_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|16384|random": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T16384_random_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|32768|prior": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T32768_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|32768|discrete": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T32768_discrete_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|32768|gaussian": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T32768_gaussian_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|32768|random": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T32768_random_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|65536|prior": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T65536_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|65536|discrete": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T65536_discrete_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|65536|gaussian": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T65536_gaussian_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "lr|65536|random": {
   "grid": {
    "src": "figures/lr/grid_marginal_4dim_T65536_random_L32.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|1|prior": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M1__source_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|1|data_memorising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M1__source_data_memorising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|1|data_generalising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M1__source_data_generalising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|2|prior": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M2__source_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|2|data_memorising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M2__source_data_memorising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|2|data_generalising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M2__source_data_generalising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|4|prior": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M4__source_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|4|data_memorising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M4__source_data_memorising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|4|data_generalising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M4__source_data_generalising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|8|prior": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M8__source_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|8|data_memorising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M8__source_data_memorising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|8|data_generalising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M8__source_data_generalising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|16|prior": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M16__source_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|16|data_memorising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M16__source_data_memorising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|16|data_generalising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M16__source_data_generalising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|32|prior": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M32__source_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|32|data_memorising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M32__source_data_memorising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|32|data_generalising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M32__source_data_generalising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|64|prior": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M64__source_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|64|data_memorising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M64__source_data_memorising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|64|data_generalising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M64__source_data_generalising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|128|prior": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M128__source_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|128|data_memorising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M128__source_data_memorising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|128|data_generalising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M128__source_data_generalising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|256|prior": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M256__source_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|256|data_memorising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M256__source_data_memorising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|256|data_generalising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M256__source_data_generalising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|512|prior": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M512__source_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|512|data_memorising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M512__source_data_memorising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|512|data_generalising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M512__source_data_generalising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|1024|prior": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M1024__source_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|1024|data_memorising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M1024__source_data_memorising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|1024|data_generalising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M1024__source_data_generalising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|2048|prior": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M2048__source_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|2048|data_memorising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M2048__source_data_memorising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|2048|data_generalising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M2048__source_data_generalising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|4096|prior": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M4096__source_prior.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|4096|data_memorising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M4096__source_data_memorising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "bau|4096|data_generalising": {
   "grid": {
    "src": "figures/bau/grid_marginal_4dim_M4096__source_data_generalising.webp",
    "w": 1600,
    "h": 560,
    "dense": true
   }
  },
  "markov|4|prior": {
   "density": {
    "src": "figures/markov/markov_full_M4_prior_marginals_matrix.webp",
    "w": 1600,
    "h": 1584,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M4_prior_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|4|posterior_in_distribution": {
   "density": {
    "src": "figures/markov/markov_full_M4_posterior_in_distribution_marginals_matrix.webp",
    "w": 1600,
    "h": 1587,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M4_posterior_in_distribution_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|4|posterior_out_of_distribution": {
   "density": {
    "src": "figures/markov/markov_full_M4_posterior_out_of_distribution_marginals_matrix.webp",
    "w": 1600,
    "h": 1587,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M4_posterior_out_of_distribution_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|8|prior": {
   "density": {
    "src": "figures/markov/markov_full_M8_prior_marginals_matrix.webp",
    "w": 1600,
    "h": 1584,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M8_prior_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|8|posterior_in_distribution": {
   "density": {
    "src": "figures/markov/markov_full_M8_posterior_in_distribution_marginals_matrix.webp",
    "w": 1600,
    "h": 1591,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M8_posterior_in_distribution_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|8|posterior_out_of_distribution": {
   "density": {
    "src": "figures/markov/markov_full_M8_posterior_out_of_distribution_marginals_matrix.webp",
    "w": 1600,
    "h": 1587,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M8_posterior_out_of_distribution_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|16|prior": {
   "density": {
    "src": "figures/markov/markov_full_M16_prior_marginals_matrix.webp",
    "w": 1600,
    "h": 1584,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M16_prior_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|16|posterior_in_distribution": {
   "density": {
    "src": "figures/markov/markov_full_M16_posterior_in_distribution_marginals_matrix.webp",
    "w": 1600,
    "h": 1593,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M16_posterior_in_distribution_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|16|posterior_out_of_distribution": {
   "density": {
    "src": "figures/markov/markov_full_M16_posterior_out_of_distribution_marginals_matrix.webp",
    "w": 1600,
    "h": 1587,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M16_posterior_out_of_distribution_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|32|prior": {
   "density": {
    "src": "figures/markov/markov_full_M32_prior_marginals_matrix.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M32_prior_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|32|posterior_in_distribution": {
   "density": {
    "src": "figures/markov/markov_full_M32_posterior_in_distribution_marginals_matrix.webp",
    "w": 1600,
    "h": 1589,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M32_posterior_in_distribution_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|32|posterior_out_of_distribution": {
   "density": {
    "src": "figures/markov/markov_full_M32_posterior_out_of_distribution_marginals_matrix.webp",
    "w": 1600,
    "h": 1587,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M32_posterior_out_of_distribution_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|64|prior": {
   "density": {
    "src": "figures/markov/markov_full_M64_prior_marginals_matrix.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M64_prior_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|64|posterior_in_distribution": {
   "density": {
    "src": "figures/markov/markov_full_M64_posterior_in_distribution_marginals_matrix.webp",
    "w": 1600,
    "h": 1593,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M64_posterior_in_distribution_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|64|posterior_out_of_distribution": {
   "density": {
    "src": "figures/markov/markov_full_M64_posterior_out_of_distribution_marginals_matrix.webp",
    "w": 1600,
    "h": 1587,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M64_posterior_out_of_distribution_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|128|prior": {
   "density": {
    "src": "figures/markov/markov_full_M128_prior_marginals_matrix.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M128_prior_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|128|posterior_in_distribution": {
   "density": {
    "src": "figures/markov/markov_full_M128_posterior_in_distribution_marginals_matrix.webp",
    "w": 1600,
    "h": 1588,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M128_posterior_in_distribution_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|128|posterior_out_of_distribution": {
   "density": {
    "src": "figures/markov/markov_full_M128_posterior_out_of_distribution_marginals_matrix.webp",
    "w": 1600,
    "h": 1587,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M128_posterior_out_of_distribution_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|256|prior": {
   "density": {
    "src": "figures/markov/markov_full_M256_prior_marginals_matrix.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M256_prior_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|256|posterior_in_distribution": {
   "density": {
    "src": "figures/markov/markov_full_M256_posterior_in_distribution_marginals_matrix.webp",
    "w": 1600,
    "h": 1587,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M256_posterior_in_distribution_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|256|posterior_out_of_distribution": {
   "density": {
    "src": "figures/markov/markov_full_M256_posterior_out_of_distribution_marginals_matrix.webp",
    "w": 1600,
    "h": 1587,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M256_posterior_out_of_distribution_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|512|prior": {
   "density": {
    "src": "figures/markov/markov_full_M512_prior_marginals_matrix.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M512_prior_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|512|posterior_in_distribution": {
   "density": {
    "src": "figures/markov/markov_full_M512_posterior_in_distribution_marginals_matrix.webp",
    "w": 1600,
    "h": 1587,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M512_posterior_in_distribution_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|512|posterior_out_of_distribution": {
   "density": {
    "src": "figures/markov/markov_full_M512_posterior_out_of_distribution_marginals_matrix.webp",
    "w": 1600,
    "h": 1587,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M512_posterior_out_of_distribution_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|1024|prior": {
   "density": {
    "src": "figures/markov/markov_full_M1024_prior_marginals_matrix.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M1024_prior_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|1024|posterior_in_distribution": {
   "density": {
    "src": "figures/markov/markov_full_M1024_posterior_in_distribution_marginals_matrix.webp",
    "w": 1600,
    "h": 1587,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M1024_posterior_in_distribution_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|1024|posterior_out_of_distribution": {
   "density": {
    "src": "figures/markov/markov_full_M1024_posterior_out_of_distribution_marginals_matrix.webp",
    "w": 1600,
    "h": 1587,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M1024_posterior_out_of_distribution_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|2048|prior": {
   "density": {
    "src": "figures/markov/markov_full_M2048_prior_marginals_matrix.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M2048_prior_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|2048|posterior_in_distribution": {
   "density": {
    "src": "figures/markov/markov_full_M2048_posterior_in_distribution_marginals_matrix.webp",
    "w": 1600,
    "h": 1583,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M2048_posterior_in_distribution_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  },
  "markov|2048|posterior_out_of_distribution": {
   "density": {
    "src": "figures/markov/markov_full_M2048_posterior_out_of_distribution_marginals_matrix.webp",
    "w": 1600,
    "h": 1587,
    "dense": true
   },
   "cdf": {
    "src": "figures/markov/markov_full_M2048_posterior_out_of_distribution_marginals_matrix_cdf.webp",
    "w": 1600,
    "h": 1590,
    "dense": true
   }
  }
 },
 "context": {
  "lr|prior": [
   {
    "slug": "stitched-density",
    "label": "Density across every M",
    "dense": true,
    "src": "figures/lr/stitched_density_prior.webp",
    "w": 1133,
    "h": 1600
   },
   {
    "slug": "stitched-cdf",
    "label": "CDF across every M",
    "dense": true,
    "src": "figures/lr/stitched_cdf_prior.webp",
    "w": 1133,
    "h": 1600
   }
  ],
  "lr|discrete": [
   {
    "slug": "stitched-density",
    "label": "Density across every M",
    "dense": true,
    "src": "figures/lr/stitched_density_discrete_L32.webp",
    "w": 1133,
    "h": 1600
   },
   {
    "slug": "stitched-cdf",
    "label": "CDF across every M",
    "dense": true,
    "src": "figures/lr/stitched_cdf_discrete_L32.webp",
    "w": 1133,
    "h": 1600
   }
  ],
  "lr|gaussian": [
   {
    "slug": "stitched-density",
    "label": "Density across every M",
    "dense": true,
    "src": "figures/lr/stitched_density_gaussian_L32.webp",
    "w": 1133,
    "h": 1600
   },
   {
    "slug": "stitched-cdf",
    "label": "CDF across every M",
    "dense": true,
    "src": "figures/lr/stitched_cdf_gaussian_L32.webp",
    "w": 1133,
    "h": 1600
   }
  ],
  "lr|random": [
   {
    "slug": "stitched-density",
    "label": "Density across every M",
    "dense": true,
    "src": "figures/lr/stitched_density_random_L32.webp",
    "w": 1133,
    "h": 1600
   },
   {
    "slug": "stitched-cdf",
    "label": "CDF across every M",
    "dense": true,
    "src": "figures/lr/stitched_cdf_random_L32.webp",
    "w": 1133,
    "h": 1600
   }
  ],
  "bau|prior": [
   {
    "slug": "stitched-density",
    "label": "Density across every M",
    "dense": true,
    "src": "figures/bau/stitched_bau_density_prior.webp",
    "w": 1600,
    "h": 1177
   },
   {
    "slug": "stitched-cdf",
    "label": "CDF across every M",
    "dense": true,
    "src": "figures/bau/stitched_bau_cdf_prior.webp",
    "w": 1600,
    "h": 1175
   }
  ],
  "bau|data_memorising": [
   {
    "slug": "stitched-density",
    "label": "Density across every M",
    "dense": true,
    "src": "figures/bau/stitched_bau_density_data_memorising.webp",
    "w": 1600,
    "h": 1175
   },
   {
    "slug": "stitched-cdf",
    "label": "CDF across every M",
    "dense": true,
    "src": "figures/bau/stitched_bau_cdf_data_memorising.webp",
    "w": 1600,
    "h": 1174
   }
  ],
  "bau|data_generalising": [
   {
    "slug": "stitched-density",
    "label": "Density across every M",
    "dense": true,
    "src": "figures/bau/stitched_bau_density_data_generalising.webp",
    "w": 1600,
    "h": 1180
   },
   {
    "slug": "stitched-cdf",
    "label": "CDF across every M",
    "dense": true,
    "src": "figures/bau/stitched_bau_cdf_data_generalising.webp",
    "w": 1600,
    "h": 1180
   }
  ]
 },
 "other": {
  "lr": [
   {
    "slug": "sweep",
    "label": "Task-diversity sweep",
    "src": "figures/lr/lr_sweep_combined_2x2.webp",
    "w": 1600,
    "h": 1300
   },
   {
    "slug": "prior",
    "label": "Prior comparisons",
    "src": "figures/lr/lr_sweep_prior_delta_ed_sw.webp",
    "w": 1600,
    "h": 409
   },
   {
    "slug": "dynamics",
    "label": "Training dynamics (M=32)",
    "src": "figures/lr/lr_dynamics_combined_logx_2x2.webp",
    "w": 1600,
    "h": 1300
   }
  ],
  "bau": [
   {
    "slug": "sweep",
    "label": "Task-diversity sweep",
    "src": "figures/bau/bau_sweep_posterior_2x2.webp",
    "w": 1600,
    "h": 1300
   },
   {
    "slug": "prior",
    "label": "Prior comparisons",
    "src": "figures/bau/bau_sweep_prior_kl_ed_sw.webp",
    "w": 1600,
    "h": 409
   },
   {
    "slug": "dynamics",
    "label": "Training dynamics (M=32)",
    "src": "figures/bau/bau_dynamics_combined_logx_2x2.webp",
    "w": 1600,
    "h": 1299
   }
  ],
  "markov": [
   {
    "slug": "sweep",
    "label": "Task-diversity sweep",
    "src": "figures/markov/markov_sweep_combined_2x2.webp",
    "w": 1600,
    "h": 1295
   },
   {
    "slug": "prior",
    "label": "Prior comparisons",
    "src": "figures/markov/markov_sweep_prior_kl_ed_sw.webp",
    "w": 1600,
    "h": 619
   },
   {
    "slug": "dynamics",
    "label": "Training dynamics (M=8)",
    "src": "figures/markov/markov_dynamics_M8_2x2.webp",
    "w": 1600,
    "h": 1295
   }
  ]
 },
 "slices": {
  "lr": {
   "header": [
    0.030000000000000002,
    0.17142857142857143
   ],
   "density": [
    0.17142857142857143,
    0.5642857142857143
   ],
   "cdf": [
    0.5642857142857143,
    0.9878571428571429
   ]
  },
  "bau": {
   "header": [
    0.030000000000000002,
    0.1875
   ],
   "density": [
    0.1875,
    0.5732142857142857
   ],
   "cdf": [
    0.5732142857142857,
    0.9878571428571429
   ]
  }
 },
 "missing": []
};
