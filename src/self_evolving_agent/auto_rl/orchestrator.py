import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.tune.search.hyperopt import HyperOptSearch
from config import load_config

class AutoRLOrchestrator:
    def __init__(self, config):
        self.config = config
        self.search_space = {
            "env": self.config["env"],
            "lr": tune.loguniform(self.config["search_space"]["lr"][0], self.config["search_space"]["lr"][1]),
            "gamma": tune.uniform(self.config["search_space"]["gamma"][0], self.config["search_space"]["gamma"][1]),
            "lambda": tune.uniform(self.config["search_space"]["lambda"][0], self.config["search_space"]["lambda"][1]),
            "clip_param": tune.uniform(self.config["search_space"]["clip_param"][0], self.config["search_space"]["clip_param"][1]),
            "entropy_coeff": tune.uniform(self.config["search_space"]["entropy_coeff"][0], self.config["search_space"]["entropy_coeff"][1]),
        }

    def run_experiment(self):
        ray.init()

        hyperopt_search = HyperOptSearch(
            metric="episode_reward_mean",
            mode="max")

        analysis = tune.run(
            PPO,
            config=self.search_space,
            stop={"training_iteration": self.config["max_iterations"]},
            resources_per_trial=self.config["resources_per_trial"],
            num_samples=self.config["num_samples"],
            search_alg=hyperopt_search,
        )

        print("Best hyperparameters found were: ", analysis.best_config)

if __name__ == "__main__":
    config = load_config()["auto_rl"]
    orchestrator = AutoRLOrchestrator(config)
    orchestrator.run_experiment()