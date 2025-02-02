import os
import sys
import csv
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from collections import defaultdict
from itertools import product

from utils.data_preprocessing import dose_class, load_data, LABEL_KEY

# Import student submission
import submission


def run_warfarine(args, data, learner, large_error_penalty=False):
    avg = []
    frac_incorrect = []
    print("running warfarine {}".format(args.model))
    for _ in range(args.runs):
        # Shuffle
        data = data.sample(frac=1)
        T = len(data)
        n_egregious = 0
        correct = np.zeros(T, dtype=bool)
        for t in range(T):
            x = dict(data.iloc[t])
            label = x.pop(LABEL_KEY)
            action = learner.choose(x)
            correct[t] = action == dose_class(label)
            reward = int(correct[t]) - 1
            if (action == 0 and dose_class(label) == 2) or (action == 2 and dose_class(label) == 0):
               n_egregious += 1
               reward = large_error_penalty
            learner.update(x, action, reward)

        results = {
            "total_fraction_correct": np.mean(correct),
            "average_fraction_incorrect": np.mean(
                [np.mean(~correct[:t]) for t in range(1, T)]
            ),
            "fraction_incorrect_per_time": [np.mean(~correct[:t]) for t in range(1, T)],
            "fraction_egregious": float(n_egregious) / T,
        }
        avg.append(results["fraction_incorrect_per_time"])
        print([(x, results[x]) for x in results if x != "fraction_incorrect_per_time"])

    frac_incorrect.append((args.model, np.mean(np.asarray(avg), 0)))
    return frac_incorrect


def plot_frac_incorrect(frac_incorrect):
    plt.xlabel("examples seen")
    plt.ylabel("fraction_incorrect")
    legend = []
    for name, values in frac_incorrect:
        legend.append(name)
        plt.plot(values[10:])
    plt.ylim(0.0, 1.0)
    plt.legend(legend)
    plt.savefig(os.path.join("results", "fraction_incorrect.png"))


def run_simulator(sim, learner, T):
    """
    Runs the learnerfor T steps on the simulator
    """
    correct = np.zeros(T, dtype=bool)
    u, x = sim.reset()
    for t in range(T):
        action = learner.choose(x)
        new_u, new_x, reward, arm_added = sim.step(u, action)
        learner.update(x, action, reward)
        if arm_added:
            learner.add_arm_params()
        x, u = new_x, new_u
        correct[t] = reward == 0

    return {
        "total_fraction_correct": np.mean(correct),
        "average_fraction_incorrect": np.mean(
            [np.mean(~correct[:t]) for t in range(1, T)]
        ),
        "fraction_incorrect_per_time": [np.mean(~correct[:t]) for t in range(1, T)],
    }


def main(args):

    if args.type == "warfarine":

        # Import data and define features
        data = load_data()

        features = [
            "Age in decades",
            "Height (cm)",
            "Weight (kg)",
            "Male",
            "Female",
            "Asian",
            "Black",
            "White",
            "Unknown race",
            "Carbamazepine (Tegretol)",
            "Phenytoin (Dilantin)",
            "Rifampin or Rifampicin",
            "Amiodarone (Cordarone)",
        ]

        extra_features = [
            "VKORC1AG",
            "VKORC1AA",
            "VKORC1UN",
            "CYP2C912",
            "CYP2C913",
            "CYP2C922",
            "CYP2C923",
            "CYP2C933",
            "CYP2C9UN",
        ]

        features = features + extra_features

        # run_warfarine the appropriate model based on user inputs
        frac_incorrect = []
        if args.model == "fixed":
            frac_incorrect = run_warfarine(args, data, submission.FixedDosePolicy())

        if args.model == "clinical":
            frac_incorrect = run_warfarine(args, data, submission.ClinicalDosingPolicy())

        if args.model == "linucb":
            frac_incorrect = run_warfarine(
                args,
                data,
                submission.LinUCB(3, features, alpha=args.alpha),
                large_error_penalty=args.large_error_penalty,
            )

        if args.model == "egreedy":
            frac_incorrect = run_warfarine(
                args,
                data,
                submission.eGreedyLinB(3, features, alpha=args.ep),
                large_error_penalty=args.large_error_penalty,
            )

        if args.model == "thompson":
            frac_incorrect = run_warfarine(
                args,
                data,
                submission.ThomSampB(3, features, alpha=args.v2),
                large_error_penalty=args.large_error_penalty,
            )

        # Store results based on frac_incorrect
        os.makedirs("results", exist_ok=True)
        if frac_incorrect != []:
            for algorithm, results in frac_incorrect:
                with open(f"results/{algorithm}.csv", "w", newline="") as f:
                    csv.writer(f).writerows(results.reshape(-1, 1).tolist())

        # Concatenate all model results
        frac_incorrect_all = []
        for filename in os.listdir("results"):
            if filename.endswith(".csv"):
                algorithm = filename.split(".")[0]
                with open(os.path.join("results", filename), "r") as f:
                    frac_incorrect_all.append(
                        (
                            algorithm,
                            np.array(list(csv.reader(f))).astype("float64").squeeze(),
                        )
                    )

        # Plot the fraction of incorrect results
        plot_frac_incorrect(frac_incorrect_all)

    elif args.type == "simulator":
        frac_incorrect = defaultdict(list)
        frac_correct = defaultdict(list)
        for k, u in product(args.update_freq, args.update_arms_strat): 
            stats = []
            print(f'Running LinUCB bandit with K={k} and U={u} with seeds {args.seeds}')
            for seed in args.seeds:
                np.random.seed(seed)
                sim = submission.Simulator(num_users=args.num_users, num_arms=args.num_arms, num_features=args.num_features, update_freq=k, update_arms_strategy=u)
                policy = submission.DynamicLinUCB(args.num_arms, ['None' for _ in range(args.num_features)], alpha=args.alpha)
                results = run_simulator(sim, policy, args.T)
                stats.append(results['total_fraction_correct'])
                frac_incorrect[f"LinUCB_K={k}_U={u}"].append(results["fraction_incorrect_per_time"])
                frac_correct[u].append((k, results['total_fraction_correct']))
            stats = np.asarray(stats)
            print(f'Total Fraction Correct: Mean: {stats.mean().round(3)}, Std: {stats.std().round(3)}')
            print('###########################################')

        if args.plot_u:
            plt.xlabel("Users seen")
            plt.ylabel("Fraction Incorrect")
            legend = []
            for name, values in frac_incorrect.items():
                legend.append(name)
                values = np.asarray(values)
                mean, std = values.mean(0)[10:], values.std(0)[10:]
                x = np.arange(len(mean))
                plt.plot(x, mean)
                plt.fill_between(x, mean+std, mean-std, alpha=0.1)
            plt.ylim(0.0, 1.0)
            plt.legend(legend)
            plt.savefig('fraction_incorrect.png')

        if args.plot_k:
            plt.xlabel("K")
            plt.ylabel("Total Fraction Correct")
            for name, values in frac_correct.items():
                x, y = [], {}
                for k, val in values:
                    if k not in x:
                        x.append(k)
                        y[k] = [val]
                    else:
                        y[k].append(val)
                mean = np.asarray([np.mean(y[k]) for k in x])
                std = np.asarray([np.std(y[k]) for k in x])
                plt.plot(x, mean, label=name)
                plt.fill_between(x, mean+std, mean-std, alpha=0.1)
            plt.ylim(0.0, 1.0)
            plt.legend()
            plt.savefig('k_analysis.png')

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        '--type', '-t', choices=["warfarine", "simulator"], default="warfarine"
    )
    parser.add_argument("--alpha", type=float, default=1.0)

    # Warfarine estimation specific parameters
    parser.add_argument(
        "--model", choices=["fixed", "clinical", "linucb", "egreedy", "thompson"], required=False
    )

    parser.add_argument("--ep", type=float, default=1)
    parser.add_argument("--v2", type=float, default=0.001)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--large-error-penalty", type=float, default=-1)

    # Recommender system simulator specific parameters
    parser.add_argument('--seeds', '-s', nargs='+', type=int, default=[0])
    parser.add_argument('--num-users', type=int, default=25)
    parser.add_argument('--num-arms', type=int, default=10)
    parser.add_argument('--num-features', type=int, default=10)
    parser.add_argument('--update-freq', '-k', type=int, nargs='+', default=[1000])
    parser.add_argument('--update-arms-strat', '-u', type=str, nargs='+', default=['none'], choices=['none', 'counterfactual', 'popular', 'corrective'])
    parser.add_argument('--T', type=int, default=10000)
    parser.add_argument('--plot-u', action='store_true')
    parser.add_argument('--plot-k', action='store_true')

    args = parser.parse_args()
    main(args)
