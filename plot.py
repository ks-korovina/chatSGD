import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

def parse_accs(filepath):
    val_nums = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("Validation"):
                num = float(line.split()[-1])
                val_nums.append(num)
    return val_nums

def plot_all(filenames, labels):
    for f,l in zip(filenames, labels):
        val = parse_accs(f)
        plt.plot(range(len(val)), val, label=l)
    plt.legend()
    # plt.yscale('log')
    # plt.show()
    plt.savefig("./figures/vary-level.png")

if __name__ == "__main__":
    # For varying bandwidth exper
    filenames = ["logs/log_5_5wrkrs", "logs/log_10_5wrkrs", "logs/log_100_5wrkrs", "logs/log_5-5-10-10-10_5wrkrs", "logs/log_5-5-100-100-100_5wrkrs"]
    labels = ["5", "10", "100", "5+10", "5+100"]

    # # Compression levels experiments
    # filenames = ["logs/log_1_10wrkrs", "logs/log_2_10wrkrs", "logs/log_20_10wrkrs", "logs/log_1000_10wrkrs"]
    # labels = ["5", "10", "5+10", "5+100"]

    plot_all(filenames, labels)