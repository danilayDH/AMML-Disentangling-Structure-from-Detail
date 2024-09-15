import pandas as pd 
import numpy as np
import sys

def read_csv_files(seed1_file, seed2_file, seed3_file):
    seed1 = pd.read_csv(seed1_file)
    seed2 = pd.read_csv(seed2_file)
    seed3 = pd.read_csv(seed3_file)

    print("\nContents of CSV files:")
    print(f"\nSeed 1 ({seed1_file}):")
    print(seed1.to_string(index=False))
    print(f"\nSeed 2 ({seed2_file}):")
    print(seed2.to_string(index=False))
    print(f"\nSeed 3 ({seed3_file}):")
    print(seed3.to_string(index=False))

    return seed1, seed2, seed3

def separate_train_test(seed1, seed2, seed3):
    train_set_results = pd.concat([seed1.iloc[0:1], seed2.iloc[0:1], seed3.iloc[0:1]])
    test_set_results = pd.concat([seed1.iloc[1:2], seed2.iloc[1:2], seed3.iloc[1:2]])
    return train_set_results, test_set_results

def calculate_stats(df):
    mean = df.mean()
    std = df.std()
    return pd.concat([mean, std], axis=1, keys=['Mean', 'Std'])

def format_results(stats, set_name):
    print(f"\n{set_name} Set Results:")
    print("=" * 100)
    print(f"{'Label':<50} {'Demographics':<20} {'VAE':<15} {'Both':<10}")
    print("-" * 100)
    
    for label in stats.index:
        if label.endswith('demographics'):
            base_label = label[:-12]
            demo_mean, demo_std = stats.loc[label, 'Mean'], stats.loc[label, 'Std']
            vae_mean, vae_std = stats.loc[f"{base_label}vae", 'Mean'], stats.loc[f"{base_label}vae", 'Std']
            both_mean, both_std = stats.loc[f"{base_label}both", 'Mean'], stats.loc[f"{base_label}both", 'Std']
            
            print(f"{base_label:<50} {demo_mean:.3f} ± {demo_std:.3f}  {vae_mean:.3f} ± {vae_std:.3f}  {both_mean:.3f} ± {both_std:.3f}")

def analyze_results(seed1_file, seed2_file, seed3_file):
    # read CSV files
    seed1, seed2, seed3 = read_csv_files(seed1_file, seed2_file, seed3_file)

    # separate train and test results
    train_results, test_results = separate_train_test(seed1, seed2, seed3)

    # calculate mean and standard deviation
    train_stats = calculate_stats(train_results)
    test_stats = calculate_stats(test_results)

    return train_stats, test_stats

def main(seed1_file, seed2_file, seed3_file):
    train_stats, test_stats = analyze_results(seed1_file, seed2_file, seed3_file)
    format_results(train_stats, "Train")
    format_results(test_stats, "Test")

if __name__ == "__main__":
        file_paths = {
        'no_mask': {
            'seed1': 'checkpoints/06bqlwhr/06bqlwhr_results.csv',
            'seed2': 'checkpoints/xqyhte28/xqyhte28_results.csv',
            'seed3': 'checkpoints/y0ujzm8r/y0ujzm8r_results.csv'
        },
        'in_encoder': {
            'seed1': 'checkpoints/863ju0oj/863ju0oj_results.csv',
            'seed2': 'checkpoints/uncc6psz/uncc6psz_results.csv',
            'seed3': 'checkpoints/djtvs3fp/djtvs3fp_results.csv'
        },
        'separate_encoder': {
            'seed1': 'checkpoints/1xz1mlxl/1xz1mlxl_results.csv',
            'seed2': 'checkpoints/luqlp9d5/luqlp9d5_results.csv',
            'seed3': 'checkpoints/0je1259s/0je1259s_results.csv'
        }
    }
        # Analyze results for each VAE type
        for vae_type, seeds in file_paths.items():
            print(f"\nAnalyzing results for {vae_type.upper()}:")
            main(seeds['seed1'], seeds['seed2'], seeds['seed3'])
    
    #if len(sys.argv) != 4:
       # print("Usage: python analyze_results.py <seed1_file> <seed2_file> <seed3_file>")
        #sys.exit(1)
    
   # seed1_file, seed2_file, seed3_file = sys.argv[1], sys.argv[2], sys.argv[3]
   # main(seed1_file, seed2_file, seed3_file)
   
  