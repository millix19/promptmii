import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datasets import load_from_disk
import argparse
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score
from ultra_clean_csv import ultra_clean_csv


@dataclass
class F1AnalysisConfig:
    """Configuration class for F1 analysis parameters"""
    input_path: str
    output_dir: str = "out/baseline_f1_macro"
    baselines: Optional[List[str]] = None
    max_rows_per_setting: int = 10
    no_cross_dataset_comparison: bool = False
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


class F1BaselineAnalyzer:
    """Main class for F1-based baseline analysis with unified interface"""
    
    def __init__(self, config: F1AnalysisConfig):
        self.config = config
        self.results_df = None
        self.color_palette = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']
        self.marker_palette = ['o', '^', 's', 'd', 'v', 'p', '*', 'h', 'H', '+', 'x', 'D']
    
    def load_and_filter_data(self) -> pd.DataFrame:
        """Load results and apply baseline filtering"""
        dataset = self._load_results(self.config.input_path)
        df = dataset.to_pandas()
        
        if self.config.baselines is None:
            print(f"[INFO] Using all {len(df['baseline'].unique())} available baselines: {sorted(df['baseline'].unique())}")
            return df
        
        available_baselines = df['baseline'].unique()
        valid_baselines = [b for b in self.config.baselines if b in available_baselines]
        invalid_baselines = [b for b in self.config.baselines if b not in available_baselines]
        
        if invalid_baselines:
            print(f"[WARNING] Requested baselines not found: {invalid_baselines}")
        
        if not valid_baselines:
            print(f"[ERROR] No requested baselines found. Available: {sorted(available_baselines)}")
            return pd.DataFrame()
        
        print(f"[INFO] Filtering to {len(valid_baselines)} baselines: {sorted(valid_baselines)}")
        filtered_df = df[df['baseline'].isin(valid_baselines)]
        print(f"[INFO] {len(df)} → {len(filtered_df)} rows after filtering")
        
        return filtered_df
    
    def _load_results(self, output_path):
        """Load results from disk - loads from /data/ subdirectory only"""
        import os
        import glob
        from datasets import concatenate_datasets
        
        # Always use /data/ subdirectory structure
        data_dir = os.path.join(output_path, "data")
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory not found: {data_dir}")
        
        # Load from data/run_*/
        run_dirs = glob.glob(os.path.join(data_dir, "run_*"))
        run_dirs = [d for d in run_dirs if os.path.isdir(d)]
        
        if not run_dirs:
            raise ValueError(f"No run directories found in {data_dir}")
            
        datasets = []
        for run_dir in sorted(run_dirs):
            try:
                dataset = load_from_disk(run_dir)
                datasets.append(dataset)
                print(f"[INFO] Loaded {len(dataset)} results from {os.path.basename(run_dir)}")
            except Exception as e:
                print(f"[WARNING] Failed to load {run_dir}: {e}")
        
        if not datasets:
            raise ValueError("No valid datasets found")
            
        # Concatenate all run datasets
        combined_dataset = concatenate_datasets(datasets)
        print(f"[INFO] Combined {len(datasets)} runs into dataset with {len(combined_dataset)} total results")
        return combined_dataset
    
    def calculate_f1_from_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate F1 scores from raw prediction data"""
        # Check for different possible column names
        label_col = 'answer' if 'answer' in df.columns else 'label'
        pred_col = 'prediction' if 'prediction' in df.columns else 'pred'
        
        if label_col not in df.columns or pred_col not in df.columns:
            print(f"[ERROR] Missing required columns for F1 calculation")
            print(f"[INFO] Available columns: {list(df.columns)}")
            return df
        
        print(f"[INFO] Using columns: '{label_col}' for labels and '{pred_col}' for predictions")
        print("[INFO] Calculating F1 scores from raw prediction data...")
        
        # Create a copy to avoid modifying original
        df_with_f1 = df.copy()
        df_with_f1['f1'] = np.nan
        
        # Group by key dimensions to calculate F1 for each group
        group_columns = ['baseline', 'n', 'prediction_model', 'dataset_name', 'config_name']
        available_group_columns = [col for col in group_columns if col in df_with_f1.columns]
        
        successful_f1_calculations = 0
        total_groups = 0
        skipped_groups = {
            'empty_group': 0,
            'no_valid_data': 0,
            'too_many_classes': 0,
            'single_class': 0,
            'calculation_error': 0
        }
        
        if available_group_columns:
            for name, group in df_with_f1.groupby(available_group_columns):
                total_groups += 1
                # Extract more detailed group info for debugging
                if len(name) >= 4:
                    baseline, n, prediction_model, dataset_name = name[:4]
                    config_name = name[4] if len(name) > 4 else 'default'
                    group_key = f"{baseline}|{dataset_name}|{config_name}|n={n}|{prediction_model}"
                else:
                    baseline, dataset_name = name[0], name[3] if len(name) > 3 else str(name)
                    group_key = str(name)
                
                if len(group) == 0:
                    skipped_groups['empty_group'] += 1
                    print(f"[SKIP] Empty group: {group_key}")
                    continue
                    
                try:
                    # Handle mixed data types more carefully
                    labels = group[label_col].fillna('').astype(str).str.lower().str.strip()
                    predictions = group[pred_col].fillna('').astype(str).str.lower().str.strip()
                    
                    # Remove empty strings and nan values
                    valid_mask = (labels != '') & (predictions != '') & (labels != 'nan') & (predictions != 'nan')
                    labels = labels[valid_mask]
                    predictions = predictions[valid_mask]
                    
                    if len(labels) == 0:
                        skipped_groups['no_valid_data'] += 1
                        print(f"[SKIP] No valid data after cleaning: {group_key}")
                        continue
                    
                    unique_labels = labels.unique()
                    unique_predictions = predictions.unique()
                    
                    # DETAILED LOGGING for debugging F1 differences across baselines
                    print(f"[DEBUG] {group_key}:")
                    print(f"        Total examples: {len(group)} -> {len(labels)} after cleaning")
                    print(f"        Unique labels: {len(unique_labels)} {sorted(unique_labels)[:10]}")
                    print(f"        Unique predictions: {len(unique_predictions)} {sorted(unique_predictions)[:10]}")
                    print(f"        Label distribution: {dict(labels.value_counts().head(5))}")
                    print(f"        Prediction distribution: {dict(predictions.value_counts().head(5))}")
                    
                    # Skip datasets with too many unique classes (likely regression-like)
                    if len(unique_labels) > len(labels) * 0.5:
                        skipped_groups['too_many_classes'] += 1
                        print(f"[SKIP] Too many unique classes ({len(unique_labels)}/{len(labels)}): {group_key}")
                        continue
                    
                    if len(unique_labels) == 2:
                        # Binary classification - use macro F1 for consistency
                        f1 = f1_score(labels, predictions, average='macro', zero_division=0)
                        print(f"        Binary F1 (macro): {f1:.4f}")
                    elif len(unique_labels) > 2:
                        # Multi-class classification
                        f1 = f1_score(labels, predictions, average='macro', zero_division=0)
                        print(f"        Multi-class F1 (macro): {f1:.4f}")
                    else:
                        # Single class - cannot calculate F1
                        skipped_groups['single_class'] += 1
                        print(f"[SKIP] Single class only ({list(unique_labels)}): {group_key}")
                        continue
                    
                    df_with_f1.loc[group.index, 'f1'] = f1
                    successful_f1_calculations += 1
                    
                except Exception as e:
                    skipped_groups['calculation_error'] += 1
                    print(f"[SKIP] Error calculating F1 for {group_key}: {str(e)}")
                    continue
        
        print(f"[INFO] F1 calculation complete: {successful_f1_calculations}/{total_groups} groups processed successfully")
        
        # Print summary of skipped groups
        total_skipped = sum(skipped_groups.values())
        if total_skipped > 0:
            print(f"[INFO] Skipped {total_skipped} groups:")
            for reason, count in skipped_groups.items():
                if count > 0:
                    print(f"  - {reason.replace('_', ' ').title()}: {count}")
        
        if df_with_f1['f1'].isna().all():
            print("[ERROR] No F1 scores could be calculated")
            return df
        
        # Print F1 statistics
        valid_f1 = df_with_f1['f1'].dropna()
        print(f"[INFO] F1 score range: {valid_f1.min():.3f} to {valid_f1.max():.3f}")
        print(f"[INFO] Mean F1: {valid_f1.mean():.3f}")
        
        return df_with_f1
    
    def _calculate_binary_f1(self, labels, predictions):
        """Handle different binary classification label formats"""
        label_set = set(labels.unique())
        pred_set = set(predictions.unique())
        
        # Handle different binary formats
        try:
            # Case 1: Standard 0/1 format
            if label_set.issubset({'0', '1'}) and pred_set.issubset({'0', '1'}):
                return f1_score(labels, predictions, average='binary', pos_label='1', zero_division=0)
            
            # Case 2: true/false format
            elif label_set.issubset({'true', 'false'}) and pred_set.issubset({'true', 'false'}):
                return f1_score(labels, predictions, average='binary', pos_label='true', zero_division=0)
            
            # Case 3: t/f format
            elif label_set.issubset({'t', 'f'}) and pred_set.issubset({'t', 'f'}):
                return f1_score(labels, predictions, average='binary', pos_label='t', zero_division=0)
            
            # Case 4: yes/no format
            elif label_set.issubset({'yes', 'no'}) and pred_set.issubset({'yes', 'no'}):
                return f1_score(labels, predictions, average='binary', pos_label='yes', zero_division=0)
            
            # Case 5: y/n format  
            elif label_set.issubset({'y', 'n'}) and pred_set.issubset({'y', 'n'}):
                return f1_score(labels, predictions, average='binary', pos_label='y', zero_division=0)
            
            # Case 6: positive/negative format
            elif label_set.issubset({'positive', 'negative'}) and pred_set.issubset({'positive', 'negative'}):
                return f1_score(labels, predictions, average='binary', pos_label='positive', zero_division=0)
            
            # Case 7: pos/neg format
            elif label_set.issubset({'pos', 'neg'}) and pred_set.issubset({'pos', 'neg'}):
                return f1_score(labels, predictions, average='binary', pos_label='pos', zero_division=0)
            
            # Case 8: Generic binary - convert to standardized format
            else:
                # Map to 0/1 format for consistency
                unique_sorted = sorted(list(label_set))
                if len(unique_sorted) == 2:
                    label_mapping = {unique_sorted[0]: '0', unique_sorted[1]: '1'}
                    labels_binary = labels.map(label_mapping)
                    predictions_binary = predictions.map(label_mapping)
                    
                    # Handle missing values in mapping
                    labels_binary = labels_binary.fillna('0')
                    predictions_binary = predictions_binary.fillna('0')
                    
                    return f1_score(labels_binary, predictions_binary, average='binary', pos_label='1', zero_division=0)
                else:
                    # Fallback to macro if not exactly binary
                    return f1_score(labels, predictions, average='macro', zero_division=0)
        
        except Exception as e:
            # Ultimate fallback to macro F1
            try:
                return f1_score(labels, predictions, average='macro', zero_division=0)
            except:
                return 0.0
    
    def get_baseline_colors_and_markers(self, baselines: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Generate consistent colors and markers for baselines"""
        colors = {}
        markers = {}
        for i, baseline in enumerate(sorted(baselines)):
            colors[baseline] = self.color_palette[i % len(self.color_palette)]
            markers[baseline] = self.marker_palette[i % len(self.marker_palette)]
        return colors, markers
    
    def filter_complete_datasets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to only include datasets that have F1 scores for ALL baselines"""
        # Get all unique baselines
        all_baselines = set(df['baseline'].unique())
        
        # For each dataset+n combination, check if all baselines have F1 scores
        complete_datasets = []
        incomplete_reasons = {}
        
        for (dataset, config, n), group in df.groupby(['dataset_name', 'config_name', 'n']):
            # Get baselines with valid F1 scores for this dataset+n
            valid_baselines = set(group.dropna(subset=['f1'])['baseline'].unique())
            missing_baselines = all_baselines - valid_baselines
            
            # Only include if ALL baselines have valid F1 scores
            if len(missing_baselines) == 0:
                complete_datasets.append((dataset, config, n))
            else:
                # Log why this dataset was excluded
                dataset_key = f"{dataset}/{config}" if config != 'default' else dataset
                if dataset_key not in incomplete_reasons:
                    incomplete_reasons[dataset_key] = {}
                incomplete_reasons[dataset_key][f"n={n}"] = sorted(missing_baselines)
        
        if len(complete_datasets) == 0:
            print("[WARNING] No datasets have F1 scores for all baselines!")
            return pd.DataFrame()
        
        # Filter original data to only include complete datasets
        mask = df.apply(lambda row: (row['dataset_name'], row['config_name'], row['n']) in complete_datasets, axis=1)
        filtered_df = df[mask].copy()
        
        total_datasets = len(df.groupby(['dataset_name', 'config_name', 'n']))
        complete_count = len(complete_datasets)
        removed_count = total_datasets - complete_count
        
        print(f"[INFO] F1 fairness filter: {total_datasets} total dataset+n combinations")
        print(f"[INFO] Kept {complete_count} complete datasets (all baselines have F1)")
        print(f"[INFO] Removed {removed_count} incomplete datasets")
        
        # Log detailed reasons for exclusions (show top 10 most problematic)
        if incomplete_reasons:
            print(f"\n[INFO] Top datasets with missing F1 scores:")
            sorted_datasets = sorted(incomplete_reasons.items(), 
                                   key=lambda x: sum(len(missing) for missing in x[1].values()), 
                                   reverse=True)
            
            for i, (dataset_key, n_missing) in enumerate(sorted_datasets[:10]):
                total_missing = sum(len(missing) for missing in n_missing.values())
                print(f"  {i+1:2d}. {dataset_key} (missing {total_missing} baseline-n combinations)")
                for n_val, missing_baselines in sorted(n_missing.items()):
                    if missing_baselines:
                        print(f"      {n_val}: missing {', '.join(missing_baselines)}")
        
        return filtered_df

    def calculate_f1_metrics(self) -> Dict[str, pd.DataFrame]:
        """Calculate all F1 analysis metrics"""
        # Filter to only datasets with F1 scores for all baselines
        fair_df = self.filter_complete_datasets(self.results_df)
        
        if len(fair_df) == 0:
            print("[ERROR] No datasets have F1 scores for all baselines - cannot create fair comparison")
            return {'f1': pd.DataFrame(), 'dataset': pd.DataFrame(), 'tokens': pd.DataFrame(), 'efficiency': pd.DataFrame()}
        
        metrics = {}
        
        # Baseline F1 metrics (using fair subset)
        metrics['f1'] = fair_df.groupby(['baseline', 'n', 'prediction_model']).agg({
            'f1': ['mean', 'count']
        }).reset_index()
        metrics['f1'].columns = ['baseline', 'n', 'prediction_model', 'f1_score', 'total_samples']
        
        # Dataset-specific F1 metrics (using fair subset)
        metrics['dataset'] = fair_df.groupby(['dataset_name', 'config_name', 'baseline', 'n', 'prediction_model']).agg({
            'f1': ['mean', 'count']
        }).reset_index()
        metrics['dataset'].columns = ['dataset_name', 'config_name', 'baseline', 'n', 'prediction_model', 'f1_score', 'total_samples']
        
        # Token analysis with F1 (using fair subset)
        metrics['tokens'] = fair_df.groupby(['baseline', 'instruction_tokens', 'prediction_model']).agg({
            'f1': ['mean', 'count']
        }).reset_index()
        metrics['tokens'].columns = ['baseline', 'instruction_tokens', 'prediction_model', 'f1_score', 'total_samples']
        
        # Efficiency metrics (F1 per token) (using fair subset)
        efficiency_data = []
        for baseline in fair_df['baseline'].unique():
            for pred_model in fair_df['prediction_model'].unique():
                mask = (fair_df['baseline'] == baseline) & (fair_df['prediction_model'] == pred_model)
                subset = fair_df[mask]
                if len(subset) > 0:
                    avg_f1 = subset['f1'].mean()
                    avg_tokens = subset['instruction_tokens'].mean()
                    efficiency = avg_f1 / avg_tokens if avg_tokens > 0 else 0
                    efficiency_data.append({
                        'baseline': baseline,
                        'prediction_model': pred_model,
                        'avg_f1_score': avg_f1,
                        'avg_tokens': avg_tokens,
                        'efficiency': efficiency,
                        'total_samples': len(subset)
                    })
        metrics['efficiency'] = pd.DataFrame(efficiency_data)
        
        return metrics
    
    def calculate_cross_dataset_f1_metrics(self, fair_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate cross-dataset F1 comparison metrics using consistent filtered data"""
        if 'naive' not in fair_df['baseline'].unique():
            print("[WARNING] Naive baseline required for cross-dataset F1 analysis")
            return None
        
        # Calculate naive baseline F1 scores per dataset (using fair data)
        naive_data = fair_df[fair_df['baseline'] == 'naive']
        naive_f1_scores = naive_data.groupby('dataset_name')['f1'].mean()
        
        results = []
        n_values = [5, 10, 20, 50, 100]
        
        grouped = fair_df.groupby(['dataset_name', 'baseline', 'n'])
        for (dataset, baseline, n), group in grouped:
            if n in n_values and dataset in naive_f1_scores:
                f1_score = group['f1'].mean()
                tokens = group['instruction_tokens'].mean()
                naive_f1 = naive_f1_scores[dataset]
                
                absolute_improvement = 0.0 if baseline == 'naive' else f1_score - naive_f1
                
                results.append({
                    'dataset': dataset,
                    'baseline': baseline,
                    'n': n,
                    'f1_score': f1_score,
                    'tokens': tokens,
                    'absolute_improvement': absolute_improvement
                })
        
        if not results:
            return None
        
        # Aggregate across datasets
        df = pd.DataFrame(results)
        aggregated = []
        for baseline in df['baseline'].unique():
            baseline_data = df[df['baseline'] == baseline]
            for n in n_values:
                n_data = baseline_data[baseline_data['n'] == n]
                if len(n_data) > 0:
                    aggregated.append({
                        'baseline': baseline,
                        'n': n,
                        'avg_absolute_improvement': n_data['absolute_improvement'].mean(),
                        'avg_tokens': n_data['tokens'].mean(),
                        'num_datasets': len(n_data)
                    })
        
        return pd.DataFrame(aggregated)
    
    def create_win_rate_matrix(self, fair_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Create win rate matrix based on F1 scores using consistent filtered data"""
        baselines = sorted(fair_df['baseline'].unique())
        if len(baselines) < 2:
            print("[WARNING] Need at least 2 baselines for win rate matrix")
            return None
        
        # Pre-compute F1 scores
        grouped_f1 = fair_df.groupby(['dataset_name', 'n', 'baseline'])['f1'].mean().reset_index()
        
        win_rates = {b1: {b2: 0 for b2 in baselines} for b1 in baselines}
        n_values = sorted(fair_df['n'].unique())
        datasets = fair_df['dataset_name'].unique()
        
        for n in n_values:
            for dataset in datasets:
                subset = grouped_f1[
                    (grouped_f1['dataset_name'] == dataset) & 
                    (grouped_f1['n'] == n)
                ]
                
                if len(subset) == 0:
                    continue
                
                baseline_f1_scores = dict(zip(subset['baseline'], subset['f1']))
                
                for b1 in baselines:
                    for b2 in baselines:
                        if b1 != b2 and b1 in baseline_f1_scores and b2 in baseline_f1_scores:
                            if baseline_f1_scores[b1] > baseline_f1_scores[b2]:
                                win_rates[b1][b2] += 1
        
        # Convert to percentages
        total_comparisons = len(datasets) * len(n_values)
        
        # Create matrix
        matrix_data = []
        for b1 in baselines:
            row = []
            for b2 in baselines:
                if b1 == b2:
                    row.append('-')
                else:
                    win_rate = (win_rates[b1][b2] / total_comparisons) * 100
                    row.append(f"{win_rate:.1f}%")
            matrix_data.append(row)
        
        return pd.DataFrame(matrix_data, index=baselines, columns=baselines)

    def create_max_win_rate_matrix(self, fair_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Create max win rate matrix - compares best F1 performance for each method per dataset using consistent filtered data"""
        baselines = sorted(fair_df['baseline'].unique())
        if len(baselines) < 2:
            print("[WARNING] Need at least 2 baselines for max win rate matrix")
            return None
        
        # Pre-compute F1 scores
        grouped_f1 = fair_df.groupby(['dataset_name', 'n', 'baseline'])['f1'].mean().reset_index()
        
        # For each dataset and baseline, get the maximum F1 score across all n values
        max_f1_scores = grouped_f1.groupby(['dataset_name', 'baseline'])['f1'].max().reset_index()
        
        win_rates = {b1: {b2: 0 for b2 in baselines} for b1 in baselines}
        datasets = fair_df['dataset_name'].unique()
        
        for dataset in datasets:
            subset = max_f1_scores[max_f1_scores['dataset_name'] == dataset]
            
            if len(subset) == 0:
                continue
                
            baseline_max_f1_scores = dict(zip(subset['baseline'], subset['f1']))
            
            for b1 in baselines:
                for b2 in baselines:
                    if b1 != b2 and b1 in baseline_max_f1_scores and b2 in baseline_max_f1_scores:
                        if baseline_max_f1_scores[b1] > baseline_max_f1_scores[b2]:
                            win_rates[b1][b2] += 1
        
        # Convert to percentages
        total_comparisons = len(datasets)
        
        # Create matrix
        matrix_data = []
        for b1 in baselines:
            row = []
            for b2 in baselines:
                if b1 == b2:
                    row.append('-')
                else:
                    win_rate = (win_rates[b1][b2] / total_comparisons) * 100
                    row.append(f"{win_rate:.1f}%")
            matrix_data.append(row)
        
        return pd.DataFrame(matrix_data, index=baselines, columns=baselines)
    
    def save_plot(self, filename: str):
        """Unified plot saving"""
        path = f"{self.config.output_dir}/{filename}"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Saved: {filename}")
    
    def plot_f1_by_n(self, f1_df: pd.DataFrame):
        """Plot F1 score vs n for different baselines"""
        pred_models = f1_df['prediction_model'].unique()
        fig, axes = plt.subplots(1, len(pred_models), figsize=(6*len(pred_models), 6))
        if len(pred_models) == 1:
            axes = [axes]
        
        for i, pred_model in enumerate(pred_models):
            model_data = f1_df[f1_df['prediction_model'] == pred_model]
            for baseline in model_data['baseline'].unique():
                baseline_data = model_data[model_data['baseline'] == baseline].sort_values('n')
                axes[i].plot(baseline_data['n'], baseline_data['f1_score'],
                           marker='o', label=baseline, linewidth=2, markersize=6)
            
            axes[i].set_xlabel('Number of Training Examples (n)')
            axes[i].set_ylabel('F1 Score')
            axes[i].set_title(f'Prediction Model: {pred_model}')
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, 1)
        
        plt.tight_layout()
        self.save_plot('f1_by_n.png')
    
    def plot_f1_baseline_comparison(self, f1_df: pd.DataFrame):
        """Plot overall F1 score comparison using max F1 per baseline"""
        baseline_max = f1_df.groupby('baseline')['f1_score'].max().sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(baseline_max)), baseline_max.values,
                      color=sns.color_palette("husl", len(baseline_max)))
        
        plt.xlabel('Baseline Method')
        plt.ylabel('Max F1 Score')
        plt.title('Overall Max F1 Score Comparison Across Baselines')
        plt.xticks(range(len(baseline_max)), baseline_max.index, rotation=45, ha='right')
        
        for i, v in enumerate(baseline_max.values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        self.save_plot('f1_baseline_comparison.png')
    
    def plot_f1_efficiency_comparison(self, f1_df: pd.DataFrame):
        """Plot F1 efficiency comparison using max F1 per baseline"""
        pred_models = f1_df['prediction_model'].unique()
        fig, axes = plt.subplots(1, len(pred_models), figsize=(6*len(pred_models), 6))
        if len(pred_models) == 1:
            axes = [axes]
        
        for i, pred_model in enumerate(pred_models):
            model_data = f1_df[f1_df['prediction_model'] == pred_model]
            
            # Calculate max efficiency per baseline (max F1 / avg tokens for that max F1 setting)
            efficiency_data = []
            for baseline in model_data['baseline'].unique():
                baseline_data = model_data[model_data['baseline'] == baseline]
                # Find the n that gives max F1
                max_f1_row = baseline_data.loc[baseline_data['f1_score'].idxmax()]
                max_f1 = max_f1_row['f1_score']
                tokens_at_max = max_f1_row['total_samples'] if 'total_samples' in max_f1_row else 1  # Fallback
                efficiency = max_f1 / tokens_at_max if tokens_at_max > 0 else 0
                efficiency_data.append({
                    'baseline': baseline,
                    'max_f1': max_f1,
                    'efficiency': efficiency
                })
            
            efficiency_df_model = pd.DataFrame(efficiency_data).sort_values('max_f1', ascending=False)
            
            bars = axes[i].bar(range(len(efficiency_df_model)), efficiency_df_model['max_f1'],
                              color=sns.color_palette("husl", len(efficiency_df_model)))
            axes[i].set_xlabel('Baseline Method')
            axes[i].set_ylabel('Max F1 Score')
            axes[i].set_title(f'Prediction Model: {pred_model} - Max F1 Performance')
            axes[i].set_xticks(range(len(efficiency_df_model)))
            axes[i].set_xticklabels(efficiency_df_model['baseline'], rotation=45, ha='right')
            
            for j, v in enumerate(efficiency_df_model['max_f1']):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
            
            axes[i].grid(True, alpha=0.3, axis='y')
            axes[i].set_ylim(0, 1)
        
        plt.tight_layout()
        self.save_plot('f1_efficiency_comparison.png')
    
    def plot_f1_dataset_comparison(self, dataset_df: pd.DataFrame):
        """Plot dataset-specific F1 comparison"""
        datasets = sorted(dataset_df['dataset_name'].unique())
        baselines = sorted(dataset_df['baseline'].unique())
        
        datasets_per_row = 5
        num_rows = (len(datasets) + datasets_per_row - 1) // datasets_per_row
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(baselines)))
        baseline_colors = dict(zip(baselines, colors))
        
        fig, axes = plt.subplots(num_rows, datasets_per_row, figsize=(20, 4*num_rows))
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        elif datasets_per_row == 1:
            axes = axes.reshape(-1, 1)
        
        for i, dataset in enumerate(datasets):
            row, col = i // datasets_per_row, i % datasets_per_row
            ax = axes[row, col]
            
            dataset_data = dataset_df[dataset_df['dataset_name'] == dataset]
            baseline_max = dataset_data.groupby('baseline')['f1_score'].max().sort_values(ascending=False)
            
            bars = ax.bar(range(len(baseline_max)), baseline_max.values,
                         color=[baseline_colors[b] for b in baseline_max.index])
            
            ax.set_xlabel('Method')
            ax.set_ylabel('Max F1 Score')
            ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(baseline_max)))
            
            # Shortened labels
            short_labels = []
            for baseline in baseline_max.index:
                label_map = {
                    'naive': 'Naive',
                    'naive+icl': 'Naive+ICL',
                    'generated_instruction': 'Gen-Instr',
                    'generated_instruction+icl': 'Gen-Instr+ICL',
                    'generated_instruction2': 'Gen-Instr2',
                    'generated_instruction2+icl': 'Gen-Instr2+ICL'
                }
                short_labels.append(label_map.get(baseline, baseline[:10] + '...' if len(baseline) > 10 else baseline))
            
            ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1)
            
            for j, v in enumerate(baseline_max.values):
                ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Hide empty subplots
        for i in range(len(datasets), num_rows * datasets_per_row):
            row, col = i // datasets_per_row, i % datasets_per_row
            axes[row, col].set_visible(False)
        
        # Legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=baseline_colors[b], label=b) for b in baselines]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                  ncol=len(baselines), frameon=True)
        
        plt.suptitle('Dataset-Specific F1 Score Comparison by Method', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        self.save_plot('f1_dataset_comparison.png')
    
    def plot_f1_dataset_trends(self, dataset_df: pd.DataFrame):
        """Plot F1 trends across n values for each dataset and method"""
        # Filter out rows with NaN F1 values
        original_count = len(dataset_df)
        dataset_df = dataset_df.dropna(subset=['f1_score'])
        filtered_count = len(dataset_df)
        removed_count = original_count - filtered_count
        
        print(f"[INFO] Dataset trends plot: {original_count} total rows, removed {removed_count} rows with NaN F1, using {filtered_count} rows")
        
        if len(dataset_df) == 0:
            print("[WARNING] No valid F1 data for dataset trends plot")
            return
            
        n_values = sorted(dataset_df['n'].unique())
        baselines = sorted(dataset_df['baseline'].unique())
        datasets = sorted(dataset_df['dataset_name'].unique())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(baselines)))
        baseline_colors = dict(zip(baselines, colors))
        
        datasets_per_row = 5
        num_rows = (len(datasets) + datasets_per_row - 1) // datasets_per_row
        
        fig, axes = plt.subplots(num_rows, datasets_per_row, figsize=(20, 4*num_rows))
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        elif datasets_per_row == 1:
            axes = axes.reshape(-1, 1)
        
        for i, dataset in enumerate(datasets):
            row = i // datasets_per_row
            col = i % datasets_per_row
            ax = axes[row, col]
            
            for baseline in baselines:
                data = dataset_df[(dataset_df['dataset_name'] == dataset) & (dataset_df['baseline'] == baseline)]
                if len(data) > 0:
                    # Data is already grouped by n, so just plot directly
                    ax.plot(data['n'], data['f1_score'], marker='o', 
                           label=baseline, color=baseline_colors[baseline], linewidth=2)
            
            ax.set_title(dataset[:30] + ('...' if len(dataset) > 30 else ''), fontsize=10)
            ax.set_xlabel('n')
            ax.set_ylabel('F1 Score')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(n_values)
        
        # Hide unused subplots
        for i in range(len(datasets), num_rows * datasets_per_row):
            row = i // datasets_per_row
            col = i % datasets_per_row
            axes[row, col].set_visible(False)
        
        # Add legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(baselines), bbox_to_anchor=(0.5, -0.02))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        self.save_plot('f1_dataset_trends.png')
    
    def plot_f1_cross_dataset_comparison(self, plot_df: pd.DataFrame, baseline_f1_df: pd.DataFrame):
        """Simple scatter plot of F1 score vs tokens"""
        if len(plot_df) == 0:
            return
        
        plt.figure(figsize=(10, 6))
        
        available_baselines = plot_df['baseline'].unique()
        colors, markers = self.get_baseline_colors_and_markers(available_baselines)
        
        # Get naive baseline F1 score
        naive_f1 = None
        if 'naive' in baseline_f1_df['baseline'].unique():
            naive_data = baseline_f1_df[baseline_f1_df['baseline'] == 'naive']
            if len(naive_data) > 0:
                naive_f1 = naive_data['f1_score'].mean()
        
        for baseline in available_baselines:
            data = plot_df[plot_df['baseline'] == baseline]
            if len(data) > 0:
                color = colors[baseline]
                marker = markers[baseline]
                
                if baseline == 'naive':
                    y_values = [naive_f1] * len(data) if naive_f1 is not None else data['avg_absolute_improvement']
                else:
                    y_values = data['avg_absolute_improvement'] + (naive_f1 if naive_f1 is not None else 0)
                
                plt.scatter(data['avg_tokens'], y_values, c=color, marker=marker, 
                           s=100, label=baseline, alpha=0.7)
        
        plt.xlabel('Average Number of Tokens')
        plt.ylabel('F1 Score')
        plt.title('All Datasets: F1 Score vs Token Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if naive_f1 is not None:
            plt.axhline(y=naive_f1, color='blue', linestyle='--', alpha=0.5)
        plt.tight_layout()
        self.save_plot('f1_all_datasets_comparison.png')
    
    def plot_f1_win_rate_matrix(self, win_rate_df: pd.DataFrame):
        """Plot F1 win rate heatmap"""
        plt.figure(figsize=(12, 10))
        
        # Convert percentage strings to numeric for heatmap
        baselines = win_rate_df.index.tolist()
        heatmap_data = []
        for i, b1 in enumerate(baselines):
            row = []
            for j, b2 in enumerate(baselines):
                if i == j:
                    row.append(50.0)
                else:
                    val = win_rate_df.iloc[i, j]
                    row.append(float(val.rstrip('%')) if val != '-' else 50.0)
            heatmap_data.append(row)
        
        sns.heatmap(heatmap_data, xticklabels=baselines, yticklabels=baselines,
                   annot=True, fmt='.1f', cmap='RdYlGn_r', center=50,
                   cbar_kws={'label': 'Win Rate (%)'}, square=True)
        
        plt.title('Method Comparison Matrix: F1 Win Rate Across All Datasets\n(Shows % of times method in row beats method in column)')
        plt.xlabel('Method Being Compared Against')
        plt.ylabel('Method')
        
        # Mark diagonal
        for i in range(len(baselines)):
            plt.text(i + 0.5, i + 0.5, 'N/A', ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        self.save_plot('f1_win_rate_matrix.png')

    def plot_f1_max_win_rate_matrix(self, max_win_rate_df: pd.DataFrame):
        """Plot max F1 win rate heatmap"""
        plt.figure(figsize=(12, 10))
        
        # Convert percentage strings to numeric for heatmap
        baselines = max_win_rate_df.index.tolist()
        heatmap_data = []
        for i, b1 in enumerate(baselines):
            row = []
            for j, b2 in enumerate(baselines):
                if i == j:
                    row.append(50.0)
                else:
                    val = max_win_rate_df.iloc[i, j]
                    row.append(float(val.rstrip('%')) if val != '-' else 50.0)
            heatmap_data.append(row)
        
        sns.heatmap(heatmap_data, xticklabels=baselines, yticklabels=baselines,
                   annot=True, fmt='.1f', cmap='RdYlGn_r', center=50,
                   cbar_kws={'label': 'Max Win Rate (%)'}, square=True)
        
        plt.title('Method Comparison Matrix: MAX F1 Win Rate Across All Datasets\n(Shows % of times max F1 of method in row beats max F1 of method in column)')
        plt.xlabel('Method Being Compared Against')
        plt.ylabel('Method')
        
        # Mark diagonal
        for i in range(len(baselines)):
            plt.text(i + 0.5, i + 0.5, 'N/A', ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        self.save_plot('f1_max_win_rate_matrix.png')
    
    def create_small_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create smaller CSV with limited rows per setting"""
        key_columns = ['baseline', 'n', 'prediction_model', 'dataset_name', 'config_name']
        available_columns = [col for col in key_columns if col in df.columns]
        
        if available_columns:
            sampled_df = pd.DataFrame()
            for name, group in df.groupby(available_columns):
                sampled_group = group.head(self.config.max_rows_per_setting)
                sampled_df = pd.concat([sampled_df, sampled_group], ignore_index=True)
        else:
            sampled_df = df.head(self.config.max_rows_per_setting)
        
        output_path = f"{self.config.output_dir}/baseline_eval_results_small.csv"
        sampled_df.to_csv(output_path, index=False, encoding="utf-8")
        
        # Create ultra-clean version
        ultra_clean_path = f"{self.config.output_dir}/baseline_eval_results_small_ultra_clean.csv"
        print(f"[INFO] Creating ultra-clean version...")
        ultra_clean_success = ultra_clean_csv(output_path, ultra_clean_path)
        
        if ultra_clean_success:
            print(f"[INFO] ✅ Ultra-clean CSV created: baseline_eval_results_small_ultra_clean.csv")
        else:
            print(f"[INFO] ❌ Ultra-clean CSV creation failed")
        
        print(f"[INFO] Created small CSV with {len(sampled_df)} rows (max {self.config.max_rows_per_setting} per setting)")
        return sampled_df
    
    def print_f1_summary(self, metrics: Dict[str, pd.DataFrame]):
        """Print comprehensive F1 summary statistics"""
        f1_df = metrics['f1']
        dataset_df = metrics['dataset']
        
        print("="*60)
        print("BASELINE EVALUATION RESULTS SUMMARY (F1 SCORES)")
        print("="*60)
        print(f"Total predictions: {f1_df['total_samples'].sum()}")
        print(f"Datasets/configs: {len(dataset_df['dataset_name'].unique())}")
        print(f"Baselines tested: {len(f1_df['baseline'].unique())}")
        print(f"Prediction models: {len(f1_df['prediction_model'].unique())}")
        
        best_baseline = f1_df.groupby('baseline')['f1_score'].max().idxmax()
        best_f1 = f1_df.groupby('baseline')['f1_score'].max().max()
        print(f"Best baseline: {best_baseline} (max F1: {best_f1:.3f})")
        
        print("\nMax F1 Score by Baseline:")
        baseline_max = f1_df.groupby('baseline')['f1_score'].max().sort_values(ascending=False)
        for baseline, f1 in baseline_max.items():
            print(f"  {baseline}: {f1:.3f}")
        
        print("\nAverage F1 Score by N (for trends):")
        n_avg = f1_df.groupby('n')['f1_score'].mean().sort_index()
        for n, f1 in n_avg.items():
            print(f"  n={n}: {f1:.3f}")
    
    def run_f1_analysis(self):
        """Main F1 analysis pipeline"""
        print("[INFO] Loading and filtering data...")
        self.results_df = self.load_and_filter_data()
        
        if len(self.results_df) == 0:
            print("[ERROR] No data available after filtering. Exiting.")
            return
        
        print(f"[INFO] Loaded {len(self.results_df)} result rows")
        print(f"[INFO] Baselines: {sorted(self.results_df['baseline'].unique())}")
        print(f"[INFO] N values: {sorted(self.results_df['n'].unique())}")
        
        # Calculate F1 scores from predictions
        print("[INFO] Calculating F1 scores...")
        self.results_df = self.calculate_f1_from_predictions(self.results_df)
        
        if 'f1' not in self.results_df.columns or self.results_df['f1'].isna().all():
            print("[ERROR] F1 calculation failed. Cannot proceed with F1 analysis.")
            return
        
        # Calculate all metrics
        print("[INFO] Calculating F1 metrics...")
        metrics = self.calculate_f1_metrics()
        
        # Save full and small CSVs
        self.results_df.to_csv(f"{self.config.output_dir}/baseline_eval_results.csv", index=False, encoding="utf-8")
        self.create_small_csv(self.results_df)
        
        # Print summary
        self.print_f1_summary(metrics)
        
        # Generate all plots
        print("[INFO] Creating F1 visualizations...")
        self.plot_f1_by_n(metrics['f1'])
        self.plot_f1_baseline_comparison(metrics['f1'])
        self.plot_f1_efficiency_comparison(metrics['f1'])
        self.plot_f1_dataset_comparison(metrics['dataset'])
        self.plot_f1_dataset_trends(metrics['dataset'])
        
        # Cross-dataset analysis - use same fair filtered data
        if not self.config.no_cross_dataset_comparison:
            print("[INFO] Running cross-dataset F1 analysis...")
            # Get the same fair filtered data used for metrics
            fair_df = self.filter_complete_datasets(self.results_df)
            
            cross_dataset_df = self.calculate_cross_dataset_f1_metrics(fair_df)
            if cross_dataset_df is not None:
                self.plot_f1_cross_dataset_comparison(cross_dataset_df, metrics['f1'])
                cross_dataset_df.to_csv(f"{self.config.output_dir}/f1_all_datasets_absolute_comparison.csv", index=False)
                
                # F1 Win rate matrix
                win_rate_df = self.create_win_rate_matrix(fair_df)
                if win_rate_df is not None:
                    self.plot_f1_win_rate_matrix(win_rate_df)
                    win_rate_df.to_csv(f"{self.config.output_dir}/f1_win_rate_matrix.csv")
                
                # Max F1 win rate matrix (best F1 for each method)
                max_win_rate_df = self.create_max_win_rate_matrix(fair_df)
                if max_win_rate_df is not None:
                    self.plot_f1_max_win_rate_matrix(max_win_rate_df)
                    max_win_rate_df.to_csv(f"{self.config.output_dir}/f1_max_win_rate_matrix.csv")
        
        # Save all metric CSVs
        for name, df in metrics.items():
            df.to_csv(f"{self.config.output_dir}/f1_{name}_results.csv", index=False)
        
        print("[INFO] F1 analysis complete!")
        print(f"[INFO] All outputs saved to: {self.config.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Analyze baseline evaluation results using F1 scores")
    parser.add_argument("--input_path", type=str, required=True, help="Input results directory path")
    parser.add_argument("--output_dir", type=str, default="out/baseline_f1", help="Output directory path")
    parser.add_argument("--baselines", type=str, nargs='+', help="Specify subset of baselines to analyze")
    parser.add_argument("--max_rows_per_setting", type=int, default=10, help="Maximum rows per setting in small CSV")
    parser.add_argument("--no_cross_dataset_comparison", action="store_true", help="Disable cross-dataset comparison")
    
    args = parser.parse_args()
    
    config = F1AnalysisConfig(
        input_path=args.input_path,
        output_dir=args.output_dir,
        baselines=args.baselines,
        max_rows_per_setting=args.max_rows_per_setting,
        no_cross_dataset_comparison=args.no_cross_dataset_comparison
    )
    
    analyzer = F1BaselineAnalyzer(config)
    analyzer.run_f1_analysis()


if __name__ == "__main__":
    main()