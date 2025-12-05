#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOT Dataset Bounding Box Analysis Tool
Complete version with normal distribution analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import datetime
import os
from scipy import stats
from scipy.stats import normaltest, shapiro, anderson, kstest
import warnings
warnings.filterwarnings('ignore')

class MOTBboxAnalyzer:
    def __init__(self, file_path):
        """Initialize MOT dataset analyzer"""
        self.file_path = Path(file_path)
        self.data = None
        self.stats = {}
        self.normality_results = {}
        self.normal_regions = {}
        
        # Create results directory with timestamp
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = Path(__file__).parent if '__file__' in globals() else Path('.')
        self.output_dir = script_dir / "results" / f"analysis_{current_time}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure matplotlib for Docker environment
        plt.switch_backend('Agg')
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        sns.set_style("whitegrid")
        
        print(f"Output directory: {self.output_dir}")
        
    def load_data(self):
        """Load MOT format data"""
        print(f"Loading data from: {self.file_path}")
        
        columns = ['frame_id', 'track_id', 'x', 'y', 'width', 'height', 
                  'conf', 'x_3d', 'y_3d', 'z_3d']
        
        try:
            self.data = pd.read_csv(self.file_path, header=None, names=columns)
            
            # Filter valid data
            valid_mask = (self.data['width'] > 0) & (self.data['height'] > 0)
            self.data = self.data[valid_mask].copy()
            
            # Calculate additional metrics
            self.data['aspect_ratio'] = self.data['width'] / self.data['height']
            self.data['area'] = self.data['width'] * self.data['height']
            
            print(f"Successfully loaded {len(self.data)} valid bounding boxes")
            print(f"Data covers {self.data['frame_id'].nunique()} frames")
            print(f"Contains {self.data['track_id'].nunique()} different tracks")
            
        except Exception as e:
            print(f"Failed to load data: {e}")
            raise
    
    def calculate_statistics(self):
        """Calculate statistical metrics"""
        print("Calculating statistical metrics...")
        
        metrics = ['width', 'height', 'aspect_ratio', 'area']
        
        for metric in metrics:
            data_series = self.data[metric]
            self.stats[metric] = {
                'count': len(data_series),
                'mean': data_series.mean(),
                'std': data_series.std(),
                'min': data_series.min(),
                'max': data_series.max(),
                'median': data_series.median(),
                'q25': data_series.quantile(0.25),
                'q75': data_series.quantile(0.75),
                'iqr': data_series.quantile(0.75) - data_series.quantile(0.25),
                'skewness': data_series.skew(),
                'kurtosis': data_series.kurtosis()
            }
    
    def print_statistics(self):
        """Print statistical results"""
        print("\n" + "="*80)
        print("MOT Dataset Bounding Box Statistical Analysis Report")
        print("="*80)
        
        print(f"Data file: {self.file_path}")
        print(f"Total bounding boxes: {len(self.data):,}")
        print(f"Total frames: {self.data['frame_id'].nunique():,}")
        print(f"Total tracks: {self.data['track_id'].nunique():,}")
        print(f"Average boxes per frame: {len(self.data) / self.data['frame_id'].nunique():.2f}")
        
        metrics_info = {
            'width': 'Width',
            'height': 'Height', 
            'aspect_ratio': 'Aspect Ratio',
            'area': 'Area'
        }
        
        for metric, name in metrics_info.items():
            stats_data = self.stats[metric]
            print(f"\n{name} Statistics:")
            print(f"  Mean: {stats_data['mean']:.3f}")
            print(f"  Std Dev: {stats_data['std']:.3f}")
            print(f"  Min: {stats_data['min']:.3f}")
            print(f"  Max: {stats_data['max']:.3f}")
            print(f"  Median: {stats_data['median']:.3f}")
            print(f"  25th Percentile: {stats_data['q25']:.3f}")
            print(f"  75th Percentile: {stats_data['q75']:.3f}")
            print(f"  IQR: {stats_data['iqr']:.3f}")
            print(f"  Skewness: {stats_data['skewness']:.3f}")
            print(f"  Kurtosis: {stats_data['kurtosis']:.3f}")
    
    def test_normality(self):
        """Test normality of distributions"""
        print("\nNormality Tests:")
        print("="*60)
        
        metrics = ['width', 'height', 'aspect_ratio', 'area']
        
        for metric in metrics:
            data = self.data[metric].dropna()
            
            if len(data) < 8:
                continue
                
            print(f"\n{metric.upper()} Normality Tests:")
            
            try:
                # Shapiro-Wilk test
                if len(data) <= 5000:
                    shapiro_stat, shapiro_p = shapiro(data)
                    print(f"  Shapiro-Wilk: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.6f}")
                else:
                    shapiro_stat, shapiro_p = None, None
                    print(f"  Shapiro-Wilk: Skipped (sample too large)")
                
                # D'Agostino's normality test
                dagostino_stat, dagostino_p = normaltest(data)
                print(f"  D'Agostino: statistic={dagostino_stat:.4f}, p-value={dagostino_p:.6f}")
                
                # Anderson-Darling test
                anderson_result = anderson(data, dist='norm')
                print(f"  Anderson-Darling: statistic={anderson_result.statistic:.4f}")
                
                # Kolmogorov-Smirnov test
                normalized_data = (data - data.mean()) / data.std()
                ks_stat, ks_p = kstest(normalized_data, 'norm')
                print(f"  Kolmogorov-Smirnov: statistic={ks_stat:.4f}, p-value={ks_p:.6f}")
                
                # Store results
                self.normality_results[metric] = {
                    'shapiro': {'statistic': shapiro_stat, 'p_value': shapiro_p},
                    'dagostino': {'statistic': dagostino_stat, 'p_value': dagostino_p},
                    'anderson': anderson_result,
                    'ks': {'statistic': ks_stat, 'p_value': ks_p}
                }
                
                # Interpretation
                alpha = 0.05
                if dagostino_p > alpha:
                    print(f"  --> {metric} appears to follow normal distribution (p > {alpha})")
                else:
                    print(f"  --> {metric} does NOT follow normal distribution (p <= {alpha})")
                    
            except Exception as e:
                print(f"  Error in normality test for {metric}: {e}")
                continue
    
    def analyze_normal_distribution_regions(self):
        """Analyze regions within normal distribution bounds"""
        print("\nNormal Distribution Region Analysis:")
        print("="*60)
        
        metrics = ['width', 'height', 'aspect_ratio', 'area']
        
        for metric in metrics:
            data = self.data[metric]
            mean_val = data.mean()
            std_val = data.std()
            
            print(f"\n{metric.upper()} Normal Distribution Regions:")
            print(f"  Mean: {mean_val:.3f}, Std: {std_val:.3f}")
            
            # Define regions based on standard deviations
            regions = {
                '1_sigma': (mean_val - std_val, mean_val + std_val),
                '2_sigma': (mean_val - 2*std_val, mean_val + 2*std_val),
                '3_sigma': (mean_val - 3*std_val, mean_val + 3*std_val)
            }
            
            region_stats = {}
            
            for region_name, (lower, upper) in regions.items():
                # Count data points in region
                in_region = data[(data >= lower) & (data <= upper)]
                percentage = len(in_region) / len(data) * 100
                
                # Expected percentages for normal distribution
                expected_pct = {
                    '1_sigma': 68.27,
                    '2_sigma': 95.45,
                    '3_sigma': 99.73
                }
                
                print(f"  {region_name.replace('_', '-')} region [{lower:.3f}, {upper:.3f}]:")
                print(f"    Actual: {percentage:.2f}% ({len(in_region)} samples)")
                print(f"    Expected: {expected_pct[region_name]:.2f}%")
                print(f"    Difference: {percentage - expected_pct[region_name]:+.2f}%")
                
                region_stats[region_name] = {
                    'range': (lower, upper),
                    'count': len(in_region),
                    'percentage': percentage,
                    'expected': expected_pct[region_name],
                    'difference': percentage - expected_pct[region_name]
                }
            
            # Outlier analysis
            outliers_lower = data[data < mean_val - 3*std_val]
            outliers_upper = data[data > mean_val + 3*std_val]
            total_outliers = len(outliers_lower) + len(outliers_upper)
            
            print(f"  Outliers (beyond 3-sigma):")
            print(f"    Lower outliers: {len(outliers_lower)} ({len(outliers_lower)/len(data)*100:.2f}%)")
            print(f"    Upper outliers: {len(outliers_upper)} ({len(outliers_upper)/len(data)*100:.2f}%)")
            print(f"    Total outliers: {total_outliers} ({total_outliers/len(data)*100:.2f}%)")
            
            self.normal_regions[metric] = {
                'regions': region_stats,
                'outliers': {
                    'lower': len(outliers_lower),
                    'upper': len(outliers_upper),
                    'total': total_outliers,
                    'percentage': total_outliers/len(data)*100
                }
            }
    
    def plot_distributions(self):
        """Plot distribution charts"""
        print("\nGenerating distribution charts...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MOT Dataset Bounding Box Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Width distribution
        axes[0, 0].hist(self.data['width'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(self.stats['width']['mean'], color='red', linestyle='--', 
                          label=f'Mean: {self.stats["width"]["mean"]:.2f}')
        axes[0, 0].axvline(self.stats['width']['median'], color='orange', linestyle='--', 
                          label=f'Median: {self.stats["width"]["median"]:.2f}')
        axes[0, 0].set_title('Bounding Box Width Distribution')
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Height distribution
        axes[0, 1].hist(self.data['height'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(self.stats['height']['mean'], color='red', linestyle='--', 
                          label=f'Mean: {self.stats["height"]["mean"]:.2f}')
        axes[0, 1].axvline(self.stats['height']['median'], color='orange', linestyle='--', 
                          label=f'Median: {self.stats["height"]["median"]:.2f}')
        axes[0, 1].set_title('Bounding Box Height Distribution')
        axes[0, 1].set_xlabel('Height (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Aspect ratio distribution
        axes[1, 0].hist(self.data['aspect_ratio'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 0].axvline(self.stats['aspect_ratio']['mean'], color='red', linestyle='--', 
                          label=f'Mean: {self.stats["aspect_ratio"]["mean"]:.3f}')
        axes[1, 0].axvline(self.stats['aspect_ratio']['median'], color='orange', linestyle='--', 
                          label=f'Median: {self.stats["aspect_ratio"]["median"]:.3f}')
        axes[1, 0].set_title('Bounding Box Aspect Ratio Distribution')
        axes[1, 0].set_xlabel('Aspect Ratio')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Area distribution (log scale)
        axes[1, 1].hist(np.log10(self.data['area']), bins=50, alpha=0.7, color='plum', edgecolor='black')
        axes[1, 1].axvline(np.log10(self.stats['area']['mean']), color='red', linestyle='--', 
                          label=f'Mean: {self.stats["area"]["mean"]:.1f}')
        axes[1, 1].axvline(np.log10(self.stats['area']['median']), color='orange', linestyle='--', 
                          label=f'Median: {self.stats["area"]["median"]:.1f}')
        axes[1, 1].set_title('Bounding Box Area Distribution (Log Scale)')
        axes[1, 1].set_xlabel('log10(Area)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.output_dir / "distributions.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Distribution chart saved to: {output_file}")
        plt.close()
    
    def plot_normal_distribution_analysis(self):
        """Plot normal distribution analysis with Q-Q plots"""
        print("Generating normal distribution analysis charts...")
        
        metrics = ['width', 'height', 'aspect_ratio', 'area']
        
        # Create Q-Q plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Q-Q Plots for Normality Assessment', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            row, col = i // 2, i % 2
            data = self.data[metric].dropna()
            
            # Q-Q plot
            stats.probplot(data, dist="norm", plot=axes[row, col])
            axes[row, col].set_title(f'{metric.replace("_", " ").title()} Q-Q Plot')
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "qq_plots.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Q-Q plots saved to: {output_file}")
        plt.close()
        
        # Create normal distribution overlay plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Distribution vs Normal Distribution Comparison', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            row, col = i // 2, i % 2
            data = self.data[metric].dropna()
            
            # Plot histogram
            n, bins, patches = axes[row, col].hist(data, bins=50, density=True, alpha=0.7, 
                                                  color='skyblue', edgecolor='black', 
                                                  label='Actual Data')
            
            # Plot normal distribution overlay
            mean_val, std_val = data.mean(), data.std()
            x = np.linspace(data.min(), data.max(), 100)
            normal_curve = stats.norm.pdf(x, mean_val, std_val)
            axes[row, col].plot(x, normal_curve, 'r-', linewidth=2, label='Normal Distribution')
            
            # Mark sigma regions
            for sigma in [1, 2, 3]:
                lower = mean_val - sigma * std_val
                upper = mean_val + sigma * std_val
                if sigma == 1:
                    axes[row, col].axvspan(lower, upper, alpha=0.1, color='green', 
                                         label=f'{sigma}-sigma region')
                else:
                    axes[row, col].axvspan(lower, upper, alpha=0.05, color='green')
            
            axes[row, col].axvline(mean_val, color='red', linestyle='--', alpha=0.7, label='Mean')
            axes[row, col].set_title(f'{metric.replace("_", " ").title()} Distribution')
            axes[row, col].set_xlabel(metric.replace("_", " ").title())
            axes[row, col].set_ylabel('Density')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "normal_distribution_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Normal distribution comparison saved to: {output_file}")
        plt.close()
    
    def plot_sigma_regions_detailed(self):
        """Plot detailed sigma regions analysis"""
        print("Generating detailed sigma regions analysis...")
        
        metrics = ['width', 'height', 'aspect_ratio', 'area']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Sigma Regions Analysis', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            row, col = i // 2, i % 2
            data = self.data[metric].dropna()
            mean_val, std_val = data.mean(), data.std()
            
            # Create histogram with sigma region coloring
            n, bins, patches = axes[row, col].hist(data, bins=60, alpha=0.7, edgecolor='black')
            
            # Color patches based on sigma regions
            for j, patch in enumerate(patches):
                bin_center = (bins[j] + bins[j+1]) / 2
                distance_from_mean = abs(bin_center - mean_val) / std_val
                
                if distance_from_mean <= 1:
                    patch.set_facecolor('green')
                    patch.set_alpha(0.8)
                elif distance_from_mean <= 2:
                    patch.set_facecolor('yellow')
                    patch.set_alpha(0.7)
                elif distance_from_mean <= 3:
                    patch.set_facecolor('orange')
                    patch.set_alpha(0.6)
                else:
                    patch.set_facecolor('red')
                    patch.set_alpha(0.5)
            
            # Add vertical lines for sigma boundaries
            for sigma in [1, 2, 3]:
                axes[row, col].axvline(mean_val - sigma * std_val, color='black', 
                                     linestyle='--', alpha=0.6, linewidth=1)
                axes[row, col].axvline(mean_val + sigma * std_val, color='black', 
                                     linestyle='--', alpha=0.6, linewidth=1)
            
            # Add mean line
            axes[row, col].axvline(mean_val, color='red', linestyle='-', linewidth=2, label='Mean')
            
            # Add statistics text
            if metric in self.normal_regions:
                region_info = self.normal_regions[metric]['regions']
                text_info = f"1σ: {region_info['1_sigma']['percentage']:.1f}%\n"
                text_info += f"2σ: {region_info['2_sigma']['percentage']:.1f}%\n"
                text_info += f"3σ: {region_info['3_sigma']['percentage']:.1f}%\n"
                text_info += f"Outliers: {self.normal_regions[metric]['outliers']['percentage']:.1f}%"
                
                axes[row, col].text(0.02, 0.98, text_info, transform=axes[row, col].transAxes,
                                  verticalalignment='top', bbox=dict(boxstyle='round', 
                                  facecolor='white', alpha=0.8), fontsize=9)
            
            axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
            axes[row, col].set_xlabel(metric.replace("_", " ").title())
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.8, label='1-sigma (68.27%)'),
            Patch(facecolor='yellow', alpha=0.7, label='2-sigma (95.45%)'),
            Patch(facecolor='orange', alpha=0.6, label='3-sigma (99.73%)'),
            Patch(facecolor='red', alpha=0.5, label='Outliers (>3-sigma)')
        ]
        fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=4)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        output_file = self.output_dir / "sigma_regions_detailed.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Detailed sigma regions analysis saved to: {output_file}")
        plt.close()
    
    def generate_concentration_analysis(self):
        """Generate concentration analysis summary table"""
        print("Generating concentration analysis...")
        
        metrics = ['width', 'height', 'aspect_ratio', 'area']
        
        # Create summary table for concentration
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Table: Sigma regions comparison
        table_data = []
        headers = ['Metric', '1-Sigma Actual', '1-Sigma Expected', 'Difference', 
                  '2-Sigma Actual', '2-Sigma Expected', 'Difference']
        
        for metric in metrics:
            if metric in self.normal_regions:
                regions = self.normal_regions[metric]['regions']
                row = [
                    metric.replace('_', ' ').title(),
                    f"{regions['1_sigma']['percentage']:.2f}%",
                    f"{regions['1_sigma']['expected']:.2f}%",
                    f"{regions['1_sigma']['difference']:+.2f}%",
                    f"{regions['2_sigma']['percentage']:.2f}%",
                    f"{regions['2_sigma']['expected']:.2f}%",
                    f"{regions['2_sigma']['difference']:+.2f}%"
                ]
                table_data.append(row)
        
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2.0)
        ax.set_title('Sigma Regions: Actual vs Expected (Normal Distribution)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        output_file = self.output_dir / "concentration_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Concentration analysis saved to: {output_file}")
        plt.close()
    
    def save_statistics_to_file(self):
        """Save statistical results to file"""
        output_file = self.output_dir / "statistics.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("MOT Dataset Bounding Box Statistical Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Data file: {self.file_path}\n")
            f.write(f"Analysis time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total bounding boxes: {len(self.data):,}\n")
            f.write(f"Total frames: {self.data['frame_id'].nunique():,}\n")
            f.write(f"Total tracks: {self.data['track_id'].nunique():,}\n")
            f.write(f"Average boxes per frame: {len(self.data) / self.data['frame_id'].nunique():.2f}\n\n")
            
            # Basic statistics
            metrics_info = {
                'width': 'Width',
                'height': 'Height', 
                'aspect_ratio': 'Aspect Ratio',
                'area': 'Area'
            }
            
            for metric, name in metrics_info.items():
                stats_data = self.stats[metric]
                f.write(f"{name} Statistics:\n")
                f.write(f"  Mean: {stats_data['mean']:.3f}\n")
                f.write(f"  Standard Deviation: {stats_data['std']:.3f}\n")
                f.write(f"  Minimum: {stats_data['min']:.3f}\n")
                f.write(f"  Maximum: {stats_data['max']:.3f}\n")
                f.write(f"  Median: {stats_data['median']:.3f}\n")
                f.write(f"  25th Percentile: {stats_data['q25']:.3f}\n")
                f.write(f"  75th Percentile: {stats_data['q75']:.3f}\n")
                f.write(f"  Interquartile Range: {stats_data['iqr']:.3f}\n")
                f.write(f"  Skewness: {stats_data['skewness']:.3f}\n")
                f.write(f"  Kurtosis: {stats_data['kurtosis']:.3f}\n\n")
            
            # Normality test results
            if self.normality_results:
                f.write("Normality Test Results:\n")
                f.write("-" * 30 + "\n")
                for metric, results in self.normality_results.items():
                    f.write(f"\n{metric.upper()}:\n")
                    if results['shapiro']['p_value'] is not None:
                        f.write(f"  Shapiro-Wilk: p-value = {results['shapiro']['p_value']:.6f}\n")
                    f.write(f"  D'Agostino: p-value = {results['dagostino']['p_value']:.6f}\n")
                    f.write(f"  Anderson-Darling: statistic = {results['anderson'].statistic:.4f}\n")
                    f.write(f"  Kolmogorov-Smirnov: p-value = {results['ks']['p_value']:.6f}\n")
                    
                    # Interpretation
                    if results['dagostino']['p_value'] > 0.05:
                        f.write(f"  Result: Follows normal distribution (p > 0.05)\n")
                    else:
                        f.write(f"  Result: Does NOT follow normal distribution (p <= 0.05)\n")
                f.write("\n")
            
            # Normal distribution region analysis
            if self.normal_regions:
                f.write("Normal Distribution Region Analysis:\n")
                f.write("-" * 40 + "\n")
                for metric, data in self.normal_regions.items():
                    f.write(f"\n{metric.upper()}:\n")
                    for region, stats_data in data['regions'].items():
                        f.write(f"  {region.replace('_', '-')} region:\n")
                        f.write(f"    Range: [{stats_data['range'][0]:.3f}, {stats_data['range'][1]:.3f}]\n")
                        f.write(f"    Actual: {stats_data['percentage']:.2f}% ({stats_data['count']} samples)\n")
                        f.write(f"    Expected: {stats_data['expected']:.2f}%\n")
                        f.write(f"    Difference: {stats_data['difference']:+.2f}%\n")
                    
                    f.write(f"  Outliers (beyond 3-sigma):\n")
                    f.write(f"    Lower: {data['outliers']['lower']}\n")
                    f.write(f"    Upper: {data['outliers']['upper']}\n")
                    f.write(f"    Total: {data['outliers']['total']} ({data['outliers']['percentage']:.2f}%)\n")
                f.write("\n")
        
        print(f"Statistics saved to: {output_file}")
    
    def save_csv_summary(self):
        """Save summary data as CSV"""
        # Basic statistics CSV
        summary_data = {
            'metric': ['width', 'height', 'aspect_ratio', 'area'],
            'mean': [self.stats[m]['mean'] for m in ['width', 'height', 'aspect_ratio', 'area']],
            'std': [self.stats[m]['std'] for m in ['width', 'height', 'aspect_ratio', 'area']],
            'min': [self.stats[m]['min'] for m in ['width', 'height', 'aspect_ratio', 'area']],
            'max': [self.stats[m]['max'] for m in ['width', 'height', 'aspect_ratio', 'area']],
            'median': [self.stats[m]['median'] for m in ['width', 'height', 'aspect_ratio', 'area']],
            'q25': [self.stats[m]['q25'] for m in ['width', 'height', 'aspect_ratio', 'area']],
            'q75': [self.stats[m]['q75'] for m in ['width', 'height', 'aspect_ratio', 'area']]
        }
        
        summary_df = pd.DataFrame(summary_data)
        output_file = self.output_dir / "summary_statistics.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"Summary statistics CSV saved to: {output_file}")
        
        # Normal distribution analysis CSV
        if self.normal_regions:
            normal_analysis_data = []
            for metric in ['width', 'height', 'aspect_ratio', 'area']:
                if metric in self.normal_regions:
                    regions = self.normal_regions[metric]['regions']
                    outliers = self.normal_regions[metric]['outliers']
                    
                    normal_analysis_data.append({
                        'metric': metric,
                        '1_sigma_actual': regions['1_sigma']['percentage'],
                        '1_sigma_expected': regions['1_sigma']['expected'],
                        '1_sigma_difference': regions['1_sigma']['difference'],
                        '2_sigma_actual': regions['2_sigma']['percentage'],
                        '2_sigma_expected': regions['2_sigma']['expected'],
                        '2_sigma_difference': regions['2_sigma']['difference'],
                        '3_sigma_actual': regions['3_sigma']['percentage'],
                        '3_sigma_expected': regions['3_sigma']['expected'],
                        '3_sigma_difference': regions['3_sigma']['difference'],
                        'outliers_count': outliers['total'],
                        'outliers_percentage': outliers['percentage']
                    })
            
            if normal_analysis_data:
                normal_df = pd.DataFrame(normal_analysis_data)
                normal_output_file = self.output_dir / "normal_distribution_analysis.csv"
                normal_df.to_csv(normal_output_file, index=False)
                print(f"Normal distribution analysis CSV saved to: {normal_output_file}")
    
    def run_complete_analysis(self):
        """Run complete analysis"""
        print("Starting MOT dataset analysis...")
        print(f"Output directory: {self.output_dir}")
        
        try:
            # Core analysis
            self.load_data()
            self.calculate_statistics()
            self.print_statistics()
            
            # Normal distribution analysis
            self.test_normality()
            self.analyze_normal_distribution_regions()
            
            # Save text results
            self.save_statistics_to_file()
            self.save_csv_summary()
            
            # Generate visualizations
            self.plot_distributions()
            self.plot_normal_distribution_analysis()
            self.plot_sigma_regions_detailed()
            self.generate_concentration_analysis()
            
            print(f"\nAnalysis complete! All results saved to: {self.output_dir}")
            print(f"Generated files:")
            for file in sorted(self.output_dir.glob("*")):
                print(f"  - {file.name}")
                
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='MOT Dataset Bounding Box Analysis Tool')
    parser.add_argument('file_path', help='Path to MOT format file')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File '{args.file_path}' not found!")
        return
    
    # Create analyzer and run analysis
    analyzer = MOTBboxAnalyzer(args.file_path)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # No command line arguments - use example file or prompt
        example_files = ["paste.txt", "data.txt", "mot_data.txt"]
        found_file = None
        
        for file_name in example_files:
            if os.path.exists(file_name):
                found_file = file_name
                break
        
        if found_file:
            print(f"No arguments provided. Using example file: {found_file}")
            analyzer = MOTBboxAnalyzer(found_file)
            analyzer.run_complete_analysis()
        else:
            print("MOT Dataset Bounding Box Analysis Tool")
            print("="*50)
            print("Usage: python mot_analyzer.py <path_to_mot_file>")
            print("\nExample:")
            print("  python mot_analyzer.py your_mot_file.txt")
            print("\nOr place your MOT file as 'paste.txt' in the same directory and run without arguments.")
            print("\nMOT format expected: frame_id,track_id,x,y,width,height,conf,x,y,z")
    else:
        main()


# End of file