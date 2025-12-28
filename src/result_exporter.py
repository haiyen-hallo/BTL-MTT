import os
import pandas as pd

class ResultExporter:
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_cau1(self, accepted, rejected, total, params):
        df = pd.DataFrame([{
            'accepted': accepted,
            'rejected': rejected,
            'total': total,
            'k_paths': params.get('k'),
            'capacity': params.get('cap')
        }])
        df.to_csv(os.path.join(self.output_dir, 'cau1_summary.csv'), index=False)

    def save_cau2(self, df_detail):
        df_detail.to_csv(os.path.join(self.output_dir, 'cau2_accepted_detail.csv'), index=False)

    def save_cau3(self, count, df3, stats):
        df_stats = pd.DataFrame([stats])
        df_stats['high_utilization_links'] = count
        df3.to_csv(os.path.join(self.output_dir, 'cau3_high_links.csv'), index=False)
        df_stats.to_csv(os.path.join(self.output_dir, 'cau3_stats.csv'), index=False)

    def save_cau4(self, result, accepted_fcfs, total, params):
        df = pd.DataFrame([{
            'accepted_cau4': result.get('accepted'),
            'pending': result.get('pending'),
            'accepted_cau1': accepted_fcfs,
            'total': total,
            'k_paths': params.get('k'),
            'capacity': params.get('cap')
        }])
        df.to_csv(os.path.join(self.output_dir, 'cau4_summary.csv'), index=False)

    def save_best_demands(self, df):
        df.to_csv(os.path.join(self.output_dir, 'best_demands.csv'), index=False)
