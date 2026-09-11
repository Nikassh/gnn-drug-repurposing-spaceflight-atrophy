import unittest
import pandas as pd
from spaceflight_gnn.data.deg_parser import DirectionAwareDEGParser

class TestGNNPipeline(unittest.TestCase):
    def test_deg_parser_orientation(self):
        parser = DirectionAwareDEGParser(q_value_threshold=0.05)
        data = pd.DataFrame({
            'contrast': ['Control v Flight', 'Flight v Control'],
            'logFC': [1.5, 1.5],
            'qval': [0.01, 0.01]
        })
        parsed = parser.parse_contrast(data, 'contrast', 'logFC', 'qval')
        self.assertEqual(parsed.iloc[0]['logFC'], -1.5)
        self.assertEqual(parsed.iloc[1]['logFC'], 1.5)

if __name__ == '__main__':
    unittest.main()
